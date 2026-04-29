from typing import Dict, Optional
import os

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from sfm_pipeline.depth_models.base import BaseDepthModel


class DepthProModel(BaseDepthModel):
    """
    Apple Depth Pro wrapper.

    This version reduces the input resolution before inference to avoid CUDA OOM.
    The predicted depth is resized back to the original image size.
    """

    def __init__(
        self,
        device,
        logger=None,
        input_size: int = 384,
        force_cpu: bool = False,
    ):
        self.logger = logger
        self.input_size = int(input_size)
        self.force_cpu = bool(force_cpu)

        if self.force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = device

        try:
            import depth_pro
        except ImportError as exc:
            raise ImportError(
                "Could not import depth_pro. "
                "Check PYTHONPATH for ~/robot_description/external/ml-depth-pro/src"
            ) from exc

        self.depth_pro = depth_pro

        if self.logger is not None:
            self.logger.info(
                f"Loading Depth Pro on [{self.device.type.upper()}], "
                f"input_size={self.input_size}, force_cpu={self.force_cpu}"
            )

        old_cwd = os.getcwd()

        try:
            depth_pro_root = os.path.expanduser(
                "~/robot_description/external/ml-depth-pro"
            )
            os.chdir(depth_pro_root)

            self.model, self.transform = self.depth_pro.create_model_and_transforms()

        finally:
            os.chdir(old_cwd)

        self.model.to(self.device)
        self.model.eval()

        if self.logger is not None:
            self.logger.info("Depth Pro loaded.")

    def predict(
        self,
        rgb_image: np.ndarray,
        camera_intrinsics: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        original_h, original_w = rgb_image.shape[:2]

        resized_rgb, scale = self.resize_keep_aspect(rgb_image)

        pil_img = PILImage.fromarray(resized_rgb)

        image_tensor = self.transform(pil_img)
        image_tensor = image_tensor.to(self.device)

        f_px_tensor = None

        if camera_intrinsics is not None:
            fx = camera_intrinsics.get("fx", None)
            fy = camera_intrinsics.get("fy", None)

            if fx is not None and fy is not None and fx > 0.0 and fy > 0.0:
                f_px = float((fx + fy) * 0.5)
                f_px = f_px * scale

                f_px_tensor = torch.tensor(
                    [f_px],
                    dtype=torch.float32,
                    device=self.device
                )

        try:
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        prediction = self.run_inference(image_tensor, f_px_tensor)
                else:
                    prediction = self.run_inference(image_tensor, f_px_tensor)

        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise RuntimeError(
                "Depth Pro CUDA out of memory. "
                "Try smaller depth_pro_input_size, for example 320 or 256, "
                "or run with depth_pro_force_cpu:=true."
            ) from exc

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise RuntimeError(
                    "Depth Pro CUDA out of memory. "
                    "Try smaller depth_pro_input_size, for example 320 or 256, "
                    "or run with depth_pro_force_cpu:=true."
                ) from exc

            raise

        depth_map = self.extract_depth_from_prediction(prediction)

        if torch.is_tensor(depth_map):
            depth_map = depth_map.detach().float().cpu().numpy()

        depth_map = np.asarray(depth_map).squeeze().astype(np.float32)

        if depth_map.ndim != 2:
            raise RuntimeError(
                f"Depth Pro output has invalid shape: {depth_map.shape}"
            )

        if depth_map.shape[0] != original_h or depth_map.shape[1] != original_w:
            depth_map = cv2.resize(
                depth_map,
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)

        return depth_map

    def run_inference(self, image_tensor, f_px_tensor):
        if f_px_tensor is not None:
            try:
                return self.model.infer(image_tensor, f_px=f_px_tensor)
            except TypeError:
                return self.model.infer(image_tensor)

        return self.model.infer(image_tensor)

    def extract_depth_from_prediction(self, prediction):
        if isinstance(prediction, dict):
            if "depth" in prediction:
                return prediction["depth"]

            if "metric_depth" in prediction:
                return prediction["metric_depth"]

            if "predicted_depth" in prediction:
                return prediction["predicted_depth"]

            raise RuntimeError(
                f"Depth Pro prediction dictionary has no depth key. "
                f"Available keys: {list(prediction.keys())}"
            )

        if torch.is_tensor(prediction):
            return prediction

        if isinstance(prediction, np.ndarray):
            return prediction

        raise RuntimeError(
            f"Unsupported Depth Pro prediction type: {type(prediction)}"
        )

    def resize_keep_aspect(self, rgb_image: np.ndarray):
        h, w = rgb_image.shape[:2]

        max_side = max(h, w)

        if self.input_size <= 0 or max_side <= self.input_size:
            return rgb_image, 1.0

        scale = float(self.input_size) / float(max_side)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(
            rgb_image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        return resized, scale