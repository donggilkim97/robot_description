from typing import Dict, Optional

import cv2
import numpy as np
import torch

from sfm_pipeline.depth_models.base import BaseDepthModel


class DepthAnythingV2MetricModel(BaseDepthModel):
    """
    Depth Anything V2 Metric wrapper.

    Expected input:
        RGB image from ROS/cv_bridge

    Output:
        Metric depth map in metres, if a metric checkpoint is used.

    Notes:
        The official Depth Anything V2 infer_image() example expects an OpenCV BGR image.
        This wrapper receives RGB, so it converts RGB to BGR before inference.
    """

    MODEL_CONFIGS = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    def __init__(
        self,
        device,
        checkpoint_path: str,
        encoder: str = "vitb",
        max_depth: float = 20.0,
        logger=None,
    ):
        self.device = device
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.encoder = encoder.lower().strip()
        self.max_depth = float(max_depth)

        if self.encoder not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported Depth Anything V2 encoder: {encoder}. "
                f"Use one of: {list(self.MODEL_CONFIGS.keys())}"
            )

        if not self.checkpoint_path:
            raise ValueError(
                "Depth Anything V2 checkpoint path is empty. "
                "Please set parameter depth_anything_checkpoint."
            )

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise ImportError(
                "Could not import depth_anything_v2. "
                "Clone Depth-Anything-V2 and add it to PYTHONPATH."
            ) from exc

        if self.logger is not None:
            self.logger.info(
                f"Loading Depth Anything V2 Metric: "
                f"encoder={self.encoder}, max_depth={self.max_depth}, "
                f"checkpoint={self.checkpoint_path}"
            )

        config = dict(self.MODEL_CONFIGS[self.encoder])
        config["max_depth"] = self.max_depth

        self.model = DepthAnythingV2(**config)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu"
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        if self.logger is not None:
            self.logger.info("Depth Anything V2 Metric loaded.")

    def predict(
        self,
        rgb_image: np.ndarray,
        camera_intrinsics: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            depth_map = self.model.infer_image(bgr_image)

        depth_map = np.asarray(depth_map).astype(np.float32)

        return depth_map