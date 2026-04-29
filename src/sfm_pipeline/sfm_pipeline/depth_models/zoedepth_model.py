from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image as PILImage

from sfm_pipeline.depth_models.base import BaseDepthModel


class ZoeDepthModel(BaseDepthModel):
    def __init__(self, device, logger=None):
        self.device = device
        self.logger = logger

        if self.logger is not None:
            self.logger.info(f"Loading ZoeDepth on [{self.device.type.upper()}]...")

        self.model = torch.hub.load(
            "isl-org/ZoeDepth",
            "ZoeD_N",
            pretrained=True
        )

        self.model.to(self.device)
        self.model.eval()

        if self.logger is not None:
            self.logger.info("ZoeDepth loaded.")

    def predict(
        self,
        rgb_image: np.ndarray,
        camera_intrinsics: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        pil_img = PILImage.fromarray(rgb_image)

        with torch.no_grad():
            depth_map = self.model.infer_pil(pil_img)

        depth_map = np.asarray(depth_map).astype(np.float32)

        return depth_map