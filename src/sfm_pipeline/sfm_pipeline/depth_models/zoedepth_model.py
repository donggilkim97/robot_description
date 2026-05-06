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
        self.patch_missing_drop_path()

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
    
    def patch_missing_drop_path(self):
        """
        Compatibility patch for ZoeDepth/MiDaS with newer timm versions.
        Some transformer blocks use drop_path1/drop_path2 instead of drop_path.
        ZoeDepth expects drop_path during inference.
        """
        patched = 0

        for module in self.model.modules():
            if hasattr(module, "drop_path"):
                continue

            if hasattr(module, "drop_path1"):
                module.drop_path = module.drop_path1
                patched += 1
            elif hasattr(module, "drop_path2"):
                module.drop_path = module.drop_path2
                patched += 1

        self.logger.info(f"Patched missing drop_path in {patched} module(s).")