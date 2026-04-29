from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class BaseDepthModel(ABC):
    """
    Common interface for all monocular depth models.

    Input:
        rgb_image: H x W x 3 RGB uint8 image
        camera_intrinsics: optional camera intrinsics dictionary

    Output:
        depth_map: H x W float32 depth map
    """

    @abstractmethod
    def predict(
        self,
        rgb_image: np.ndarray,
        camera_intrinsics: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        pass