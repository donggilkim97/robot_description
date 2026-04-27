from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from geometry_msgs.msg import PoseStamped


@dataclass
class GraspContext:
    """
    Shared input container for different grasp models.

    Some models use point clouds.
    Some models use depth maps.
    Some models use RGB + depth.
    """

    rgb_image: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None
    object_pcd: Optional[Any] = None
    camera_intrinsics: Optional[Dict[str, float]] = None
    transform_base_from_camera: Optional[Any] = None
    frame_id: str = "base_link"
    stamp: Optional[Any] = None


@dataclass
class GraspPrediction:
    pose: PoseStamped
    score: float = 1.0
    width: Optional[float] = None
    debug: Dict[str, Any] = field(default_factory=dict)


class BaseGraspModel:
    def predict(self, context: GraspContext) -> GraspPrediction:
        raise NotImplementedError