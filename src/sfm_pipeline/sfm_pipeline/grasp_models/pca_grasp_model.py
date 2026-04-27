import numpy as np
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from sfm_pipeline.grasp_models.base import (
    BaseGraspModel,
    GraspContext,
    GraspPrediction,
)


class PCAGraspModel(BaseGraspModel):
    """
    Rule-based baseline grasp model.

    This is equivalent to your current method:
    - Grasp x/y from object point cloud median
    - Grasp z from upper object surface
    - Grasp yaw from PCA main axis
    """

    def __init__(self, z_offset=0.035):
        self.z_offset = float(z_offset)

    def predict(self, context: GraspContext) -> GraspPrediction:
        if context.object_pcd is None:
            raise ValueError("PCAGraspModel requires context.object_pcd")

        object_points = np.asarray(context.object_pcd.points)

        if object_points.shape[0] < 30:
            raise ValueError("Not enough object points for PCA grasp estimation")

        grasp_x = float(np.median(object_points[:, 0]))
        grasp_y = float(np.median(object_points[:, 1]))
        top_z = float(np.percentile(object_points[:, 2], 95))
        grasp_z = top_z + self.z_offset

        yaw = self.estimate_yaw_from_pca(object_points)

        pose = PoseStamped()

        if context.stamp is not None:
            pose.header.stamp = context.stamp

        pose.header.frame_id = context.frame_id

        pose.pose.position.x = grasp_x
        pose.pose.position.y = grasp_y
        pose.pose.position.z = grasp_z

        quat = R.from_euler("z", yaw).as_quat()

        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])

        return GraspPrediction(
            pose=pose,
            score=1.0,
            width=None,
            debug={
                "method": "pca",
                "yaw_rad": float(yaw),
                "yaw_deg": float(np.degrees(yaw)),
                "num_points": int(object_points.shape[0]),
                "top_z": float(top_z),
            },
        )

    @staticmethod
    def estimate_yaw_from_pca(points):
        xy = points[:, :2]
        xy = xy - np.mean(xy, axis=0)

        if len(xy) < 3:
            return 0.0

        cov = np.cov(xy.T)
        eig_vals, eig_vecs = np.linalg.eig(cov)

        main_axis = eig_vecs[:, np.argmax(eig_vals)]
        yaw = np.arctan2(main_axis[1], main_axis[0])

        return float(yaw)