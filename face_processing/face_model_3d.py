from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def extract_euler_from_transform(transform_matrix: np.ndarray) -> tuple[float, float, float]:
    """Extract (yaw, pitch, roll) from MediaPipe facial_transformation_matrix.

    Uses intrinsic XYZ convention:
        angles[0] = pitch (rotation around X, up/down nod)
        angles[1] = yaw   (rotation around Y, left/right turn)
        angles[2] = roll  (rotation around Z, head tilt)
    """
    r = Rot.from_matrix(transform_matrix[:3, :3])
    angles = r.as_euler("XYZ", degrees=True)
    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
    return yaw, pitch, roll
