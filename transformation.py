import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_quat(pose):
    """
    Convert [x, y, z, rx, ry, rz] (UR axis–angle form)
    to (position, quaternion) tuple.
    Quaternion is in [x, y, z, w] order.
    """
    x, y, z, rx, ry, rz = pose
    rotvec = np.array([rx, ry, rz], dtype=np.float64)
    r = R.from_rotvec(rotvec)
    q = r.as_quat()
    q = q / np.linalg.norm(q)
    return np.array([x, y, z], dtype=np.float64), q


def quat_multiply(q1, q2):
    """Hamilton product q = q1 ⊗ q2  (both [x, y, z, w])"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)


def pose_inv(pose):
    """Inverse of a pose (p, q)"""
    p, q = pose
    q_inv = quat_conjugate(q) / np.dot(q, q)
    p_inv = -rotate_vector(q_inv, p)
    return p_inv, q_inv


def quat_conjugate(q):
    """Return quaternion conjugate"""
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)


def rotate_vector(q, v):
    """Rotate a 3-vector v by quaternion q"""
    qv = np.concatenate((v, [0.0]))
    q_conj = quat_conjugate(q)
    return quat_multiply(quat_multiply(q, qv), q_conj)[:3]


def pose_multiply(pose1, pose2):
    """
    Compose two poses in quaternion form.
    Each pose = (p, q) where p is 3×1 position, q is quaternion [x, y, z, w].
    Returns (p_out, q_out).
    """
    p1, q1 = pose1
    p2, q2 = pose2
    # new orientation
    q_out = quat_multiply(q1, q2)
    q_out /= np.linalg.norm(q_out)
    # new position
    p_out = p1 + rotate_vector(q1, p2)
    return p_out, q_out
