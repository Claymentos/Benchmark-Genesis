"""
Pose representation utilities for handling different rotation representations:
- Quaternions (4D): [x, y, z, w] (scipy convention)
- 6D rotation: First two columns of rotation matrix
- 9D rotation: Full flattened rotation matrix (3x3)

All functions support conversion between these representations and standard
pose formats (rotation + translation + scale).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rsc


def coord_transform():
    """
    Standard coordinate transformation matrix (OpenCV to OpenGL convention).
    
    Returns:
        4x4 transformation matrix
    """
    t = np.eye(4)
    t[:3, :3] = np.array([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]])
    return t


def decompose_transform(T, rot='quaternion'):
    """
    Decompose a 4x4 transformation matrix into rotation, translation, and scale.
    
    Args:
        T: 4x4 transformation matrix
        rot: Rotation output format - 'quaternion' or 'matrix'
    
    Returns:
        (R, t, s): Rotation (quaternion or matrix), translation, scale
    """
    RS = T[:3, :3]
    t = T[:3, 3]
    s = np.cbrt(np.linalg.det(RS))  # handles uniform positive scale
    R = RS / s
    if rot == 'matrix':
        return R, t, s
    elif rot == 'quaternion':
        q = Rsc.from_matrix(R).as_quat()
        return q, t, s


def compose_transform(R_or_q, t, s=1.0):
    """
    Compose a 4x4 transformation matrix from rotation, translation, and scale.
    
    Args:
        R_or_q: Either 3x3 rotation matrix or 4-element quaternion [x, y, z, w]
        t: 3-element translation vector
        s: Scale factor (default: 1.0)
    
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    
    R_or_q = np.asarray(R_or_q)
    if R_or_q.shape == (3, 3):
        R_mat = R_or_q
    elif R_or_q.shape == (4,):  # quaternion [x, y, z, w]
        R_mat = Rsc.from_quat(R_or_q).as_matrix()
    else:
        raise ValueError("Rotation must be a 3×3 matrix or a 4-element quaternion [x, y, z, w].")
    
    T[:3, :3] = R_mat * s
    T[:3, 3] = np.asarray(t)
    return T


# ============================================================================
# Quaternion conversions
# ============================================================================

def from_quaternion_to_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: 4-element quaternion [x, y, z, w] (scipy convention)
    
    Returns:
        3x3 rotation matrix
    """
    return Rsc.from_quat(q).as_matrix()


def from_quaternion_to_6D(q):
    """
    Convert quaternion to 6D rotation representation.
    
    Args:
        q: 4-element quaternion [x, y, z, w]
    
    Returns:
        6-element 6D rotation representation
    """
    R = from_quaternion_to_matrix(q)
    return from_matrix_to_6D(R)


# ============================================================================
# 6D rotation conversions
# ============================================================================

def from_matrix_to_6D(R):
    """
    Convert 3x3 rotation matrix to 6D representation.
    
    6D representation = first two columns of rotation matrix flattened.
    This representation is continuous and differentiable, making it suitable
    for neural network outputs.
    
    Args:
        R: (3, 3) rotation matrix
    
    Returns:
        rot6d: (6,) 6D rotation representation [r11, r21, r31, r12, r22, r32]
    """
    R = np.asarray(R)
    # Take first two columns and flatten: [r11, r21, r31, r12, r22, r32]
    return R[:, :2].T.flatten()


def from_6D_to_matrix(rot6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    Uses Gram-Schmidt orthonormalization to reconstruct the third column
    from the first two columns encoded in the 6D representation.
    
    Args:
        rot6d: (6,) 6D rotation representation [r11, r21, r31, r12, r22, r32]
    
    Returns:
        R: (3, 3) rotation matrix
    """
    rot6d = np.asarray(rot6d)
    # Reshape to get first two columns
    a1 = rot6d[:3]  # first column
    a2 = rot6d[3:6]  # second column
    
    # Gram-Schmidt orthonormalization
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    
    R = np.stack([b1, b2, b3], axis=1)
    return R


def from_6D_to_quaternion(rot6d):
    """
    Convert 6D rotation representation to quaternion.
    
    Args:
        rot6d: (6,) 6D rotation representation
    
    Returns:
        q: (4,) quaternion [x, y, z, w]
    """
    R = from_6D_to_matrix(rot6d)
    return Rsc.from_matrix(R).as_quat()


def get_pose_dimensions(pose_representation):
    """
    Get the dimensions and indices for different pose representations.
    
    Args:
        pose_representation: One of 'quaternion_translation_scale', 
                           '6d_translation_scale', '9d_translation_scale'
    
    Returns:
        tuple: (rot_dim, trans_start, scale_idx, total_dim)
            - rot_dim: Number of rotation dimensions
            - trans_start: Starting index of translation
            - scale_idx: Index of scale value
            - total_dim: Total pose tensor dimension
    
    Examples:
        >>> rot_dim, trans_start, scale_idx, total_dim = get_pose_dimensions('6d_translation_scale')
        >>> print(f"6D pose: {total_dim}D total, rotation={rot_dim}D")
        6D pose: 10D total, rotation=6D
    """
    if pose_representation == 'quaternion_translation_scale':
        # 8D: [quat(4), trans(3), scale(1)]
        return 4, 4, 7, 8
    elif pose_representation == '6d_translation_scale':
        # 10D: [6d_rot(6), trans(3), scale(1)]
        return 6, 6, 9, 10
    elif pose_representation == '9d_translation_scale':
        # 13D: [rot_matrix(9), trans(3), scale(1)]
        return 9, 9, 12, 13
    else:
        raise ValueError(
            f"Unknown pose_representation: {pose_representation}. "
            f"Must be one of: 'quaternion_translation_scale', '6d_translation_scale', '9d_translation_scale'"
        )

def parse_pose_output(pose_tensor, pose_representation="quaternion_translation_scale"):
    """
    Parse pose tensor from model output to standard quaternion format.
    
    This function handles different pose representations (quaternion, 6D, 9D rotation matrix)
    and converts them to a unified format with quaternion for downstream use.
    
    Args:
        pose_tensor: Model output pose tensor (numpy array or torch.Tensor)
                    - 8D: [quat(4), trans(3), scale(1)] for quaternion_translation_scale
                    - 10D: [6d_rot(6), trans(3), scale(1)] for 6d_translation_scale
                    - 13D: [rot_matrix(9), trans(3), scale(1)] for 9d_translation_scale
        pose_representation: One of 'quaternion_translation_scale', '6d_translation_scale', '9d_translation_scale'
    
    Returns:
        dict with keys:
            - 'quaternion': (4,) array in [x, y, z, w] format (scipy convention)
            - 'translation': (3,) array
            - 'scale': float
            - 'rotation_matrix': (3, 3) array
    
    Examples:
        >>> # Parse quaternion pose (8D)
        >>> pose_8d = np.array([0, 0, 0, 1, 0, 0, 0, 1.0])
        >>> result = parse_pose_output(pose_8d, "quaternion_translation_scale")
        >>> print(result['quaternion'])  # [0, 0, 0, 1]
        
        >>> # Parse 6D rotation pose (10D)
        >>> pose_10d = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1.0])
        >>> result = parse_pose_output(pose_10d, "6d_translation_scale")
        >>> print(result['scale'])  # 1.0
    """
    # Convert to numpy if needed
    if isinstance(pose_tensor, torch.Tensor):
        pose_np = pose_tensor.detach().cpu().numpy()
    else:
        pose_np = np.array(pose_tensor)
    
    if pose_representation == "quaternion_translation_scale":
        # 8D: [quat(4), trans(3), scale(1)]
        quat = pose_np[:4]
        trans = pose_np[4:7]
        scale = float(pose_np[7])
        rot_matrix = from_quaternion_to_matrix(quat)
        
    elif pose_representation == "6d_translation_scale":
        # 10D: [6d_rot(6), trans(3), scale(1)]
        rot_6d = pose_np[:6]
        trans = pose_np[6:9]
        scale = float(pose_np[9])
        quat = from_6D_to_quaternion(rot_6d)
        rot_matrix = from_6D_to_matrix(rot_6d)
        
    elif pose_representation == "9d_translation_scale":
        # 13D: [rot_matrix(9), trans(3), scale(1)]
        rot_9d = pose_np[:9]
        trans = pose_np[9:12]
        scale = float(pose_np[12])
        rot_matrix = rot_9d.reshape(3, 3)
        quat = Rsc.from_matrix(rot_matrix).as_quat()
    else:
        raise ValueError(f"Unknown pose_representation: {pose_representation}. "
                        f"Must be one of: 'quaternion_translation_scale', '6d_translation_scale', '9d_translation_scale'")
    
    return {
        'quaternion': quat,
        'translation': trans,
        'scale': scale,
        'rotation_matrix': rot_matrix
    }


def compute_geodesic_rotation_error(pose_pred, pose_gt, pose_representation, return_degrees=True):
    """
    Compute geodesic rotation error between predicted and ground truth poses.
    
    The geodesic distance is the angle of rotation needed to align the predicted
    rotation with the ground truth rotation. This is a representation-agnostic
    metric that measures actual rotation error rather than representation error.
    
    Args:
        pose_pred: Predicted pose tensor or array (any supported representation)
        pose_gt: Ground truth pose tensor or array (same representation as pose_pred)
        pose_representation: Pose format string
        return_degrees: If True, return error in degrees; if False, return radians
    
    Returns:
        float: Geodesic rotation error in degrees (or radians if return_degrees=False)
               Returns NaN if computation fails
    
    Formula:
        geodesic_distance = arccos((trace(R_pred^T @ R_gt) - 1) / 2)
    
    This measures the minimal rotation angle needed to align the two rotations,
    independent of the pose representation used.
    """
    try:
        # Parse both predicted and GT poses to rotation matrices
        pred_parsed = parse_pose_output(pose_pred, pose_representation)
        gt_parsed = parse_pose_output(pose_gt, pose_representation)
        
        # Compute relative rotation: R_error = R_pred^T @ R_gt
        R_pred = pred_parsed['rotation_matrix']
        R_gt = gt_parsed['rotation_matrix']
        R_error = R_pred.T @ R_gt
        
        # Geodesic distance = arccos((trace(R_error) - 1) / 2)
        trace = np.trace(R_error)
        # Clamp to avoid numerical issues with arccos (domain is [-1, 1])
        geodesic_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        
        if return_degrees:
            return np.degrees(geodesic_rad)
        else:
            return geodesic_rad
    except Exception:
        return np.nan


def compute_geodesic_rotation_errors_batch(pose_samples, pose_gt, pose_representation, return_degrees=True):
    """
    Compute geodesic rotation errors for a batch of poses.
    
    Efficient vectorized implementation inspired by PyTorch3D's so3_relative_angle.
    Computes: angle = arccos((trace(R_pred^T @ R_gt) - 1) / 2) for all samples.
    
    Args:
        pose_samples: Batch of predicted poses (N, D) tensor or array
        pose_gt: Batch of ground truth poses (N, D) tensor or array
        pose_representation: Pose format string
        return_degrees: If True, return errors in degrees; if False, return radians
    
    Returns:
        np.ndarray: Array of geodesic rotation errors, one per sample
                   Invalid samples are marked with NaN
    
    Example:
        >>> import torch
        >>> pose_pred = torch.randn(10, 10)  # 10 samples, 6D rotation
        >>> pose_gt = torch.randn(10, 10)
        >>> errors = compute_geodesic_rotation_errors_batch(
        ...     pose_pred, pose_gt, '6d_translation_scale'
        ... )
        >>> print(f"Mean error: {np.nanmean(errors):.2f} degrees")
    
    References:
        - PyTorch3D so3_relative_angle: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html
    """
    try:
        # Parse all poses to rotation matrices efficiently
        batch_size = pose_samples.shape[0]
        R_pred_batch = np.zeros((batch_size, 3, 3))
        R_gt_batch = np.zeros((batch_size, 3, 3))
        
        for i in range(batch_size):
            pred_parsed = parse_pose_output(pose_samples[i], pose_representation)
            gt_parsed = parse_pose_output(pose_gt[i], pose_representation)
            R_pred_batch[i] = pred_parsed['rotation_matrix']
            R_gt_batch[i] = gt_parsed['rotation_matrix']
        
        # Vectorized computation: R_relative = R_pred^T @ R_gt
        # Using einsum for batch matrix multiplication: (N,3,3) @ (N,3,3)^T
        R_relative = np.einsum('nij,nkj->nik', R_pred_batch, R_gt_batch)
        
        # Compute traces for all matrices at once
        traces = np.trace(R_relative, axis1=1, axis2=2)
        
        # Geodesic distance = arccos((trace - 1) / 2)
        # Clamp to avoid numerical issues with arccos
        cos_angles = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
        geodesic_rad = np.arccos(cos_angles)
        
        if return_degrees:
            return np.degrees(geodesic_rad)
        else:
            return geodesic_rad
            
    except Exception as e:
        # Fallback to per-sample computation if vectorization fails
        errors = []
        for i in range(pose_samples.shape[0]):
            error = compute_geodesic_rotation_error(
                pose_samples[i], pose_gt[i], pose_representation, return_degrees
            )
            errors.append(error)
        return np.array(errors)


# ============================================================================
# Pose Normalization, in SAM3D was showing better results with this approach
# ============================================================================

class SimpleStatsAggregator:
    """
    Simple statistics aggregator for computing mean and std of pose components.

    Example:
        >>> aggregator = SimpleStatsAggregator(dim=6)
        >>> for batch in data_loader:
        >>>     aggregator.update(batch)
        >>> stats = aggregator.finalize()
    """

    def __init__(self, dim: int):
        """
        Initialize stats aggregator.

        Args:
            dim: Dimensionality of the data (e.g., 6 for 6D rotation)
        """
        self.dim = dim
        self.samples = []

    def update(self, batch: np.ndarray) -> None:
        """Update with new batch of data."""
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        assert batch.shape[1] == self.dim
        self.samples.append(batch)

    def finalize(self) -> dict:
        """Return computed statistics."""
        if len(self.samples) == 0:
            return {
                'mean': np.zeros(self.dim, dtype=np.float64),
                'std': np.ones(self.dim, dtype=np.float64),
                'count': 0
            }

        all_samples = np.concatenate(self.samples, axis=0)
        mean = np.mean(all_samples, axis=0)
        std = np.std(all_samples, axis=0)
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero

        return {'mean': mean, 'std': std, 'count': len(all_samples)}


class PoseNormalizer:
    """
    Handles normalization and denormalization of pose data.

    Applies z-score normalization to pose components and uses existing
    from_6D_to_matrix() for Gram-Schmidt orthonormalization.

    Example:
        >>> from recgen_inference.recgen_modules.utils.pose_utils import PoseNormalizer, load_pose_normalization_stats
        >>> stats = load_pose_normalization_stats('configs/pose_stats.json')
        >>> normalizer = PoseNormalizer(stats)
        >>> pose_norm = normalizer.normalize(pose, '6d_translation_scale')
        >>> pose_denorm = normalizer.denormalize(pose_norm, '6d_translation_scale')
    """

    def __init__(self, stats_config):
        """
        Initialize with normalization statistics.

        Args:
            stats_config: Either a path (str) to JSON file or a dict with 'global_statistics' containing:
                - '6d_rotation': {'mean': [...], 'std': [...]}
                - 'quaternion': {'mean': [...], 'std': [...]} (optional)
                - '9d_rotation': {'mean': [...], 'std': [...]} (optional)
                - 'translation': {'mean': [...], 'std': [...]}
                - 'scale': {'mean': ..., 'std': ...}
        """
        import torch

        # If path is provided, load stats from file
        if isinstance(stats_config, str):
            stats_config = load_pose_normalization_stats(stats_config)

        global_stats = stats_config.get('global_statistics', stats_config)

        # Load 6D rotation stats
        rot_stats = global_stats['6d_rotation']
        self.rot_6d_mean = torch.tensor(rot_stats['mean'], dtype=torch.float32)
        self.rot_6d_std = torch.tensor(rot_stats['std'], dtype=torch.float32)

        # Load quaternion stats (optional)
        if 'quaternion' in global_stats:
            quat_stats = global_stats['quaternion']
            self.quat_mean = torch.tensor(quat_stats['mean'], dtype=torch.float32)
            self.quat_std = torch.tensor(quat_stats['std'], dtype=torch.float32)
        else:
            self.quat_mean = None
            self.quat_std = None

        # Load 9D rotation stats (optional)
        if '9d_rotation' in global_stats:
            rot_9d_stats = global_stats['9d_rotation']
            self.rot_9d_mean = torch.tensor(rot_9d_stats['mean'], dtype=torch.float32)
            self.rot_9d_std = torch.tensor(rot_9d_stats['std'], dtype=torch.float32)
        else:
            self.rot_9d_mean = None
            self.rot_9d_std = None

        # Load translation stats
        trans_stats = global_stats['translation']
        self.trans_mean = torch.tensor(trans_stats['mean'], dtype=torch.float32)
        self.trans_std = torch.tensor(trans_stats['std'], dtype=torch.float32)

        # Load scale stats
        scale_stats = global_stats['scale']
        self.scale_mean = torch.tensor(scale_stats['mean'], dtype=torch.float32)
        self.scale_std = torch.tensor(scale_stats['std'], dtype=torch.float32)

        # Track current device to avoid repeated .to() calls
        self._current_device = torch.device('cpu')

    def to(self, device):
        """Move normalizer tensors to specified device."""
        import torch
        device = torch.device(device) if isinstance(device, str) else device
        if device == self._current_device:
            return self
        self.rot_6d_mean = self.rot_6d_mean.to(device)
        self.rot_6d_std = self.rot_6d_std.to(device)
        if self.quat_mean is not None:
            self.quat_mean = self.quat_mean.to(device)
            self.quat_std = self.quat_std.to(device)
        if self.rot_9d_mean is not None:
            self.rot_9d_mean = self.rot_9d_mean.to(device)
            self.rot_9d_std = self.rot_9d_std.to(device)
        self.trans_mean = self.trans_mean.to(device)
        self.trans_std = self.trans_std.to(device)
        self.scale_mean = self.scale_mean.to(device)
        self.scale_std = self.scale_std.to(device)
        self._current_device = device
        return self

    def _ensure_device(self, device):
        """Ensure tensors are on the correct device, moving only if necessary."""
        if device != self._current_device:
            self.to(device)

    def _apply_gram_schmidt_6d_torch(self, rot_6d):
        """
        Apply Gram-Schmidt to 6D rotation using existing from_6D_to_matrix logic.

        Args:
            rot_6d: (6,) or (batch_size, 6) tensor

        Returns:
            Orthonormalized 6D rotation tensor
        """
        import torch
        import torch.nn.functional as F

        if rot_6d.ndim == 1:
            rot_6d = rot_6d.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Extract first two columns
        b1 = rot_6d[:, :3]
        b2 = rot_6d[:, 3:6]

        # Gram-Schmidt (same logic as from_6D_to_matrix)
        b1 = F.normalize(b1, dim=1, eps=1e-8)
        b2 = b2 - (b1 * b2).sum(dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1, eps=1e-8)

        rot_6d_ortho = torch.cat([b1, b2], dim=1)

        if squeeze_output:
            rot_6d_ortho = rot_6d_ortho.squeeze(0)

        return rot_6d_ortho

    def normalize(self, pose_tensor, representation: str):
        """
        Normalize pose tensor.

        Args:
            pose_tensor: Pose tensor (8D/10D/13D)
            representation: One of 'quaternion_translation_scale',
                          '6d_translation_scale', '9d_translation_scale'

        Returns:
            Normalized pose tensor
        """
        import torch

        if pose_tensor.ndim == 1:
            pose_tensor = pose_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        original_device = pose_tensor.device
        # Compute on CPU to avoid CUDA errors in DataLoader workers
        # (persistent workers inherit forked CUDA context which can't reinitialize)
        if original_device.type == 'cuda':
            pose_tensor = pose_tensor.cpu()
        device = pose_tensor.device
        self._ensure_device(device)

        if representation == '6d_translation_scale':
            # 10D: [6d_rot(6), trans(3), scale(1)]
            rot_6d = pose_tensor[:, :6]
            trans = pose_tensor[:, 6:9]
            scale = pose_tensor[:, 9:10]

            rot_norm = (rot_6d - self.rot_6d_mean) / self.rot_6d_std
            trans_norm = (trans - self.trans_mean) / self.trans_std
            scale_norm = (scale - self.scale_mean) / self.scale_std

            pose_norm = torch.cat([rot_norm, trans_norm, scale_norm], dim=1)

        elif representation == 'quaternion_translation_scale':
            # 8D: [quat(4), trans(3), scale(1)]
            quat = pose_tensor[:, :4]
            trans = pose_tensor[:, 4:7]
            scale = pose_tensor[:, 7:8]

            if self.quat_mean is not None:
                quat_norm = (quat - self.quat_mean) / self.quat_std
            else:
                quat_norm = quat

            trans_norm = (trans - self.trans_mean) / self.trans_std
            scale_norm = (scale - self.scale_mean) / self.scale_std

            pose_norm = torch.cat([quat_norm, trans_norm, scale_norm], dim=1)

        elif representation == '9d_translation_scale':
            # 13D: [rot_matrix(9), trans(3), scale(1)]
            rot_9d = pose_tensor[:, :9]
            trans = pose_tensor[:, 9:12]
            scale = pose_tensor[:, 12:13]

            if self.rot_9d_mean is not None:
                rot_9d_norm = (rot_9d - self.rot_9d_mean) / self.rot_9d_std
            else:
                rot_9d_norm = rot_9d

            trans_norm = (trans - self.trans_mean) / self.trans_std
            scale_norm = (scale - self.scale_mean) / self.scale_std

            pose_norm = torch.cat([rot_9d_norm, trans_norm, scale_norm], dim=1)

        else:
            raise ValueError(f"Unknown representation: {representation}")

        if squeeze_output:
            pose_norm = pose_norm.squeeze(0)

        return pose_norm.to(original_device)

    def denormalize(self, pose_tensor, representation: str):
        """
        Denormalize pose tensor.

        Uses existing from_6D_to_matrix() for Gram-Schmidt on 6D rotations.

        Args:
            pose_tensor: Normalized pose tensor
            representation: Pose representation type

        Returns:
            Denormalized pose tensor
        """
        import torch

        if pose_tensor.ndim == 1:
            pose_tensor = pose_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        original_device = pose_tensor.device
        # Compute on CPU to avoid CUDA errors in DataLoader workers
        # (persistent workers inherit forked CUDA context which can't reinitialize)
        if original_device.type == 'cuda':
            pose_tensor = pose_tensor.cpu()
        device = pose_tensor.device
        self._ensure_device(device)

        if representation == '6d_translation_scale':
            # 10D: [6d_rot(6), trans(3), scale(1)]
            rot_6d_norm = pose_tensor[:, :6]
            trans_norm = pose_tensor[:, 6:9]
            scale_norm = pose_tensor[:, 9:10]

            # Denormalize
            rot_6d = rot_6d_norm * self.rot_6d_std + self.rot_6d_mean
            trans = trans_norm * self.trans_std + self.trans_mean
            scale = scale_norm * self.scale_std + self.scale_mean
            # we don't need to apply Gram-Schmidt here
            # it should be applied after denormalization

            # Ensure positive scale
            scale = torch.clamp(scale, min=1e-6)

            pose_denorm = torch.cat([rot_6d, trans, scale], dim=1)

        elif representation == 'quaternion_translation_scale':
            # 8D: [quat(4), trans(3), scale(1)]
            quat_norm = pose_tensor[:, :4]
            trans_norm = pose_tensor[:, 4:7]
            scale_norm = pose_tensor[:, 7:8]

            # Denormalize
            if self.quat_mean is not None:
                quat = quat_norm * self.quat_std + self.quat_mean
                # Renormalize to unit length
                quat = quat / (torch.norm(quat, dim=1, keepdim=True) + 1e-8)
            else:
                quat = quat_norm

            trans = trans_norm * self.trans_std + self.trans_mean
            scale = scale_norm * self.scale_std + self.scale_mean
            scale = torch.clamp(scale, min=1e-6)

            pose_denorm = torch.cat([quat, trans, scale], dim=1)

        elif representation == '9d_translation_scale':
            # 13D: [rot_matrix(9), trans(3), scale(1)]
            rot_9d_norm = pose_tensor[:, :9]
            trans_norm = pose_tensor[:, 9:12]
            scale_norm = pose_tensor[:, 12:13]

            # Denormalize
            if self.rot_9d_mean is not None:
                rot_9d = rot_9d_norm * self.rot_9d_std + self.rot_9d_mean
            else:
                rot_9d = rot_9d_norm

            trans = trans_norm * self.trans_std + self.trans_mean
            scale = scale_norm * self.scale_std + self.scale_mean
            scale = torch.clamp(scale, min=1e-6)

            pose_denorm = torch.cat([rot_9d, trans, scale], dim=1)

        else:
            raise ValueError(f"Unknown representation: {representation}")

        if squeeze_output:
            pose_denorm = pose_denorm.squeeze(0)

        return pose_denorm.to(original_device)


def load_pose_normalization_stats(config_path: str) -> dict:
    """
    Load pose normalization statistics from JSON file.

    Args:
        config_path: Path to JSON file with statistics

    Returns:
        Dictionary with statistics
    """
    import json
    from pathlib import Path

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Normalization config not found: {config_path}")

    with open(config_path, 'r') as f:
        stats = json.load(f)

    # Validate required fields
    global_stats = stats.get('global_statistics', stats)
    required_keys = ['6d_rotation', 'translation', 'scale']
    for key in required_keys:
        if key not in global_stats:
            raise ValueError(f"Missing required key '{key}' in normalization config")

    return stats


def save_pose_normalization_stats(stats: dict, output_path: str) -> None:
    """
    Save pose normalization statistics to JSON file.

    Args:
        stats: Dictionary with statistics
        output_path: Output path for JSON file
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists
    def convert_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    stats_serializable = convert_arrays(stats)

    with open(output_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"Saved normalization statistics to: {output_path}")


def validate_round_trip(normalizer: PoseNormalizer, test_samples, representation: str, tolerance: float = 1e-5) -> dict:
    """
    Validate that normalization -> denormalization preserves original values.

    Args:
        normalizer: PoseNormalizer instance
        test_samples: (N, D) tensor of test poses
        representation: Pose representation type
        tolerance: Maximum allowed error

    Returns:
        Dictionary with validation metrics
    """
    import torch

    # Normalize
    normalized = normalizer.normalize(test_samples, representation)

    # Denormalize
    denormalized = normalizer.denormalize(normalized, representation)

    # Compute errors
    errors = torch.abs(test_samples - denormalized)
    max_error = errors.max().item()
    mean_error = errors.mean().item()

    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'passed': max_error < tolerance
    }