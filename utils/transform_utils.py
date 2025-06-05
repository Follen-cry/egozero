import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation


def interpolate_translations(indices: np.ndarray, t: np.ndarray):
    interp_func = interp1d(indices, t, axis=0, kind="linear", fill_value="extrapolate")
    indices = np.arange(indices.min(), indices.max() + 1)
    t = interp_func(indices)
    return t


def interpolate_rotations(indices: np.ndarray, r: np.ndarray):
    interp_func = CubicSpline(indices, r, axis=0)
    indices = np.arange(indices.min(), indices.max() + 1)
    r = interp_func(indices)
    return r


def filter_rotations(
    R: np.ndarray,
    window_size: int = 5,
    threshold: float = 1.0,
) -> np.ndarray:
    n = R.shape[0]

    # Convert rotation matrices to Euler angles (unwrap to avoid discontinuities)
    rotations = Rotation.from_matrix(R)
    eulers = np.unwrap(rotations.as_euler("xyz"), axis=0)  # shape (n, 3)

    # Compute sum of absolute angular differences
    delta_angles = np.sum(np.abs(np.diff(eulers, axis=0, prepend=eulers[:1])), axis=1)

    # Outlier detection based on relative change
    mask = np.ones(n, dtype=bool)
    for t in range(n):
        start, end = max(0, t - window_size), min(n, t + window_size + 1)
        window_diffs = np.concatenate(
            [delta_angles[start:t], delta_angles[t + 1 : end]]
        )

        if len(window_diffs) > 1:
            mean_window_change = np.mean(window_diffs)
            std_window_change = np.std(window_diffs)
            if (
                std_window_change > 0
                and (delta_angles[t] - mean_window_change)
                > threshold * std_window_change
            ):
                mask[t] = False

    valid_indices = np.where(mask)[0]
    return valid_indices


def moving_average_filter(signal: np.ndarray, window_size: int, axis: int = 0):
    if window_size % 2 == 0:
        window_size += 1
    pad_width = window_size // 2
    padded_signal = np.pad(signal, ((pad_width, pad_width), (0, 0)), mode="reflect")
    kernel = np.ones(window_size) / window_size
    smoothed = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="valid"), axis=axis, arr=padded_signal
    )
    return smoothed


def smooth_rotations(R: np.ndarray, window_size: int = 7, order: str = "xyz"):
    eulers = Rotation.from_matrix(R).as_euler(order)
    eulers = np.unwrap(eulers, axis=0)
    smoothed_eulers = moving_average_filter(eulers, window_size=window_size, axis=0)
    return Rotation.from_euler(order, smoothed_eulers).as_matrix()


def smooth_translations(T: np.ndarray, window_size: int = 7):
    return moving_average_filter(T, window_size=window_size, axis=0)


def filter_and_interpolate_fingertips(fingertips):
    all_fingertips, indices = [], []
    for i, fingertip in enumerate(fingertips):
        if fingertip is not None:
            indices.append(i)
            all_fingertips.append(fingertip - fingertip[0])
    indices = np.array(indices)
    all_fingertips = interpolate_translations(indices, all_fingertips)
    indices = np.arange(indices.min(), indices.max() + 1)
    return indices, all_fingertips


def filter_and_interpolate_poses(poses, threshold=2.0, filter=False, smooth=False):
    # remove all missing (None) poses
    indices, poses = zip(*[(i, p) for i, p in enumerate(poses) if p is not None])
    indices = np.stack(indices)
    poses = np.stack(poses)

    # remove all outlier poses
    if filter:
        valid_indices = filter_rotations(
            poses[:, :3, :3], window_size=7, threshold=threshold
        )
        indices = indices[valid_indices]
        poses = poses[valid_indices]

    r = interpolate_rotations(indices, poses[:, :3, :3])
    t = interpolate_translations(indices, poses[:, :3, 3])

    if smooth:
        r = smooth_rotations(r, window_size=7)
        # t = smooth_translations(t, window_size=7)

    indices = np.arange(indices.min(), indices.max() + 1)
    poses = np.array([np.eye(4) for _ in range(len(indices))])
    poses[:, :3, :3] = r
    poses[:, :3, 3] = t
    return indices, poses


def average_poses(T: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Computes the average pose from a sequence of 4x4 transformation matrices.

    Args:
        T (np.ndarray): Array of shape (n, 4, 4).
        method (str): 'mean' or 'median' for translation and rotation averaging.

    Returns:
        np.ndarray: A single 4x4 homogeneous transformation matrix representing the average.
    """
    assert T.ndim == 3 and T.shape[1:] == (4, 4), "Expected shape (n, 4, 4)"
    assert method in ["mean", "median"], "method must be 'mean' or 'median'"

    translations = T[:, :3, 3]  # shape (n, 3)
    rotations = T[:, :3, :3]  # shape (n, 3, 3)

    # --- Average translation ---
    if method == "mean":
        avg_translation = translations.mean(axis=0)
    else:
        avg_translation = np.median(translations, axis=0)

    # --- Average rotation via quaternion mean ---
    rot_objs = Rotation.from_matrix(rotations)
    quats = rot_objs.as_quat()  # (n, 4), format: [x, y, z, w]

    # Normalize quaternions to ensure averaging is meaningful
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    if method == "mean":
        avg_quat = np.mean(quats, axis=0)
    else:
        avg_quat = np.median(quats, axis=0)

    avg_quat /= np.linalg.norm(avg_quat)  # normalize again
    avg_rot = Rotation.from_quat(avg_quat).as_matrix()  # back to 3x3 matrix

    # --- Assemble final transformation ---
    T_avg = np.eye(4)
    T_avg[:3, :3] = avg_rot
    T_avg[:3, 3] = avg_translation
    return T_avg


def trimmed_average(data, lower_quantile=0.3, upper_quantile=0.7):
    """
    Compute trimmed means along axis 0 (time), keeping only values
    between the given quantiles for each row independently.

    Args:
        data: np.ndarray of shape (t, n)
        lower_quantile: float
        upper_quantile: float

    Returns:
        means: np.ndarray of shape (n,)
    """
    is_1d = data.ndim == 1
    if is_1d:
        data = data[:, None]
    data = np.asarray(data).T
    assert data.ndim == 2, "Input must be 2D: (n, t)"

    # Compute quantiles per row
    lower = np.quantile(data, lower_quantile, axis=1, keepdims=True)
    upper = np.quantile(data, upper_quantile, axis=1, keepdims=True)

    # Broadcast comparison
    mask = (data >= lower) & (data <= upper)

    # Use masked array to ignore values outside quantiles
    masked = np.ma.array(data, mask=~mask)

    # Compute mean ignoring masked values
    masked_mean = masked.mean(axis=1).filled(np.nan)  # nan if all values masked

    if is_1d:
        masked_mean = masked_mean.reshape(-1)

    return masked_mean


def trimmed_average_poses(
    T: np.ndarray, lower_quantile=0.3, upper_quantile=0.7, save_path=None
):
    t = T[:, :3, 3]
    t = trimmed_average(t, lower_quantile=lower_quantile, upper_quantile=upper_quantile)

    r = Rotation.from_matrix(T[:, :3, :3]).as_euler("xyz", degrees=False)
    r = trimmed_average(r, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    r = Rotation.from_euler("xyz", r, degrees=False).as_matrix()

    mean_T = np.eye(4)
    mean_T[:3, 3] = t
    mean_T[:3, :3] = r

    if save_path is not None:
        from .vis_utils import plot_transforms_over_time

        plot_transforms_over_time(T, mean_T, save_path)
    return mean_T
