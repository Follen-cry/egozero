from typing import Tuple

import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# these are hard-coded. change per aruco tag
aruco_length = 0.061
aruco_id = 0
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()


def detect_aruco(
    rgb: np.ndarray,
    k: np.ndarray,
    d: np.ndarray = np.zeros(5, dtype=np.float32),
    aruco_length: float = aruco_length,
    aruco_id: int = aruco_id,
    aruco_dict: aruco.Dictionary = aruco_dict,
    aruco_params: aruco.DetectorParameters = aruco_params,
):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params
    )
    if ids is not None:
        for i, detected_id in enumerate(ids):
            detected_id = int(detected_id[0])
            if detected_id == aruco_id:
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[i], aruco_length, k, d
                )
                r_cam, _ = cv2.Rodrigues(rvec[0])
                t_cam = tvec[0].flatten()
                pose = np.eye(4)
                pose[:3, :3] = r_cam
                pose[:3, 3] = t_cam
                return pose


def add_border(
    rgb: np.ndarray, text: str = None, color: Tuple[int, int, int] = (255, 255, 255)
):
    # border
    cv2.rectangle(rgb, (0, 0), (rgb.shape[1] - 1, rgb.shape[0] - 1), color, 10)

    # text
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x, y = 10, 30
        padding_x, padding_y = 10, 10
        bg_rect = (
            x - padding_x,
            y - text_size[1] - padding_y,
            text_size[0] + 2 * padding_x,
            text_size[1] + 2 * padding_y,
        )
        bg_rect = (x - 5, y - 25, text_size[0] + 10, text_size[1] + 10)
        cv2.rectangle(
            rgb,
            (bg_rect[0], bg_rect[1]),
            (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]),
            (0, 0, 0),
            -1,
        )
        cv2.putText(rgb, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return rgb


def plot_points(
    rgb: np.ndarray,
    pose: np.ndarray,
    k: np.ndarray,
    d: np.ndarray,
    points: np.ndarray = np.zeros((3,)),
    labels: list[str] = None,
    **kwargs,
):
    r = pose[:3, :3]
    t = pose[:3, 3]
    rvec, _ = cv2.Rodrigues(r)
    xys, _ = cv2.projectPoints(points, rvec, t, k, d)
    xys = [xy.ravel().astype(np.int32) for xy in xys]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 if max(rgb.shape[:-1]) > 720 else 0.25
    thickness = 1
    offset_x, offset_y = 15, 5

    for i, xy in enumerate(xys):
        cv2.circle(rgb, tuple(xy), **kwargs)

        if labels is not None:
            text = str(labels[i])
            position = (xy[0] + offset_x, xy[1] + offset_y)
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            rect_topleft = (position[0] - 5, position[1] - th - 5)
            rect_bottomright = (position[0] + tw + 5, position[1] + baseline + 5)

            # Black background rectangle
            cv2.rectangle(rgb, rect_topleft, rect_bottomright, (0, 0, 0), -1)
            # Text overlay
            cv2.putText(
                rgb, text, position, font, font_scale, kwargs["color"], thickness
            )

    return rgb


def pt_is_oob(pt, image_size):
    return not ((0 <= pt[0] < image_size[1]) and (0 <= pt[1] < image_size[0]))


def draw_axis(
    img: np.ndarray,
    pose: np.ndarray,
    k: np.ndarray,
    d: np.ndarray,
    upper_left: str = "",
    upper_right: str = "",
):
    """Function to manually draw the axes on the detected markers"""
    r = pose[:3, :3]
    t = pose[:3, 3]
    r, _ = cv2.Rodrigues(r)

    # Project points from the object frame to the image frame
    l = 0.05
    axis = np.float32([[l, 0, 0], [0, l, 0], [0, 0, l], [0, 0, 0]]).reshape(-1, 3)
    img_pts, _ = cv2.projectPoints(axis, r, t, k, d)

    # Convert to tuples for the line function
    img_pts = [pt.ravel().astype(np.int32) for pt in img_pts]
    img_shape = img.shape[:2]
    if not pt_is_oob(img_pts[3], img_shape):
        if not pt_is_oob(img_pts[0], img_shape):
            img = cv2.line(
                img,
                tuple(img_pts[3].astype(int)),
                tuple(img_pts[0].astype(int)),
                (0, 0, 255),
                3,
            )
        if not pt_is_oob(img_pts[1], img_shape):
            img = cv2.line(
                img, img_pts[3].astype(int), img_pts[1].astype(int), (0, 255, 0), 3
            )
        if not pt_is_oob(img_pts[2], img_shape):
            img = cv2.line(
                img, img_pts[3].astype(int), img_pts[2].astype(int), (255, 0, 0), 3
            )

    if upper_left:
        text = upper_left
        position = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1 if max(img.shape[:-1]) > 720 else 0.5
        color = (232, 168, 0)
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        rect_top_left = (position[0] - 5, position[1] - text_height - 5)
        rect_bottom_right = (position[0] + text_width + 5, position[1] + baseline + 5)
        cv2.rectangle(img, rect_top_left, rect_bottom_right, (0, 0, 0), -1)
        cv2.putText(img, text, position, font, font_scale, color, thickness)

    if upper_right:
        text = upper_right
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1 if max(img.shape[:-1]) > 720 else 0.5
        text_color = (0, 135, 255)
        thickness = 2
        image_height, image_width, _ = img.shape
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        position = (image_width - text_width - 10, 30)  # 10px padding from the right
        rect_top_left = (position[0] - 5, position[1] - text_height - 5)
        rect_bottom_right = (position[0] + text_width + 5, position[1] + baseline + 5)
        cv2.rectangle(
            img, rect_top_left, rect_bottom_right, (0, 0, 0), -1
        )  # -1 fills the rectangle
        cv2.putText(img, text, position, font, font_scale, text_color, thickness)

    return img, img_pts[3]


def plot_transforms_over_time(T: np.ndarray, T_avg: np.ndarray, save_path: str):
    """
    Plots translation and unwrapped Euler angles over time from a sequence of 4x4 transformation matrices.
    Also plots the trimmed mean values as horizontal lines in separate subplots,
    keeping axis limits matched between each pair.

    Args:
        T (np.ndarray): Array of shape (n, 4, 4) representing transformation matrices.
        T_avg (np.ndarray): Array of shape (4, 4) representing the average transformation matrix.
    """
    assert T.ndim == 3 and T.shape[1:] == (4, 4), "Input must be of shape (n, 4, 4)"
    n = T.shape[0]

    # Extract translations
    translations = T[:, :3, 3]  # shape (n, 3)
    avg_translation = T_avg[:3, 3]

    # Extract rotations and convert to Euler angles
    rotations = T[:, :3, :3]  # shape (n, 3, 3)
    eulers = R.from_matrix(rotations).as_euler("xyz", degrees=True)  # shape (n, 3)
    eulers_unwrapped = np.unwrap(np.deg2rad(eulers), axis=0)  # unwrap in radians
    eulers_unwrapped_deg = np.rad2deg(eulers_unwrapped)

    # Use median Euler angle over time as reference to wrap around
    avg_euler_deg = R.from_matrix(T_avg[:3, :3]).as_euler(
        "xyz", degrees=True
    )  # shape (3,)
    median_euler_deg = np.median(eulers_unwrapped_deg, axis=0)
    avg_euler_wrapped = np.array(
        [
            ref + ((avg - ref + 180) % 360 - 180)
            for avg, ref in zip(avg_euler_deg, median_euler_deg)
        ]
    )

    time = np.arange(n)

    fig, axs = plt.subplots(2, 2, figsize=(36, 18), sharex="col")

    # --- Translation over time ---
    axs[0, 0].plot(time, translations[:, 0], label="x")
    axs[0, 0].plot(time, translations[:, 1], label="y")
    axs[0, 0].plot(time, translations[:, 2], label="z")
    axs[0, 0].set_ylabel("Translation (m)")
    axs[0, 0].legend()
    axs[0, 0].set_title("Translation over Time")

    # --- Translation trimmed mean ---
    for i in range(3):
        mean = avg_translation[i]
        axs[0, 1].hlines(mean, 0, n - 1, label=f"{'xyz'[i]}", colors=f"C{i}")
    axs[0, 1].set_title("Trimmed Mean Translation")
    axs[0, 1].legend()
    axs[0, 1].set_ylabel("Translation (m)")

    # Match translation subplot limits
    axs[0, 1].set_xlim(axs[0, 0].get_xlim())
    axs[0, 1].set_ylim(axs[0, 0].get_ylim())

    # --- Euler angles over time ---
    axs[1, 0].plot(time, eulers_unwrapped_deg[:, 0], label="roll (x)")
    axs[1, 0].plot(time, eulers_unwrapped_deg[:, 1], label="pitch (y)")
    axs[1, 0].plot(time, eulers_unwrapped_deg[:, 2], label="yaw (z)")
    axs[1, 0].set_ylabel("Euler Angle (deg)")
    axs[1, 0].set_xlabel("Time step")
    axs[1, 0].legend()
    axs[1, 0].set_title("Unwrapped Euler Angles over Time")

    # --- Euler angle trimmed mean ---
    for i, axis in enumerate(["roll", "pitch", "yaw"]):
        mean = avg_euler_wrapped[i]
        axs[1, 1].hlines(mean, 0, n - 1, label=axis, colors=f"C{i}")
    axs[1, 1].set_title("Trimmed Mean Euler Angles")
    axs[1, 1].set_xlabel("Time step")
    axs[1, 1].set_ylabel("Euler Angle (deg)")
    axs[1, 1].legend()

    # Match euler angle subplot limits
    axs[1, 1].set_xlim(axs[1, 0].get_xlim())
    axs[1, 1].set_ylim(axs[1, 0].get_ylim())

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
