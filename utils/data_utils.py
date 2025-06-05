"""Implements extra utility functionality around MPS data"""

import os
import warnings
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.mps import MpsDataPathsProvider, MpsDataProvider
from projectaria_tools.core.mps.utils import filter_points_from_confidence
from projectaria_tools.core.sensor_data import TimeQueryOptions
from torch.utils.data import Dataset, get_worker_info

from point_policy.read_data.aria import load_eeff_in_aruco_frame
from utils.hand_utils import homogenize_mps_wrist_and_palm

DEVIGNETTING_MASKS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "aria_devignetting_masks"
)


@dataclass
class MpsStruct:
    """Abstraction for representing the information fetched by a single frame of MPS recording"""

    idx: int  # frame index
    ts: float  # frame timestamp
    rgb: np.ndarray  # undistorted rgb image
    k: np.ndarray  # linear camera intrinsics of undistorted rgb image
    d: np.ndarray = np.zeros(5)
    c2w: np.ndarray = (
        None  # 4x4 homogeneous camera-to-world extrinsics of undistorted rgb image
    )
    online_calibration: calibration.CameraCalibration = None
    left_wrist: np.ndarray = None
    left_palm: np.ndarray = None
    right_wrist: np.ndarray = None
    right_palm: np.ndarray = None

    def project_points(self, points):
        # map point cloud into camera frame
        T_w2c = np.linalg.inv(self.c2w)
        R_w2c = T_w2c[:3, :3]
        t_w2c = T_w2c[:3, 3]
        points = np.einsum("ij, nj -> ni", R_w2c, points) + t_w2c

        h, w = self.rgb.shape[:2]

        # project point into pixels
        z = points[:, 2]
        valid_mask = z > 0
        homogeneous_pixels = (self.k @ points.T).T
        u = homogeneous_pixels[:, 0] / homogeneous_pixels[:, 2]
        v = homogeneous_pixels[:, 1] / homogeneous_pixels[:, 2]
        in_bounds_mask = (0 <= u) & (u < w) & (0 <= v) & (v < h)
        mask = valid_mask & in_bounds_mask
        projected_points = np.stack((u[mask], v[mask]), axis=-1)
        return projected_points, mask


@dataclass
class MpsMetadata:
    fps: float


class MpsDataset(Dataset):
    """Wrapper around Project Aria's MPS library for easier iteration"""

    def __init__(self, mps_data_path: str, vrs_data_path: str):
        super().__init__()
        self.mps_data_path = mps_data_path
        self.vrs_data_path = vrs_data_path

        self.mps_data_provider, self.provider = self._init_providers()

        # devignetting mask to improve image quality
        device_calib = self.provider.get_device_calibration()
        device_calib.set_devignetting_mask_folder_path(DEVIGNETTING_MASKS_PATH)
        self.devignetting_mask = device_calib.load_devignetting_mask("camera-rgb")

        self.rgb_stream_id = self.provider.get_stream_id_from_label("camera-rgb")

        self.has_online_calibration = (
            self.mps_data_provider is not None
            and self.mps_data_provider.has_semidense_point_cloud()
        )
        self.point_cloud = None
        if self.has_online_calibration:
            point_cloud = self.mps_data_provider.get_semidense_point_cloud()
            point_cloud = filter_points_from_confidence(point_cloud, 0.001, 0.15)
            point_cloud = np.stack([it.position_world for it in point_cloud])
            self.point_cloud = point_cloud
        print(f"found has_online_calibration={self.has_online_calibration}")

    def _init_providers(self) -> Tuple[MpsDataProvider, VrsDataProvider]:
        try:
            mps_provider = MpsDataProvider(
                MpsDataPathsProvider(self.mps_data_path).get_data_paths()
            )
        except:
            mps_provider = None
        vrs_provider = data_provider.create_vrs_data_provider(self.vrs_data_path)
        return mps_provider, vrs_provider

    @property
    def metadata(self):
        first_ts = self.provider.get_image_data_by_index(self.rgb_stream_id, 0)[
            1
        ].capture_timestamp_ns
        last_ts = self.provider.get_image_data_by_index(
            self.rgb_stream_id, len(self) - 1
        )[1].capture_timestamp_ns
        if self.has_online_calibration:
            first_ts = self.mps_data_provider.get_rgb_corrected_timestamp_ns(
                first_ts, TimeQueryOptions.CLOSEST
            )
            last_ts = self.mps_data_provider.get_rgb_corrected_timestamp_ns(
                last_ts, TimeQueryOptions.CLOSEST
            )
        fps = len(self) / (last_ts - first_ts) * 1e9
        return MpsMetadata(fps=fps)

    def __getitem__(self, idx) -> MpsStruct:
        if get_worker_info() is None:
            mps_data_provider, provider = self.mps_data_provider, self.provider
        else:
            warnings.warn(
                "You're multiprocessing the MpsDataset iteration, which will severely slow down dataloading!"
            )
            mps_data_provider, provider = self._init_providers()

        rgb_data = provider.get_image_data_by_index(self.rgb_stream_id, idx)
        assert rgb_data[0] is not None, "no rgb frame"
        rgb = np.copy(rgb_data[0].to_numpy_array())
        capture_timestamp_ns = rgb_data[1].capture_timestamp_ns

        # rgb camera intrinsics
        if self.has_online_calibration:
            capture_timestamp_ns = mps_data_provider.get_rgb_corrected_timestamp_ns(
                capture_timestamp_ns, TimeQueryOptions.CLOSEST
            )
            rgb_pose = mps_data_provider.get_rgb_corrected_closed_loop_pose(
                capture_timestamp_ns, TimeQueryOptions.CLOSEST
            ).to_matrix()
            rgb_calib = mps_data_provider.get_online_calibration(
                capture_timestamp_ns, TimeQueryOptions.CLOSEST
            ).get_camera_calib("camera-rgb")
        else:
            # rgb_pose = provider.get_imu_data_by_time_ns(self.rgb_stream_id, capture_timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
            rgb_pose = None  # TODO: compute this with offline imu readings
            rgb_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")

        rgb_linear_calib = calibration.get_linear_camera_calibration(
            int(rgb_calib.get_image_size()[0]),
            int(rgb_calib.get_image_size()[1]),
            rgb_calib.get_focal_lengths()[0],
            "camera-rgb",
            rgb_calib.get_transform_device_camera(),
        )
        fx, fy = rgb_linear_calib.get_focal_lengths()
        cx, cy = rgb_linear_calib.get_principal_point()
        k = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

        # rgb image
        rgb = calibration.devignetting(rgb, self.devignetting_mask).astype(np.uint8)
        rgb = calibration.distort_by_calibration(
            rgb,
            rgb_linear_calib,
            rgb_calib,
            InterpolationMethod.BILINEAR,
        )
        rgb = np.rot90(rgb, k=-1)
        rgb = np.ascontiguousarray(rgb)

        # 6dof wrist and palm poses
        if mps_data_provider is not None:
            hand6dof = mps_data_provider.get_wrist_and_palm_pose(
                capture_timestamp_ns, TimeQueryOptions.CLOSEST
            )
            left_wrist, left_palm, right_wrist, right_palm = (
                homogenize_mps_wrist_and_palm(
                    hand6dof,
                    rgb_calib.get_transform_device_camera().inverse().to_matrix(),
                    threshold=0.9,
                )
            )
        else:
            left_wrist = left_palm = right_wrist = right_palm = None

        return MpsStruct(
            idx=idx,
            ts=capture_timestamp_ns,
            rgb=rgb,
            k=k,
            c2w=rgb_pose,
            online_calibration=rgb_linear_calib,
            left_wrist=left_wrist,
            left_palm=left_palm,
            right_wrist=right_wrist,
            right_palm=right_palm,
        )

    def __len__(self):
        return self.provider.get_num_data(self.rgb_stream_id)


@dataclass
class PreprocessedStruct:
    eeff: np.ndarray
    index: np.ndarray
    thumb: np.ndarray
    grasp: bool
    rgb: np.ndarray


class PreprocessedDataset(Dataset):
    """Dataset wrapper around reading from large numpy arrays"""

    # TODO: implement data reading with `np.memmap`

    def __init__(self, demonstration_dir: str):
        super().__init__()

        self.video_path = os.path.join(demonstration_dir, "original.mp4")
        self.total_frames = int(
            cv2.VideoCapture(self.video_path).get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.palm = np.load(
            os.path.join(demonstration_dir, "palm.npy"),
        )
        self.wrist = np.load(os.path.join(demonstration_dir, "wrist.npy"))
        self.index = np.load(os.path.join(demonstration_dir, "index.npy"))
        self.thumb = np.load(os.path.join(demonstration_dir, "thumb.npy"))
        self.grasp = np.load(os.path.join(demonstration_dir, "grasp.npy"))

        self.eeff_to_aruco, self.index_to_aruco, self.thumb_to_aruco = (
            load_eeff_in_aruco_frame(demonstration_dir)
        )

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, rgb = cap.read()
        if not ret:
            rgb = None
        else:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return PreprocessedStruct(
            eeff=self.eeff_to_aruco[idx].copy(),
            index=self.index_to_aruco[idx].copy(),
            thumb=self.thumb_to_aruco[idx].copy(),
            grasp=bool(self.grasp[idx]),
            rgb=rgb,
        )
