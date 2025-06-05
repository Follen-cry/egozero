import argparse
import os
import sys

import cv2
import imageio.v3 as iio
import numpy as np
import torch

try:
    from hamer.hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.hamer.models import DEFAULT_CHECKPOINT, load_hamer
    from hamer.hamer.utils import recursive_to
    from hamer.hamer.utils.renderer import Renderer, cam_crop_to_full
    from hamer.vitpose_model import ViTPoseModel
except ModuleNotFoundError:
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.models import DEFAULT_CHECKPOINT, load_hamer
    from hamer.utils import recursive_to
    from hamer.utils.renderer import Renderer, cam_crop_to_full
    from vitpose_model import ViTPoseModel

from projectaria_tools.core import mps
from scipy.spatial.transform import Rotation as R
from torch.utils.data._utils.collate import default_collate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ultralytics import YOLO
from WiLoR.wilor.configs import get_config
from WiLoR.wilor.models.wilor import WiLoR as WiLoRModel

from point_policy.robot_utils.franka.utils import rigid_transform_3D
from utils.io_utils import suppress

R_opencv_to_aria = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
T_opencv_to_aria = np.eye(4)
T_opencv_to_aria[:3, :3] = R_opencv_to_aria


def homogenize_mps_wrist_and_palm(
    wrist_and_palm_pose: mps.hand_tracking.WristAndPalmPose,
    T_camera_to_device: np.ndarray,
    threshold: float = 0.5,
):
    """Processes WristAndPalmPose MPS data object into 4x4 homogeneous poses in camera frame"""
    left_hand = wrist_and_palm_pose.left_hand
    right_hand = wrist_and_palm_pose.right_hand

    def get_palm_wrist_rotations(hand):
        wrist = None
        palm = None
        if hand.confidence > threshold:
            wrist_normal = hand.wrist_and_palm_normal_device.wrist_normal_device
            wrist_position = hand.wrist_position_device
            palm_normal = hand.wrist_and_palm_normal_device.palm_normal_device
            palm_position = hand.palm_position_device

            if np.linalg.norm(wrist_normal) > 0 and np.linalg.norm(palm_normal) > 0:
                palm_wrist_vector = palm_position - wrist_position
                palm_wrist_vector /= np.linalg.norm(palm_wrist_vector)

                wrist_normal = wrist_normal / np.linalg.norm(wrist_normal)
                palm_normal = palm_normal / np.linalg.norm(palm_normal)
                wrist_3rd_vector = np.cross(palm_wrist_vector, wrist_normal)
                wrist_3rd_vector /= np.linalg.norm(wrist_3rd_vector)
                palm_3rd_vector = np.cross(palm_wrist_vector, palm_normal)
                palm_3rd_vector /= np.linalg.norm(palm_3rd_vector)

                wrist = np.eye(4)
                wrist[:3, :3] = np.column_stack(
                    [wrist_3rd_vector, palm_wrist_vector, wrist_normal]
                )
                wrist[:3, 3] = wrist_position
                wrist = T_opencv_to_aria @ T_camera_to_device @ wrist

                palm = np.eye(4)
                palm[:3, :3] = np.column_stack(
                    [palm_3rd_vector, palm_wrist_vector, palm_normal]
                )
                palm[:3, 3] = palm_position
                palm = T_opencv_to_aria @ T_camera_to_device @ palm

        return wrist, palm

    left_wrist, left_palm = get_palm_wrist_rotations(left_hand)
    right_wrist, right_palm = get_palm_wrist_rotations(right_hand)
    return left_wrist, left_palm, right_wrist, right_palm


@torch.no_grad()
def run_hamer_from_video(
    video,
    checkpoint=DEFAULT_CHECKPOINT,
    body_detector="vitdet",
    render=True,
    is_right_hand=False,
):
    import ray

    in_ray_worker = (
        ray.is_initialized() and ray.get_runtime_context().get_job_id() is not None
    )
    if in_ray_worker:
        from ray.experimental.tqdm_ray import tqdm
    else:
        from tqdm import tqdm

    with suppress(stdout=True):
        os.chdir(os.path.join(os.path.dirname(__file__), "..", "hamer"))
        model, model_cfg, detector, cpm, device, renderer = load_hamer_model(
            checkpoint, body_detector
        )
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    if isinstance(video, str):
        frames = iio.imread(video)
    else:
        frames = np.stack(video)

    assert frames.ndim == 4  # shape (n, h, w, 3)

    all_frames = []
    all_fingertips = []
    n_detected = 0
    n_missing = 0
    pbar = tqdm(total=len(frames), desc="processing hamer")
    for frame in frames:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            fingertips, hamer_frame = detect_hamer_in_frame(
                model,
                model_cfg,
                detector,
                cpm,
                device,
                renderer,
                frame,
                render=render,
                is_right_hand=is_right_hand,
            )
            if fingertips is not None:
                n_detected += 1
            else:
                n_missing += 1

        hamer_frame_rgb = cv2.cvtColor(hamer_frame, cv2.COLOR_BGR2RGB)
        hamer_frame_rgb_uint8 = np.uint8(hamer_frame_rgb)
        all_frames.append(hamer_frame_rgb_uint8)
        all_fingertips.append(fingertips)

        if not in_ray_worker:
            pbar.set_postfix(dict(detected=n_detected, missing=n_missing))
        pbar.update(1)

    pbar.close()
    del pbar
    return all_frames, all_fingertips


def load_hamer_model(checkpoint, body_detector):
    model, model_cfg = load_hamer(checkpoint)

    # Setup HaMeR model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Load detector
    from pathlib import Path

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    detector = None
    if body_detector == "vitdet":
        from detectron2.config import LazyConfig

        try:
            from hamer import hamer

            cfg_path = (
                Path(hamer.__file__).parent
                / "configs"
                / "cascade_mask_rcnn_vitdet_h_75ep.py"
            )
        except:
            import hamer

            cfg_path = (
                Path(hamer.__file__).parent
                / "configs"
                / "cascade_mask_rcnn_vitdet_h_75ep.py"
            )

        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif body_detector == "regnety":
        from detectron2 import model_zoo

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    return model, model_cfg, detector, cpm, device, renderer


def detect_hamer_in_frame(
    model,
    model_cfg,
    detector,
    cpm,
    device,
    renderer,
    img_cv2,
    render=True,
    focal_length=torch.tensor([610.51381834], device="cuda"),
    rescale_factor=2,
    is_right_hand=False,
):
    # Detect humans in image
    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]
    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img_cv2,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )
    bboxes = []
    bbox_sizes = []
    is_hand = []

    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]

        # Rejecting not confident detections and left hand
        if is_right_hand:
            keyp = right_hand_keyp
        else:
            keyp = left_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if bbox_size < 1000:
                continue
            bboxes.append(bbox)
            bbox_sizes.append(bbox_size)
            is_hand.append(1 if is_right_hand else 0)

    if len(bboxes) == 0:
        return None, img

    # Choose largest bbox prediction
    # largest_bbox_idx = np.argmax(bbox_sizes)
    # boxes = np.array([bboxes[largest_bbox_idx]])
    # is_hand = np.array([1]) if is_right_hand else np.array([0])

    boxes = np.array(bboxes)
    is_hand = np.array(is_hand)

    # Run reconstruction on all detected hands
    dataset = ViTDetDataset(
        model_cfg, img, boxes, is_hand, rescale_factor=rescale_factor
    )
    all_predicted_3d = []
    all_verts = []
    all_cam_t = []

    with suppress(stdout=True):

        for i in range(len(dataset)):
            batch = default_collate([dataset[i]])
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            all_predicted_3d.append(out["pred_keypoints_3d"])

            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()

            # scaled_focal_length = (
            #     model_cfg.EXTRA.FOCAL_LENGTH
            #     / model_cfg.MODEL.IMAGE_SIZE
            #     * img_size.max()
            # )
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_length)
                .detach()
                .cpu()
                .numpy()
            )

            verts = out["pred_vertices"][0].detach().cpu().numpy()
            is_right = batch["right"][0].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            cam_t = pred_cam_t_full[0].reshape(3)
            all_verts.append(verts)
            all_cam_t.append(cam_t)

        closest_hand_idx = np.argmin(np.array(all_cam_t)[:, -1])
        verts = all_verts[closest_hand_idx]
        cam_t = all_cam_t[closest_hand_idx]
        fingertips = all_predicted_3d[closest_hand_idx].cpu().numpy().squeeze(0)

        if render:
            misc_args = dict(
                mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
                scene_bg_color=(1, 1, 1),
                focal_length=focal_length,
            )
            cam_view = renderer.render_rgba_multiple(
                [verts],
                cam_t=[cam_t],
                render_res=img_size[0],
                is_right=[1 if is_right_hand else 0],
                **misc_args,
            )

            input_img = img.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate(
                [input_img, np.ones_like(input_img[:, :, :1])], axis=2
            )  # Add alpha channel
            input_img_overlay = (
                input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                + cam_view[:, :, :3] * cam_view[:, :, 3:]
            )
        else:
            input_img_overlay = img.astype(np.float32)[:, :, ::-1] / 255.0

        if not is_right_hand:  # Flip x coordinates if left hand
            fingertips[:, 0] = -1 * fingertips[:, 0]
        return (
            fingertips,
            255 * input_img_overlay[:, :, ::-1],
        )


def load_wilor_model(checkpoint, cfg_path, detector_path):
    if isinstance(checkpoint, tuple):
        checkpoint = checkpoint[0]

    if isinstance(cfg_path, tuple):
        cfg_path = cfg_path[0]

    print("Loading ", checkpoint)
    model_cfg = get_config(cfg_path, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if ("vit" in model_cfg.MODEL.BACKBONE.TYPE) and (
        "BBOX_SHAPE" not in model_cfg.MODEL
    ):

        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

        # Update config to be compatible with demo

    if "DATA_DIR" in model_cfg.MANO:
        model_cfg.defrost()
        model_cfg.MANO.DATA_DIR = "./mano_data/"
        model_cfg.MANO.MODEL_PATH = "./mano_data/"
        model_cfg.MANO.MEAN_PARAMS = "./mano_data/mano_mean_params.npz"
        model_cfg.freeze()

    model = WiLoRModel.load_from_checkpoint(checkpoint, strict=False, cfg=model_cfg)

    # Setup wilor model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    if isinstance(detector_path, tuple):
        detector_path = detector_path[0]
    detector = YOLO(detector_path)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)

    return model, model_cfg, detector, device, renderer, renderer_side


def detect_wilor_in_frame(
    model,
    model_cfg,
    detector,
    device,
    renderer,
    img_cv2,
    render=True,
    focal_length=torch.tensor([610.51381834], device="cuda"),
    rescale_factor=2,
    is_right_hand=True,
):

    img = img_cv2.copy()[:, :, ::-1]
    detections = detector(img_cv2, conf=0.3, verbose=False)[0]
    bboxes = []
    is_hand = []
    for det in detections:
        Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        if is_right_hand == det.boxes.cls.cpu().detach().squeeze().item():
            is_hand.append(det.boxes.cls.cpu().detach().squeeze().item())
            bbox = Bbox[:4].tolist()
            bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if bbox_size < 1000:
                continue
            bboxes.append(bbox)

    if len(bboxes) == 0:
        return None, img

    boxes = np.array(bboxes)
    is_hand = np.array(is_hand)
    dataset = ViTDetDataset(
        model_cfg, img_cv2, boxes, is_hand, rescale_factor=rescale_factor
    )
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    all_predicted_3d = []
    all_verts = []
    all_cam_t = []

    with suppress(stdout=True):
        for i in range(len(dataset)):
            batch = default_collate([dataset[i]])
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            all_predicted_3d.append(out["pred_keypoints_3d"])

            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()

            # scaled_focal_length = (
            #     model_cfg.EXTRA.FOCAL_LENGTH
            #     / model_cfg.MODEL.IMAGE_SIZE
            #     * img_size.max()
            # )
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_length)
                .detach()
                .cpu()
                .numpy()
            )
            verts = out["pred_vertices"][0].detach().cpu().numpy()
            is_right = batch["right"][0].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            cam_t = pred_cam_t_full[0].reshape(3)
            all_verts.append(verts)
            all_cam_t.append(cam_t)

        closest_hand_idx = np.argmin(np.array(all_cam_t)[:, -1])
        verts = all_verts[closest_hand_idx]
        cam_t = all_cam_t[closest_hand_idx]
        fingertips = all_predicted_3d[closest_hand_idx].cpu().numpy().squeeze(0)

        if render:
            misc_args = dict(
                mesh_base_color=(0.25098039, 0.274117647, 0.65882353),
                scene_bg_color=(1, 1, 1),
                focal_length=focal_length,
            )
            cam_view = renderer.render_rgba_multiple(
                [verts],
                cam_t=[cam_t],
                render_res=img_size[0],
                is_right=[1 if is_right_hand else 0],
                **misc_args,
            )

            input_img = img.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate(
                [input_img, np.ones_like(input_img[:, :, :1])], axis=2
            )  # Add alpha channel
            input_img_overlay = (
                input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
                + cam_view[:, :, :3] * cam_view[:, :, 3:]
            )
        else:
            input_img_overlay = img.astype(np.float32)[:, :, ::-1] / 255.0

        if not is_right_hand:  # Flip x coordinates if left hand
            fingertips[:, 0] = -1 * fingertips[:, 0]
        return (
            fingertips,
            255 * input_img_overlay[:, :, ::-1],
        )


@torch.no_grad()
def run_wilor_from_video(
    video,
    checkpoint="./pretrained_models/wilor_final.ckpt",
    cfg_path="./pretrained_models/model_config.yaml",
    detector_path="./pretrained_models/detector.pt",
    render=True,
    is_right_hand=False,
):
    import ray

    in_ray_worker = (
        ray.is_initialized() and ray.get_runtime_context().get_job_id() is not None
    )
    if in_ray_worker:
        from ray.experimental.tqdm_ray import tqdm
    else:
        from tqdm import tqdm

    with suppress(stdout=True):
        os.chdir(os.path.join(os.path.dirname(__file__), "..", "WiLoR"))
        model, model_cfg, detector, device, renderer, _ = load_wilor_model(
            checkpoint, cfg_path, detector_path
        )
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    if isinstance(video, str):
        frames = iio.imread(video)
    else:
        frames = np.stack(video)

    assert frames.ndim == 4  # shape (n, h, w, 3)

    all_frames = []
    all_fingertips = []
    n_detected = 0
    n_missing = 0
    pbar = tqdm(total=len(frames), desc="processing wilor")
    for frame in frames:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            fingertips, wilor_frame = detect_wilor_in_frame(
                model,
                model_cfg,
                detector,
                device,
                renderer,
                frame,
                render=render,
                is_right_hand=is_right_hand,
            )
            if fingertips is not None:
                n_detected += 1
            else:
                n_missing += 1

        wilor_frame_rgb = cv2.cvtColor(wilor_frame, cv2.COLOR_BGR2RGB)
        wilor_frame_rgb_uint8 = np.uint8(wilor_frame_rgb)
        all_frames.append(wilor_frame_rgb_uint8)
        all_fingertips.append(fingertips)

        if not in_ray_worker:
            pbar.set_postfix(dict(detected=n_detected, missing=n_missing))
        pbar.update(1)

    pbar.close()
    del pbar
    return all_frames, all_fingertips


def correct_hand_model_from_aria(fingerskeypoints, aria_poses):

    def vector_projection(projected_vector, plane_normal, correct_vector):
        vector_proj = (
            projected_vector - np.dot(projected_vector, plane_normal) * plane_normal
        )
        # Find angle between z axis of hamer model and aria model
        cos_theta = np.dot(vector_proj, correct_vector) / (
            np.linalg.norm(vector_proj) * np.linalg.norm(correct_vector)
        )
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        rotvec = plane_normal * (angle / 2)
        rotation = R.from_rotvec(rotvec).as_matrix()
        return rotation

    correct_fingerskeypoints = []
    R_hand_poses = []

    for fingerkeypoints, aria_pose in zip(fingerskeypoints, aria_poses):

        R_aria = aria_pose[:3, :3]
        palm2wrist = fingerkeypoints[9] - fingerkeypoints[0]
        palm2wrist = palm2wrist / np.linalg.norm(palm2wrist)

        mid_cmc2index_cmc = fingerkeypoints[5] - fingerkeypoints[9]
        mid_cmc2index_cmc = mid_cmc2index_cmc / np.linalg.norm(mid_cmc2index_cmc)
        third_vector = np.cross(mid_cmc2index_cmc, palm2wrist)

        R_hand_model = np.column_stack([mid_cmc2index_cmc, palm2wrist, third_vector])

        u, _, vh = np.linalg.svd(R_hand_model)
        R_hand_model = u @ vh

        yz_plane_normal = R_aria[:3, 0]
        R_aria_z = R_aria[:3, 2]
        R_hamer_z = R_hand_model[:3, 2]

        xy_plane_normal = R_aria[:3, 2]
        R_aria_y = R_aria[:3, 1]
        R_hamer_y = R_hand_model[:3, 1]

        xz_plane_normal = R_aria[:3, 1]
        R_aria_x = R_aria[:3, 0]
        R_hamer_x = R_hand_model[:3, 0]

        rotation_x = vector_projection(R_hamer_z, yz_plane_normal, R_aria_z)
        rotation_z = vector_projection(R_hamer_y, xy_plane_normal, R_aria_y)

        rotation = rotation_z @ rotation_x

        # vector_proj = (
        #     R_hand_model[:3, 2]
        #     - np.dot(R_hand_model[:3, 2], R_aria[:3, 0]) * R_aria[:3, 0]
        # )
        # # Find angle between z axis of hamer model and aria model
        # cos_theta = np.dot(vector_proj, R_aria[:3, 2]) / (
        #     np.linalg.norm(vector_proj) * np.linalg.norm(R_aria[:3, 2])
        # )
        # cos_theta = np.clip(cos_theta, -1.0, 1.0)
        # angle = np.arccos(cos_theta)

        # # Rotation along x axis
        # rotvec = R_aria[:3, 0] * (angle / 2)
        # rotation = R.from_rotvec(rotvec).as_matrix()

        correct_fingerkeypoints = fingerkeypoints @ rotation.T
        R_hand_pose = np.eye(4)
        R_hand_pose[:3, :3] = R_hand_model
        R_hand_pose[:3, 3] = aria_pose[:3, 3]

        correct_fingerskeypoints.append(correct_fingerkeypoints)
        R_hand_poses.append(R_hand_pose)

    correct_fingerskeypoints = np.stack(correct_fingerskeypoints)
    R_hand_poses = np.stack(R_hand_poses)

    return correct_fingerskeypoints, R_hand_poses


def absolute_ori_from_robot_frame_fingertips(
    fingerkeypoints_robot_frame,
    robot_base_orientation=R.from_rotvec([np.pi, 0, 0]).as_matrix(),
) -> np.ndarray:
    """
    Assmue cage left robot
    Args:
        fingerkeypoints_robot_frame: Array of shape (9, 3) containing right hand keypoints in robot frame
        robot_base_orientation: Initial robot orientation
    """

    y_axis = fingerkeypoints_robot_frame[5] - fingerkeypoints_robot_frame[2]
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Extract two vector to comput x axis
    vector_23 = (
        fingerkeypoints_robot_frame[3] - fingerkeypoints_robot_frame[2]
    )  # thumb cmc to mcp
    vector_56 = (
        fingerkeypoints_robot_frame[6] - fingerkeypoints_robot_frame[5]
    )  # index cmc to mcp
    x_axis = np.cross(vector_56, vector_23)
    x_axis = -x_axis / np.linalg.norm(x_axis)

    z_axis = np.cross(x_axis, y_axis)

    hand_rotation = np.column_stack([x_axis, y_axis, z_axis])
    gripper_rotation = hand_rotation @ robot_base_orientation.T

    return R.from_matrix(gripper_rotation).as_quat()  # quat x y z w


def relative_ori_from_robot_frame_fingertips(
    fingerkeypoints_robot_frame,
    base_hand_points,
    robot_base_orientation=R.from_rotvec([np.pi, 0, 0]).as_matrix(),
) -> np.ndarray:
    """
    Assmue cage left robot
    Args:
        fingerkeypoints_robot_frame: Array of shape (9, 3) containing right hand keypoints in robot frame
        base_hand_points : Initial hand keypoints in first frame
        robot_base_orientation: Initial robot orientation
    """

    current_hand_points = fingerkeypoints_robot_frame.copy()
    # find the rotation matrix between the base hand points and the current hand points
    rot, pos = rigid_transform_3D(base_hand_points, current_hand_points)
    robot_ori = rot @ robot_base_orientation.T

    return R.from_matrix(robot_ori).as_quat()  # quat x y z w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to MPS server output (directory)",
    )
    args = parser.parse_args()

    frames = list(iio.imiter(args.video_path))
    all_frames, _ = run_hamer_from_video(frames[0:200], is_right_hand=False)
    hamer_video = os.path.join(args.mps_sample_path, "output_rgb_hamer.mp4")
    iio.imwrite(hamer_video, all_frames, fps=30, codec="libx264")
