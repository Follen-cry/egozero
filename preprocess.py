"""
Preprocess MPS data into robot-replayable format
"""

import argparse
import json
import multiprocessing
import os
import pickle
import time
from collections import deque
from functools import partial
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
import yaml
from cotracker.predictor import CoTrackerOnlinePredictor
from PIL import Image
from scipy.ndimage import uniform_filter1d
from torch.utils.data import DataLoader
from tqdm import tqdm

from point_policy.point_utils.correspondence import Correspondence
from utils.data_utils import MpsDataset, MpsStruct
from utils.depth_utils import robust_ransac_triangulation
from utils.hand_utils import (
    T_opencv_to_aria,
    correct_hand_model_from_aria,
    run_hamer_from_video,
    run_wilor_from_video,
)
from utils.io_utils import concatenate_frames, jsonify, save_video
from utils.transform_utils import (
    filter_and_interpolate_fingertips,
    filter_and_interpolate_poses,
    trimmed_average_poses,
)
from utils.vis_utils import add_border, detect_aruco, draw_axis, plot_points

os.environ["TOKENIZERS_PARALLELISM"] = "false"

GPU_FRAC = 0.5  # hamer requires ~12GB and horizon has 48GB RAM
CPU_FRAC = 8  # horizon has 64 cores
RAY_SPILL_DIR = os.path.abspath(
    os.path.expanduser("./src/raylogs")
)  # annoying ray logs

# preprocessing flags
THRESHOLD = 0.07  # threshold to grasp between index/thumb


def suppress_short_detected_segments(lst, n):
    """If contiguous segments are shorter than length `n`, set them to `None`"""
    count = 0
    indices = []

    for i, val in enumerate(lst):
        if val is not None:
            count += 1
        else:
            if 0 < count < n:
                indices.extend(range(i - count, i))
            count = 0

    if 0 < count < n:
        indices.extend(range(len(lst) - count, len(lst)))

    for i in indices:
        lst[i] = None

    return lst


def estimate_a2w(rgbs, c2ws, k, d, save_path):
    fn = partial(detect_aruco, k=k, d=d)
    with multiprocessing.Pool(processes=128) as pool:
        a2cs = list(
            tqdm(
                pool.imap(fn, rgbs),
                total=len(rgbs),
                ncols=80,
                desc="detecting aruco",
                leave=False,
            )
        )

    a2ws = []
    for i, a2c in enumerate(a2cs):
        if a2c is not None:
            a2w = np.linalg.inv(c2ws[0]) @ c2ws[i] @ a2c
            a2ws.append(a2w)

    print(f"found {len(a2ws)} aruco measurements")
    a2w = trimmed_average_poses(
        np.stack(a2ws),
        lower_quantile=0.3,
        upper_quantile=0.7,
        save_path=save_path,
    )
    return a2w


def crop_center(img: np.ndarray, crop_size: int = 512):
    h, w = img.shape[:2]
    ch, cw = crop_size
    top = (h - ch) // 2
    left = (w - cw) // 2
    return img[top : top + ch, left : left + cw]


@ray.remote(num_gpus=GPU_FRAC, num_cpus=CPU_FRAC)
def process_single_demo(*args, **kwargs):
    """Exception handling wrapper function"""
    try:
        return _process_single_demo(*args, **kwargs)
    except:
        import traceback

        print("\033[91m" + traceback.format_exc() + "\033[0m")


@torch.no_grad()
def _process_single_demo(
    job_id: int,
    save_dir: str,
    rgbs: List[np.ndarray],
    k: np.ndarray,
    d: np.ndarray,
    palm: List[np.ndarray],
    wrist: List[np.ndarray],
    pose: List[np.ndarray],  # in frame of session (demo=0)
    prompts: List[str] = [],
    visualize: bool = True,
    threshold: float = THRESHOLD,
    is_right_hand: bool = False,
    is_wilor: bool = False,
):
    """Postprocess hand_model/aruco/mps detections for a single demonstration"""
    # notes on notation and variables
    # define world frame (w)  := first frame of demo
    # define global frame (g) := first frame of recording session
    g2w = np.linalg.inv(pose[0])
    orig_rgbs = np.copy(rgbs)
    first_frame = np.copy(rgbs[0])
    c2g = np.copy(pose)  # for cotracking and triangulation later

    # throw out segments of detections that are shorter than 0.5 seconds.
    # work well on right hand. Set to 0 since don't need this.
    min_mps_detection_frames = 0
    indices, palm = filter_and_interpolate_poses(
        suppress_short_detected_segments(palm, min_mps_detection_frames), filter=False
    )
    indices, wrist = filter_and_interpolate_poses(
        suppress_short_detected_segments(wrist, min_mps_detection_frames), filter=False
    )
    mps_start, mps_end = indices.min(), indices.max()

    # postprocess hand_model predictions
    # --------------------------------

    render_hand_model = False
    hand_model_fn = run_wilor_from_video if is_wilor else run_hamer_from_video
    hand_model_rgbs, fingertips = hand_model_fn(
        rgbs, render=render_hand_model, is_right_hand=is_right_hand
    )  # rendering causes ~2x slowdown

    # interpolate missing hand model frames
    indices, fingertips = filter_and_interpolate_fingertips(fingertips)
    hand_model_start, hand_model_end = indices.min(), indices.max()

    # construct payloads
    # --------------------------------

    global_start = max(mps_start, hand_model_start)
    global_end = min(mps_end, hand_model_end) + 1

    palm = palm[global_start - mps_start : global_end - mps_start]
    wrist = wrist[global_start - mps_start : global_end - mps_start]
    fingertips = fingertips[
        global_start - hand_model_start : global_end - hand_model_start
    ]
    rgbs = rgbs[global_start:global_end]
    # orig_rgbs = orig_rgbs[global_start:global_end]  # TODO: do we need original video to match the trimmed video?
    hand_model_rgbs = hand_model_rgbs[global_start:global_end]
    pose = pose[global_start:global_end]

    # construct index/thumb in camera frame arrays
    fingertips, _ = correct_hand_model_from_aria(fingertips, palm)
    hand_model_palm = (fingertips[:, 1] + fingertips[:, 5] + fingertips[:, 9]) / 3
    index = fingertips[:, 8] - hand_model_palm + palm[:, :3, 3]
    thumb = fingertips[:, 4] - hand_model_palm + palm[:, :3, 3]

    fingers_keypoints = (
        fingertips[:, :, :] - hand_model_palm[:, None, :] + palm[:, :3, 3][:, None, :]
    )

    assert (
        fingers_keypoints.ndim == 3
        and fingers_keypoints.shape[1] == 21
        and fingers_keypoints.shape[2] == 3
    )

    # map index/thumb/palm/wrist into first frame
    index2w, thumb2w, palm2w, wrist2w = [], [], [], []
    for i in range(global_end - global_start):
        T_index_aria = np.eye(4)
        T_index_aria[:3, 3] = index[i]
        T_thumb_aria = np.eye(4)
        T_thumb_aria[:3, 3] = thumb[i]
        T_palm_aria = palm[i]
        T_wrist_aria = wrist[i]

        index2w.append((g2w @ pose[i] @ T_index_aria)[:3, 3])
        thumb2w.append((g2w @ pose[i] @ T_thumb_aria)[:3, 3])
        palm2w.append(g2w @ pose[i] @ T_palm_aria)
        wrist2w.append(g2w @ pose[i] @ T_wrist_aria)

    index2w = np.stack(index2w)
    thumb2w = np.stack(thumb2w)
    palm2w = np.stack(palm2w)
    wrist2w = np.stack(wrist2w)

    # majority filter to deflicker grasp detection
    distance = np.linalg.norm(index - thumb, axis=-1, ord=2)
    filter_size = int(fps / 3.0)
    grasp = distance < threshold
    grasp = uniform_filter1d(grasp.astype(np.float32), size=filter_size, mode="nearest")
    grasp = (grasp > 0.5).astype(bool)
    grasp[: filter_size // 2] = grasp[filter_size]
    grasp[-filter_size // 2 :] = grasp[filter_size]

    # triangulate depth from corrspondence + cotracked points
    # --------------------------------

    # grounded dino for computing bounding box origin of scene
    label_keypoints_image = Image.open(os.path.join(save_dir, "label_keypoints.png"))
    with open(os.path.join(save_dir, "label_keypoints.pkl"), "rb") as f:
        label_keypoints_coords = np.array(pickle.load(f))
    correspondence = Correspondence(device="cuda", use_segmentation=(len(prompts) > 0))
    correspondence.set_expert_correspondence(label_keypoints_image, prompts)
    torch.cuda.empty_cache()
    dift_coords, dift_image = correspondence.find_correspondence(
        Image.fromarray(first_frame), label_keypoints_coords
    )
    dift_coords = dift_coords.reshape(1, -1, 3)
    num_tracking_points = dift_coords.shape[1]

    window_len = 16
    cotracker = (
        CoTrackerOnlinePredictor(
            checkpoint="./checkpoints/scaled_online.pth", window_len=window_len
        )
        .to(device)
        .eval()
    )

    # so the indexing/slicing of the arrays is a bit tricky here
    #
    # we want to compute t* in world frame, so start frame needs to be 0 wrt the untrimmed array
    # we don't want to use frames after the object is moved, so end frame needs to be grasp_start wrt trimmed array
    #
    # start = 0
    # end = global_start + grasp_start - buffer
    pre_grasp_buffer = 15  # 0.5 seconds
    tracking_start = 0
    tracking_end = (
        global_start + np.nonzero(grasp.astype(int))[0][0].item() - pre_grasp_buffer
    )  # first true index
    num_tracking_frames = int(tracking_end - tracking_start)

    # crop image and shift coordinates
    crop_h = crop_w = 768
    orig_h, orig_w = orig_rgbs[0].shape[:2]
    offset_y = (orig_h - crop_h) // 2
    offset_x = (orig_w - crop_w) // 2
    cropped_coords = np.copy(dift_coords)
    cropped_coords[..., 0] = window_len - 2
    cropped_coords[..., 1] -= offset_y  # y
    cropped_coords[..., 2] -= offset_x  # x

    buffer = [crop_center(orig_rgbs[0], (crop_h, crop_w)) for _ in range(window_len)]
    tracked_uvs = []
    for i in range(num_tracking_frames):
        buffer.pop(0)
        buffer.append(crop_center(orig_rgbs[i], (crop_h, crop_w)))
        video_chunk = torch.tensor(
            np.stack(buffer), dtype=torch.float32, device=device
        ).permute(0, 3, 1, 2)[None]

        if i == 0:
            cotracker(
                video_chunk=video_chunk[0, 0].unsqueeze(0).unsqueeze(0),
                is_first_step=True,
                add_support_grid=True,
                queries=torch.tensor(
                    cropped_coords, device=device, dtype=torch.float32
                ),
            )

        pred_tracks, _ = cotracker(video_chunk, one_frame=True)
        # remove support points, take predictions on last frame
        pred_tracks = pred_tracks[0, -1, :num_tracking_points, :]
        # shift coordinates from cropped to uncropped frame
        pred_tracks = pred_tracks.reshape(-1, 2).detach().cpu().numpy()
        pred_tracks[:, 0] += offset_y
        pred_tracks[:, 1] += offset_x
        tracked_uvs.append(pred_tracks)

    assert len(tracked_uvs) == num_tracking_frames
    tracked_uvs = np.stack(tracked_uvs)

    # in case there is drift between dift and what's predicted by cotracker in the first frame
    drift = tracked_uvs[:1, :, :] - dift_coords[:, :, 1:]
    tracked_uvs -= drift

    # we want camera poses to be in world frame, not global frame
    c2ws = np.einsum(
        "ij,njk->nik",
        np.linalg.inv(c2g[tracking_start]),
        c2g[tracking_start:tracking_end],
    )
    opt_results = []
    for i in range(num_tracking_points):
        opt_results.append(
            robust_ransac_triangulation(
                tracked_uvs[:, i, :], k, c2ws, ransac_thresh=5, ransac_iters=1000
            )
        )
    opt_results = {k: [d[k] for d in opt_results] for k in opt_results[0]}

    cmap = plt.get_cmap("viridis", num_tracking_points)
    uv_colors = [
        tuple(int(c * 255) for c in cmap(j / num_tracking_points)[:3])
        for j in range(num_tracking_points)
    ]
    uv_rgbs = []
    for i in range(num_tracking_frames):
        rgb1 = np.copy(orig_rgbs[i])
        rgb2 = np.copy(orig_rgbs[i])
        for j in range(num_tracking_points):
            rgb1 = cv2.circle(
                rgb1,
                (
                    int(opt_results["reprojected_uvs"][j][i][0]),
                    int(opt_results["reprojected_uvs"][j][i][1]),
                ),
                7,
                uv_colors[j],
                -1,
            )
            rgb2 = cv2.circle(
                rgb2,
                (int(tracked_uvs[i, j, 0]), int(tracked_uvs[i, j, 1])),
                7,
                uv_colors[j],
                -1,
            )
        uv_rgbs.append(concatenate_frames(rgb1, rgb2))

    # visualize
    # --------------------------------

    if visualize:
        annotated_rgbs = []
        annotated_rgbs_moving = []
        colors = [
            (255, 0, 255),
            (255, 255, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 0),
        ]  # colors to plot hand

        for i in range(global_end - global_start):

            # use first_frame instead of rgbs[0] since rgbs array is truncated above
            rgb = np.copy(np.ascontiguousarray(first_frame).astype(np.uint8))

            # grasp
            rgb = add_border(
                rgb,
                text=f"{distance[i]:.4f}",
                color=(0, 255, 0) if grasp[i] else (255, 0, 0),
            )

            # aruco
            # rgb, _ = draw_axis(
            #     rgb,
            #     a2w,
            #     k,
            #     d,
            #     upper_left=f"aruco z: {a2w[2, 3]:.4f}",
            # )
            # a2c = np.linalg.inv(pose[i]) @ np.linalg.inv(g2w) @ a2w
            # hand_model_rgbs[i], _ = draw_axis(
            #     hand_model_rgbs[i],
            #     a2c,
            #     k,
            #     d,
            #     upper_left=f"aruco z: {a2c[2, 3]:.4f}",
            # )

            # palm/wrist
            rgb, _ = draw_axis(
                rgb,
                palm2w[i],
                k,
                d,
                upper_right=f"palm z: {palm2w[i][2, 3]:.4f}",
            )
            rgb, _ = draw_axis(rgb, wrist2w[i], k, d)

            # index/thumb
            rgb = plot_points(
                rgb,
                np.eye(4),
                k,
                d,
                points=index2w[i],
                color=(232, 168, 0),
                radius=5,
                thickness=-1,
            )
            rgb = plot_points(
                rgb,
                np.eye(4),
                k,
                d,
                points=thumb2w[i],
                color=(0, 135, 255),
                radius=5,
                thickness=-1,
            )

            annotated_rgbs.append(rgb)

            # index/thumb in moving frame
            rgb = np.copy(rgbs[i])

            plot_points(  # plot wrist with black
                rgb,
                np.eye(4),
                k,
                d,
                points=fingers_keypoints[i][0],
                color=(0, 0, 0),
                radius=5,
                thickness=-1,
            )

            for j in range(20):
                plot_points(
                    rgb,
                    np.eye(4),
                    k,
                    d,
                    points=fingers_keypoints[i][j + 1],
                    color=colors[j // 4],
                    radius=5,
                    thickness=-1,
                )

            # moving palm/wrist
            rgb, _ = draw_axis(
                rgb,
                palm[i],
                k,
                d,
                upper_right=f"middle2wrist: {np.linalg.norm(fingers_keypoints[i][0] - fingers_keypoints[i][9]):.4f}",
            )
            rgb, _ = draw_axis(
                rgb,
                wrist[i],
                k,
                d,
                upper_left=f"ring2wrist: {np.linalg.norm(fingers_keypoints[i][0] - fingers_keypoints[i][13]):.4f}",
            )

            annotated_rgbs_moving.append(rgb)

    # save payloads
    # --------------------------------

    save_dir = os.path.join(save_dir, f"demonstration_{job_id:05d}")
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(first_frame).save(os.path.join(save_dir, "first_frame.png"))
    np.save(os.path.join(save_dir, "first_frame_g2w.npy"), g2w)

    save_video(os.path.join(save_dir, "original.mp4"), orig_rgbs, fps=fps)
    np.save(os.path.join(save_dir, "palm.npy"), np.array(palm))
    np.save(os.path.join(save_dir, "wrist.npy"), np.array(wrist))
    np.save(os.path.join(save_dir, "index.npy"), np.array(index))
    np.save(os.path.join(save_dir, "thumb.npy"), np.array(thumb))
    np.save(os.path.join(save_dir, "grasp.npy"), np.array(grasp))
    np.save(os.path.join(save_dir, "pose.npy"), np.array(pose))
    np.save(
        os.path.join(save_dir, "fingers_keypoints.npy"), np.array(fingers_keypoints)
    )  # N x 9 x 3
    if visualize:
        save_video(
            os.path.join(save_dir, "annotated_actions.mp4"), annotated_rgbs, fps=fps
        )
        save_video(
            os.path.join(save_dir, "annotated_actions_moving.mp4"),
            annotated_rgbs_moving,
            fps=fps,
        )
        if render_hand_model:
            save_video(
                os.path.join(save_dir, "annotated_hand.mp4"), hand_model_rgbs, fps=fps
            )
        save_video(os.path.join(save_dir, "annotated_states.mp4"), uv_rgbs, fps=fps)
        dift_image.save(os.path.join(save_dir, "dift_image.png"))
        with open(os.path.join(save_dir, "triangulation.json"), "w") as f:
            opt_results.pop("reprojected_uvs")
            opt_results.pop("reprojected_xyzs")
            json.dump(jsonify(opt_results), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mps_sample_path",
        type=str,
        required=True,
        help="Path to MPS server output (directory)",
    )
    parser.add_argument(
        "--is_right_hand",
        default=False,
        action="store_true",
        help="Is right hand",
    )
    parser.add_argument(
        "--is_wilor",
        default=False,
        action="store_true",
        help="Using hamor or wilor",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Prompts to use to return a single bbox of the scene",
    )
    args = parser.parse_args()

    # the prompts are stored in the task cfg yamls
    with open(
        os.path.join(
            os.getcwd(), f"point_policy/cfgs/suite/task/franka_env/{args.task}.yaml"
        ),
        "r",
    ) as f:
        data = yaml.safe_load(f)
    args.prompts = data["prompts"]
    print(args.prompts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = os.path.join(args.mps_sample_path, "preprocess")
    os.makedirs(save_dir, exist_ok=True)

    # test the correspondence model here instead of 5 minutes into the run
    label_keypoints_image = Image.open(os.path.join(save_dir, "label_keypoints.png"))
    with open(os.path.join(save_dir, "label_keypoints.pkl"), "rb") as f:
        label_keypoints_coords = np.array(pickle.load(f))
    correspondence = Correspondence(
        device=device, use_segmentation=(len(args.prompts) > 0)
    )
    cropped_image, _ = correspondence._forward_grounded_dino(
        label_keypoints_image, args.prompts
    )
    cropped_image.save(os.path.join(save_dir, "dift_image.png"))
    print(
        f"expert image cropped from {label_keypoints_image.size} to {cropped_image.size}"
    )
    if cropped_image.size == label_keypoints_image.size:
        raise RuntimeError("cropped image is the same size as original")

    # load mps data (using pytorch dataloader is faster than indexing directly into the dataset)
    mps_dataset = MpsDataset(
        args.mps_sample_path, os.path.join(args.mps_sample_path, "sample.vrs")
    )
    mps_loader = DataLoader(
        mps_dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=0,
        collate_fn=list,
        pin_memory_device=device,
    )
    num_rgb_frames = len(mps_loader)
    fps = mps_dataset.metadata.fps

    def has_hand(mps_struct):
        return (
            (mps_struct.right_palm is not None and mps_struct.right_wrist is not None)
            if args.is_right_hand
            else (
                mps_struct.left_palm is not None and mps_struct.left_wrist is not None
            )
        )

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    ray.init(_temp_dir=RAY_SPILL_DIR, num_gpus=num_gpus, resources={"memory_slot": 5})
    print(f"initialized ray to {num_gpus} gpus!")

    # track variables
    job_refs = []
    min_duration_between_demos = 1.0  # in seconds
    min_demo_frames = fps * 3
    min_frames_between_demos = int(min_duration_between_demos * fps)
    pre_buffer_frames = 15  # number of frames to pad the start of demos
    pre_buffer = deque(maxlen=pre_buffer_frames)  # rolling pre-buffer
    demo_buffer = []  # to accumulate frames in the current demo
    last_demo_end = -min_frames_between_demos - 1  # init far in the past
    inside_demo = False
    k, d = None, None

    # since it takes a long time to iterate over the dataloader, we will construct and preprocess
    # each interval as we iterate to reduce the overall runtime and cpu/gpu ram consumption
    start = time.perf_counter()
    with tqdm(total=num_rgb_frames) as pbar:
        for i, mps_batch in enumerate(mps_loader):
            if i >= num_rgb_frames:
                break

            mps_struct: MpsStruct = mps_batch[0]
            if k is None:
                k = mps_struct.k
            if d is None:
                d = mps_struct.d

            frame_data = {
                "rgb": mps_struct.rgb,
                "palm": (
                    mps_struct.right_palm
                    if args.is_right_hand
                    else mps_struct.left_palm
                ),
                "wrist": (
                    mps_struct.right_wrist
                    if args.is_right_hand
                    else mps_struct.left_wrist
                ),
                "c2w": mps_struct.c2w @ np.linalg.inv(T_opencv_to_aria),
            }

            # update rolling pre-buffer
            pre_buffer.append(frame_data)

            if has_hand(mps_struct):
                if not inside_demo:
                    # check demo spacing
                    if i - last_demo_end - 1 >= min_frames_between_demos:
                        demo_buffer = list(pre_buffer)  # start from rolling pre-buffer
                        inside_demo = True
                    else:
                        continue  # skip until enough spacing
                else:
                    demo_buffer.append(frame_data)
            else:
                if inside_demo and len(demo_buffer) >= min_demo_frames:
                    # end of a valid demo
                    start_idx = i - len(demo_buffer)
                    end_idx = i - 1
                    pbar.set_description(
                        f"Dispatching demo {len(job_refs)}: frames {start_idx} to {end_idx}"
                    )
                    job_refs.append(
                        process_single_demo.remote(
                            len(job_refs),
                            save_dir,
                            [f["rgb"] for f in demo_buffer],
                            k,
                            d,
                            [f["palm"] for f in demo_buffer],
                            [f["wrist"] for f in demo_buffer],
                            [f["c2w"] for f in demo_buffer],
                            is_right_hand=args.is_right_hand,
                            is_wilor=args.is_wilor,
                            prompts=args.prompts,
                        )
                    )
                    last_demo_end = i - 1
                inside_demo = False
                demo_buffer = []

            pbar.update(1)

        # handle tail-end demo if valid
        if inside_demo and len(demo_buffer) >= min_demo_frames:
            start_idx = num_rgb_frames - len(demo_buffer)
            print(
                f"Dispatching final demo {len(job_refs)}: frames {start_idx} to {num_rgb_frames-1}"
            )
            job_refs.append(
                process_single_demo.remote(
                    len(job_refs),
                    save_dir,
                    [f["rgb"] for f in demo_buffer],
                    k,
                    d,
                    [f["palm"] for f in demo_buffer],
                    [f["wrist"] for f in demo_buffer],
                    [f["c2w"] for f in demo_buffer],
                    is_right_hand=args.is_right_hand,
                    is_wilor=args.is_wilor,
                    prompts=args.prompts,
                )
            )

    # Track completion with tqdm
    while job_refs:
        done_refs, job_refs = ray.wait(job_refs, num_returns=1)
        ray.get(done_refs)

    end = time.perf_counter()
    print(f"all jobs completed in {end - start:.4f}s !!")
    ray.shutdown()
