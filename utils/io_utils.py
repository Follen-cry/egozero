import os
import sys
from functools import wraps
from typing import List, Union

import cv2
import imageio.v3 as iio
import numpy as np
from skimage.transform import resize


class suppress:
    def __init__(self, stdout=True, stderr=True):
        self.suppress_stdout = stdout
        self.suppress_stderr = stderr
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        """Enter the context and suppress stdout/stderr."""
        if self.suppress_stdout:
            self.original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        if self.suppress_stderr:
            self.original_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context and restore stdout/stderr."""
        if self.suppress_stdout:
            sys.stdout.close()
            sys.stdout = self.original_stdout
        if self.suppress_stderr:
            sys.stderr.close()
            sys.stderr = self.original_stderr

    def __call__(self, func):
        """Allow this class to be used as a decorator."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:  # Use the context manager behavior
                return func(*args, **kwargs)

        return wrapper


def save_video(
    save_path: str, frames: Union[np.ndarray, List], max_size: int = None, fps: int = 30
):
    if not isinstance(frames, np.ndarray):
        frames = np.stack(frames)
    n, h, w = frames.shape[:-1]
    if max_size is not None:
        scale = max_size / max(w, h)
        h = int(h * scale // 16) * 16
        w = int(w * scale // 16) * 16
        frames = (resize(frames, output_shape=(n, h, w)) * 255).astype(np.uint8)
    print(f"saving video of size {(n, h, w)} to {save_path}")
    iio.imwrite(save_path, frames, fps=fps, codec="libx264")


def concatenate_frames(*frames: List[np.ndarray]):
    # frames: list of arbitrarily sized rgb images, each of some shape (h, w, 3)
    target_height = max(frame.shape[0] for frame in frames)
    resized_frames = [
        cv2.resize(
            frame, (int(frame.shape[1] * target_height / frame.shape[0]), target_height)
        )
        for frame in frames
    ]
    return np.concatenate(resized_frames, axis=1)


def jsonify(obj):
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [jsonify(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(jsonify(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
