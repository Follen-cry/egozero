from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def safe_crop(image, center, crop_h, crop_w):
    """
    Crops a crop_h x crop_w region centered at `center`, adjusting if too close to image edges.
    """
    H, W, _ = image.shape
    cx, cy = center

    # Compute bounds
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Adjust if crop goes out of bounds
    if x2 > W:
        x2 = W
        x1 = W - crop_w
    if y2 > H:
        y2 = H
        y1 = H - crop_h

    # Final clip in case image is smaller than crop size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    return image[y1:y2, x1:x2]


class GroundedSAM2:
    def __init__(self, device="cuda"):
        self.device = device

        # uncomment for sam-2 tracking
        # CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
        # SAM2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "sam-2", "sam2"))
        # sam2_checkpoint = os.path.join(CKPT_DIR, "sam2.1_hiera_large.pt")
        # model_cfg = os.path.join(SAM2_DIR, "configs/sam2.1/sam2.1_hiera_l.yaml")
        # self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        # self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        grounded_dino = "IDEA-Research/grounding-dino-base"
        self.gdino_processor = AutoProcessor.from_pretrained(grounded_dino)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounded_dino
        ).to(device)

    def __call__(self, image: Union[str, Image.Image, np.ndarray], prompt: str):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # self.sam2_predictor.set_image(np.array(image.convert("RGB")))

        inputs = self.gdino_processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.gdino_model(**inputs)

        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )
        results = results[0]
        if len(results["scores"]) == 0:
            return None
        idx = torch.argmax(results["scores"])
        bbox = results["boxes"][idx].cpu().numpy().astype(int)
        return bbox


if __name__ == "__main__":
    grounded_sam2 = GroundedSAM2(device="cuda")
    image_path = "/home/ademi_adeniji/egozero/mps/mps_right-hand-3pos-v1_vrs/preprocess/demonstration_00002/first_frame.png"
    prompt = "a brown box."

    bbox = grounded_sam2(image_path, prompt)
    bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    image = np.array(Image.open(image_path).convert("RGB"))
    image = safe_crop(image, bbox_center, 512, 512)
    Image.fromarray(image).save("test2.png")
