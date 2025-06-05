import gc
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class Correspondence:

    def __init__(
        self,
        device,
        width=-1,
        height=-1,
        image_size_multiplier=0.5,
        ensemble_size=8,
        dift_layer=1,
        dift_steps=50,
        use_segmentation=True,
    ):
        """
        Initialize the Correspondence class.

        Parameters:
        -----------
        device : str
            The device to use for computation, either 'cpu' or 'cuda' (for GPU). If you need to put the dift model on a different device to
            save space you can set this to cuda:1

        width : int
            The width that should be used in the correspondence model.

        height : int
            The height that should be used in the correspondence model.

        image_size_multiplier : int
            The multiplier to use for the image size in the DIFT model if height and weight are -1.

        ensemble_size : int
            The size of the ensemble for the DIFT model.

        dift_layer : int
            The specific layer of the DIFT model to use for feature extraction.

        dift_steps : int
            The number of steps/iterations for the DIFT model to use in feature extraction.

        use_segmentation: bool
            Whether to use grounded-SAM to restrict DIFT predictions with segmentation mask.
        """
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from dift.src.models.dift_sd import SDFeaturizer

        self.dift = SDFeaturizer(device=device)
        if torch.cuda.get_device_capability() == (12, 0):
            self.dift.pipe.disable_xformers_memory_efficient_attention()
        self.dift.pipe.enable_attention_slicing(
            slice_size="max"
        )  # reduce ram requirements

        if use_segmentation:
            grounded_dino = "IDEA-Research/grounding-dino-base"
            self.grounded_dino_processor = AutoProcessor.from_pretrained(grounded_dino)
            self.grounded_dino_model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(grounded_dino).to(
                    device
                )
            )

        self.device = device
        self.width = width
        self.height = height
        self.image_size_multiplier = image_size_multiplier
        self.ensemble_size = ensemble_size
        self.dift_layer = dift_layer
        self.dift_steps = dift_steps
        self.use_segmentation = use_segmentation

        # to be populated later in `set_expert_correspondence`
        self.prompts = None
        self.expert_box = None
        self.expert_features = None

    @torch.no_grad()
    def _forward_grounded_dino(
        self,
        image: Image.Image,
        prompts: List[str],
        crop_size: int = 512,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ):
        crop = False
        if crop:
            orig_width, orig_height = image.size
            left = (orig_width - crop_size) // 2
            upper = (orig_height - crop_size) // 2
            right = left + crop_size
            lower = upper + crop_size
            cropped_image = image.crop((left, upper, right, lower))
        else:
            orig_width, orig_height = image.size
            left = 0
            upper = 0
            right = orig_width
            lower = upper + orig_height
            cropped_image = image

        boxes = []
        for prompt in prompts:
            inputs = self.grounded_dino_processor(
                images=cropped_image, text=prompt, return_tensors="pt"
            ).to(self.device)
            outputs = self.grounded_dino_model(**inputs)
            results = (
                self.grounded_dino_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[cropped_image.size[::-1]],
                )
            )
            results = results[0]
            if len(results["scores"]) == 0:
                continue

            idx = torch.argmax(results["scores"])
            box = results["boxes"][idx].cpu().numpy().astype(int)
            box[0] += left  # x1
            box[1] += upper  # y1
            box[2] += left  # x2
            box[3] += upper  # y2
            boxes.append(box)

        if len(boxes) == 0:
            return cropped_image, (left, upper, right, lower)

        # find the union of the box corners
        boxes = np.stack(boxes, axis=-1)  # shape (4, n) where n is the number of boxes
        box = [
            np.min(boxes[0]).item(),
            np.min(boxes[1]).item(),
            np.max(boxes[2]).item(),
            np.max(boxes[3]).item(),
        ]

        cropped_image = image.crop((box[0], box[1], box[2], box[3]))
        return cropped_image, box

    @torch.no_grad()
    def _forward_dift(self, image: Image.Image, prompt: str):
        image = image.resize((self.width, self.height), resample=Image.BILINEAR)
        image = (TF.pil_to_tensor(image) / 255.0 - 0.5) * 2
        image = image.to(self.device)
        features = self.dift.forward(
            image,
            prompt=prompt,
            ensemble_size=self.ensemble_size,
            up_ft_index=self.dift_layer,
            t=self.dift_steps,
        )
        features = features.to(self.device)
        return features

    # Get the feature map from the DIFT model for the expert image to compare with the first frame of each episode later on
    def set_expert_correspondence(self, expert_image, prompts):
        self.prompts = prompts
        if self.width == -1 or self.height == -1:
            self.width = int(expert_image.size[0] * self.image_size_multiplier)
            self.height = int(expert_image.size[1] * self.image_size_multiplier)

        # extract segmented image + bbox with grounded-sam
        if self.use_segmentation:
            expert_image, self.expert_box = self._forward_grounded_dino(
                expert_image, prompts, box_threshold=0.4, text_threshold=0.3
            )
        else:
            self.expert_box = (0, 0, expert_image.size[0], expert_image.size[1])

        # crop -> transform -> dift (this order is important for `find_correspondence`!)
        # note that all transforms are on the crop of the image, so all transformed images share the same bbox offset
        self.expert_features = self._forward_dift(expert_image, " and ".join(prompts))
        return expert_image

    def find_correspondence(self, current_image: Image.Image, coords: List):
        """
        Find the corresponding points between the expert image and the current image

        Parameters:
        -----------
        current_image : Image.Image
            The current image to compare with the expert image.

        coords : list
            The coordinates of the points to find correspondence between the expert image and the current image.
        """
        with torch.no_grad(), torch.amp.autocast(self.device, dtype=torch.float16):
            if self.use_segmentation:
                current_image, current_box = self._forward_grounded_dino(
                    current_image, self.prompts
                )
            else:
                current_box = (0, 0, current_image.size[0], current_image.size[1])
            current_features = self._forward_dift(
                current_image, " and ".join(self.prompts)
            )

            out_coords = torch.zeros(coords.shape)
            num_channel = self.expert_features.shape[1]
            src_ft = F.interpolate(
                self.expert_features,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=True,
            )

            for i, coord in enumerate(coords):
                # crop -> transform -> cossim points (same order as crop -> transform -> dift for image!)
                # all global coords share the same bbox offset, so subtract this offset and then transform the points
                x = int(
                    (coord[1] - self.expert_box[0])
                    * self.width
                    / (self.expert_box[2] - self.expert_box[0])
                )
                y = int(
                    (coord[2] - self.expert_box[1])
                    * self.height
                    / (self.expert_box[3] - self.expert_box[1])
                )

                src_vec = src_ft[0, :, y, x].view(1, num_channel).clone()
                trg_ft = F.interpolate(
                    current_features,
                    size=(self.height, self.width),
                    mode="bilinear",
                    align_corners=True,
                )
                trg_vec = trg_ft.view(1, num_channel, -1)  # N, C, HW

                src_vec = torch.nn.functional.normalize(src_vec)  # 1, C
                trg_vec = torch.nn.functional.normalize(trg_vec)  # N, C, HW
                cos_map = (
                    torch.matmul(src_vec, trg_vec)
                    .view(1, self.height, self.width)
                    .cpu()
                    .numpy()
                )
                cos_val = cos_map[0].max()

                max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
                out_coords[i, 1] = (
                    int(max_yx[1] * current_image.size[0] / self.width) + current_box[0]
                )
                out_coords[i, 2] = (
                    int(max_yx[0] * current_image.size[1] / self.height)
                    + current_box[1]
                )

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            return out_coords.cpu().numpy(), current_image
