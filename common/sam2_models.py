import torch
import os
import tempfile
import shutil
import cv2
import numpy as np
import sys

# import logging
import random

try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("SAM2 not installed. Please install SAM2 to refine masks")
    sys.exit(1)


class SAM2VideoMaskModel:
    def __init__(
        self, sam2_checkpoint, model_cfg, num_points=5, device="cuda", temp_dir=None
    ):
        """
        Initialize SAM2 model and set device.
        """
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        self.predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=self.device
        )

        # Placeholder for rgbs, masks, masks_padded, rgbs_padded, padding information, and scores
        self.rgbs = []
        self.rgbs_padded = []
        self.masks = []
        self.masks_padded = []
        self.padding_info = []
        self.scores = []

        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        else:
            self.temp_dir = temp_dir
        # logging.info(f"Temporary directory created at: {self.temp_dir}")

        self.num_points = num_points
        self.predictor_inference_state = None
        self.refined_flag = False

    def store_data(self, rgbs, masks, scores):
        """
        Pads the RGB images and masks to the size of the largest image and stores them.
        """
        if self.predictor_inference_state is not None:
            raise ValueError(
                "Inference state already initialized. Please cleanup before storing new data."
            )

        if len(rgbs) != len(masks) or len(rgbs) != len(scores):
            raise ValueError("The number of RGB images, masks, and scores must match.")

        max_h, max_w = self._get_max_dimensions(rgbs)

        for idx, (rgb, mask, score) in enumerate(zip(rgbs, masks, scores)):
            padded_rgb, padded_mask, padding_info = self._pad_image_and_mask(
                rgb, mask, max_h, max_w
            )
            self._store_padded_data(
                idx, rgb, padded_rgb, mask, padded_mask, score, padding_info
            )

        self.predictor_inference_state = self._initialize_inference_state()

    def refine_masks_and_propagate(self, mode=None):
        """
        Refine masks using SAM2 predictor.
        """
        if self.predictor_inference_state is None:
            raise ValueError(
                "Inference state not initialized. Please store data before refining masks."
            )

        if mode == "points":
            points, labels, highest_score_idx = self._get_initial_prompts(points=True)
            _ = self.refine_mask_w_points_prompt(points, labels, highest_score_idx)
            refined_masks = self.propagate_prompt()
            self.masks = refined_masks
            return refined_masks

        elif mode == "mask":
            highest_score_mask, highest_score_idx = self._get_initial_prompts(mask=True)
            _ = self.refine_mask_w_mask_prompt(highest_score_mask, highest_score_idx)
            refined_masks = self.propagate_prompt()
            self.masks = refined_masks
            return refined_masks

        else:
            raise ValueError("Mode must be either 'points' or 'mask'.")

    def reset_inference_state(self):
        """
        Reset the inference state to the initial state.
        """
        self.refined_flag = False
        if self.predictor_inference_state is None:
            raise ValueError("Inference state is not set.")

        self.predictor.reset_state(self.predictor_inference_state)
        self.predictor_inference_state = self._initialize_inference_state()

    def cleanup(self):
        """
        Clean up the temporary directory and clear stored data.
        """
        self.refined_flag = False
        if self.predictor_inference_state is not None:
            self.predictor.reset_state(self.predictor_inference_state)
            self.predictor_inference_state = None
        self._clear_temp_directory()
        self._clear_storage()

    def refine_mask_w_points_prompt(self, points, labels, frame_idx):
        """
        Refine the masks with point prompts.
        """
        if frame_idx >= len(self.rgbs):
            raise ValueError("Invalid frame index provided.")

        if self.predictor_inference_state is None:
            raise ValueError(
                "Inference state not initialized. Please store data before refining masks."
            )

        self.refined_flag = True

        _, _, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.predictor_inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
            box=None,
        )

        refined_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
        unpadded_refined_mask = self._unpad_mask(refined_mask, frame_idx)

        return unpadded_refined_mask

    def refine_mask_w_mask_prompt(self, highest_score_mask, frame_idx):
        """
        Refine the masks using the mask with the highest score.
        """
        if frame_idx >= len(self.rgbs):
            raise ValueError("Invalid frame index provided.")

        if self.predictor_inference_state is None:
            raise ValueError(
                "Inference state not initialized. Please store data before refining masks."
            )

        self.refined_flag = True

        _, _, out_mask_logits = self.predictor.add_new_mask(
            inference_state=self.predictor_inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            mask=highest_score_mask,
        )

        refined_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
        unpadded_refined_mask = self._unpad_mask(refined_mask, frame_idx)

        return unpadded_refined_mask

    def propagate_prompt(self):
        """
        Propagate the prompt through the video.
        """
        refined_masks = []

        if self.predictor_inference_state is None:
            raise ValueError(
                "Inference state not initialized. Please store data before refining masks."
            )

        if not self.refined_flag:
            raise ValueError(
                "Refinement not performed. Please refine masks before propagating."
            )

        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(self.predictor_inference_state):
            refined_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            unpadded_refined_mask = self._unpad_mask(refined_mask, out_frame_idx)

            if np.sum(refined_mask) == 0:
                # logging.warning(
                #     f"Refined mask is empty for frame {out_frame_idx}. Using the old mask."
                # )
                refined_mask = self.masks_padded[out_frame_idx]

            refined_masks.append(unpadded_refined_mask)

        if len(refined_masks) < len(self.masks):
            raise ValueError(
                "The number of refined masks is less than the number of input masks."
            )

        return refined_masks

    def _get_max_dimensions(self, rgbs):
        """
        Get the maximum height and width among the provided RGB images.
        """
        max_h = max([rgb.shape[0] for rgb in rgbs])
        max_w = max([rgb.shape[1] for rgb in rgbs])
        return max_h, max_w

    def _pad_image_and_mask(self, rgb, mask, max_h, max_w):
        """
        Pads the RGB and mask to the provided max dimensions.
        """
        h, w, _ = rgb.shape
        pad_h, pad_w = max_h - h, max_w - w
        padding_info = ((0, pad_h), (0, pad_w))

        padded_rgb = np.pad(
            rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0
        )
        padded_mask = np.pad(
            mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
        )

        return padded_rgb, padded_mask, padding_info

    def _store_padded_data(
        self, idx, rgb, padded_rgb, mask, padded_mask, score, padding_info
    ):
        """
        Store padded data, and save the padded RGB image to the temp directory.
        """
        rgb_filename = os.path.join(self.temp_dir, f"{idx}.jpg")
        cv2.imwrite(rgb_filename, padded_rgb)

        self.rgbs.append(rgb)
        self.rgbs_padded.append(padded_rgb)
        self.masks.append(mask)
        self.masks_padded.append(padded_mask)
        self.scores.append(score)
        self.padding_info.append(padding_info)

    def _initialize_inference_state(self):
        """
        Initialize the predictor's inference state.
        """
        return self.predictor.init_state(video_path=self.temp_dir)

    def _get_initial_prompts(self, points=False, mask=False):
        """
        Determine initial points based on the mask with the highest score.
        """
        highest_score_idx = np.argmax(self.scores)
        highest_score_mask = self.masks_padded[highest_score_idx]

        if highest_score_idx != 0:
            print(
                "Using mask with the highest score from frame {highest_score_idx} as the initial prompt."
            )
            # logging.info(
            #     f"Using mask with the highest score from frame {highest_score_idx} as the initial prompt."
            # )

        if points:
            mask_indices = np.argwhere(highest_score_mask > 0)
            points = np.array(
                [
                    mask_indices[np.random.choice(len(mask_indices))]
                    for _ in range(self.num_points)
                ],
                dtype=np.float32,
            )
            labels = np.ones(self.num_points, dtype=np.int32)

            return points, labels, highest_score_idx
        elif mask:
            return highest_score_mask, highest_score_idx
        else:
            raise ValueError("Either points or mask must be True.")

    def _unpad_mask(self, mask, frame_idx):
        """
        Unpad a single mask based on its padding information.
        """
        pad_h, pad_w = self.padding_info[frame_idx]

        # Unpad the mask based on padding info
        unpadded_mask = mask
        if pad_h[1] > 0:  # Check if there was padding at the bottom
            unpadded_mask = unpadded_mask[: -pad_h[1], :]
        if pad_w[1] > 0:  # Check if there was padding at the right
            unpadded_mask = unpadded_mask[:, : -pad_w[1]]

        return unpadded_mask

    def _clear_temp_directory(self):
        """
        Clears the contents of the temporary directory but leaves the directory itself.
        """
        if os.path.exists(self.temp_dir):
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file or symbolic link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove subdirectory and its contents
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
                    # logging.error(f"Failed to delete {file_path}. Reason: {e}")

            # logging.info(f"Contents of temporary directory {self.temp_dir} removed.")

    def _clear_storage(self):
        """
        Clears all stored data except for the SAM2 model.
        """
        self.rgbs = []
        self.rgbs_padded = []
        self.masks = []
        self.masks_padded = []
        self.scores = []
        self.padding_info = []
        # logging.info(
        #     "Cleared all stored images, masks, scores, and padding information."
        # )


class SAM2ImageMaskModel:
    def __init__(
        self,
        sam2_checkpoint,
        model_cfg,
        device="cuda",
        num_points=5,
        ransac_iterations=10,
    ):
        """
        Initialize SAM2 model, set device, and configure number of points and RANSAC iterations.
        """
        self.device = torch.device(device)
        self.num_points = num_points  # Number of points to sample from the mask
        self.ransac_iterations = ransac_iterations  # Number of RANSAC iterations

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.predictor = self._build_predictor(sam2_checkpoint, model_cfg)
        self._initialize_storage()

    def store_data(self, rgbs, masks, scores):
        """
        Stores the RGB images, masks, and scores.
        """
        if self.rgbs or self.masks or self.scores:
            raise ValueError(
                "Stored data already exists. Please cleanup before storing new data."
            )

        if len(rgbs) != len(masks) or len(rgbs) != len(scores):
            raise ValueError("The number of RGB images, masks, and scores must match.")

        for idx, (rgb, mask, score) in enumerate(zip(rgbs, masks, scores)):
            self._store_data(idx, rgb, mask, score)

    def refine_masks(self):
        """
        Public method to trigger RANSAC-based mask refinement.
        """
        if not self.rgbs or not self.masks or not self.scores:
            raise ValueError("No stored data to refine masks for.")

        masks_refined, sam_scores = self._ransac_mask_selection()
        self.masks = masks_refined
        self.scores = sam_scores
        return masks_refined, sam_scores

    def refine_mask_w_points_prompt(self, points, labels, frame_idx):
        """
        Refine the mask for the provided frame index.
        """
        if not self.rgbs or not self.masks or not self.scores:
            raise ValueError("No stored data to refine masks for.")
        if frame_idx >= len(self.rgbs):
            raise ValueError("Invalid frame index provided.")

        rgb = self.rgbs[frame_idx]
        bbox = None

        predicted_mask, predicted_scores = self._predict_mask(rgb, points, bbox, labels)

        self.masks[frame_idx] = predicted_mask[0]
        self.scores[frame_idx] = predicted_scores[0]

        return predicted_mask[0], predicted_scores[0]

    def cleanup(self):
        """
        Cleans up the stored data, including RGBs, masks, and scores,
        while keeping the SAM model ready for further use.
        """
        # Clear all stored data (RGBs, masks, scores)
        self.rgbs = []
        self.masks = []
        self.scores = []

        # logging.info("Cleared all stored images, masks, scores, and refined masks.")

    def _ransac_mask_selection(self):
        """
        Perform RANSAC-like sampling of points and select the best mask based on SAM score.
        """
        masks_refined = []
        sam_scores = []

        for idx, (rgb, mask) in enumerate(zip(self.rgbs, self.masks)):
            best_score = -float("inf")
            best_mask = None

            for _ in range(self.ransac_iterations):
                try:
                    # Sample points from the mask
                    sampled_points = self._sample_points_from_mask(mask)
                    labels = np.ones(len(sampled_points), dtype=np.int32)
                    bbox = self._get_bounding_box_from_mask(mask)

                    # Get mask prediction and score from SAM, assume single mask and score
                    predicted_mask, predicted_scores = self._predict_mask(
                        rgb, sampled_points, bbox, labels
                    )

                    predicted_mask = predicted_mask[0]

                    score = predicted_scores[0]  # Assuming single mask is returned
                    if score > best_score:
                        best_score = score
                        best_mask = predicted_mask

                except ValueError as e:
                    # logging.warning(f"Skipping frame {idx} due to error: {e}")
                    continue

            # Store the best mask and score for the current frame
            masks_refined.append(best_mask)
            sam_scores.append(best_score)

        return masks_refined, sam_scores

    def _build_predictor(self, sam2_checkpoint, model_cfg):
        """
        Helper function to build the SAM2 image predictor.
        """
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        return SAM2ImagePredictor(sam2)

    def _initialize_storage(self):
        """
        Initializes storage for RGB images, masks, and related data.
        """
        self.rgbs = []
        self.masks = []
        self.scores = []

    def _store_data(self, idx, rgb, mask, score):
        """
        Store RGB, mask, and score data.
        """
        self.rgbs.append(rgb)
        self.masks.append(mask)
        self.scores.append(score)

    def _sample_points_from_mask(self, mask):
        """
        Sample n points from the provided mask where the mask is non-zero.
        """
        mask_indices = np.argwhere(mask > 0)  # Get non-zero mask points
        if len(mask_indices) == 0:
            raise ValueError("No valid mask points to sample from.")

        # Randomly sample 'n' points
        sampled_points = np.array(
            random.choices(mask_indices, k=self.num_points), dtype=np.float32
        )
        return sampled_points

    def _predict_mask(self, rgb, points, bbox, labels):
        """
        Use sampled points to prompt SAM for mask prediction and return mask and score.
        """

        self.predictor.set_image(rgb)

        if bbox is not None:
            bbox = bbox[None, :]

        # Perform mask prediction using the points and labels
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox,
            multimask_output=False,
        )

        if masks.dtype != bool:
            masks = masks.astype(bool)

        return masks, scores

    def _get_bounding_box_from_mask(self, mask):
        """
        Given a binary mask, return the bounding box in xyxy format.
        """

        # Find the indices where the mask is True (non-zero)
        rows, cols = np.where(mask)

        # If the mask is empty (no True values), return an empty bounding box
        if len(rows) == 0 or len(cols) == 0:
            return [0, 0, 0, 0]

        # Get the bounding box coordinates
        x_min = np.min(cols)
        y_min = np.min(rows)
        x_max = np.max(cols)
        y_max = np.max(rows)

        return np.array([x_min, y_min, x_max, y_max])
