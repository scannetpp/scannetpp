import torch
import os
import tempfile
import shutil
import cv2
import numpy as np
import logging
import random

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    filename="sam2_model.log",  # Log file path
    filemode="w",  # 'w' to overwrite the log file each time, 'a' to append
)


class SAM2VideoMaskModel:
    def __init__(self, sam2_checkpoint, model_cfg, num_points=5, device="cuda"):
        """
        Initialize SAM2 model and set device.
        """
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        self.predictor = self._build_predictor(sam2_checkpoint, model_cfg)
        self._initialize_storage()
        self.num_points = num_points

    def _build_predictor(self, sam2_checkpoint, model_cfg):
        """
        Helper function to build the SAM2 video predictor.
        """
        return build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=self.device
        )

    def _initialize_storage(self):
        """
        Initializes temporary directory and storage for RGB images, masks, and related data.
        """
        self.temp_dir = tempfile.mkdtemp()
        logging.info(f"Temporary directory created at: {self.temp_dir}")

        # Placeholder for resized images, masks, padded_masks, rgb_padded, padding information, and scores
        self.rgbs = []
        self.rgbs_padded = []
        self.masks = []
        self.masks_padded = []
        self.masks_refined = []
        self.padding_info = []
        self.scores = []

    def pad_and_store(self, rgbs, masks, scores):
        """
        Pads the RGB images and masks to the size of the largest image and stores them.
        """
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

    def set_state_and_refine_masks(self):
        """
        Set the state for the SAM2 predictor and refine masks.
        """
        inference_state = self._initialize_inference_state()
        points, labels, highest_score_idx = self._get_initial_prompts(points=True)

        self._refine_masks(inference_state, points, labels, highest_score_idx)
        self.predictor.reset_state(inference_state)

    def set_state_and_refine_masks_w_mask_prompt(self):
        """
        Set the state for the SAM2 predictor and refine masks using the mask with the highest score.
        """
        inference_state = self._initialize_inference_state()
        highest_score_mask, highest_score_idx = self._get_initial_prompts(mask=True)

        self._refine_masks_w_mask_prompt(
            inference_state, highest_score_mask, highest_score_idx
        )
        self.predictor.reset_state(inference_state)

    def set_state_and_refine_masks_w_manual_prompt(self, points, labels, frame_idx):
        """
        Set the state for the SAM2 predictor and refine masks using the provided points and labels.
        """
        inference_state = self._initialize_inference_state()

        # _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
        #     inference_state=inference_state,
        #     frame_idx=frame_idx,
        #     obj_id=1,
        #     mask=self.masks_padded[frame_idx],
        # )

        self._refine_masks(inference_state, points, labels, frame_idx)
        self.predictor.reset_state(inference_state)

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
            logging.info(
                f"Using mask with the highest score from frame {highest_score_idx} as the initial prompt."
            )

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

    def _refine_masks(self, inference_state, points, labels, highest_score_idx):
        """
        Refine the masks and propagate through frames.
        """
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=highest_score_idx,
            obj_id=1,
            points=points,
            labels=labels,
            box=None,
        )

        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(inference_state):
            refined_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

            if np.sum(refined_mask) == 0:
                logging.warning(
                    f"Refined mask is empty for frame {out_frame_idx}. Using the old mask."
                )
                refined_mask = self.masks_padded[out_frame_idx]

            self.masks_refined.append(refined_mask)

    def _refine_masks_w_mask_prompt(
        self, inference_state, highest_score_mask, highest_score_idx
    ):
        """
        Refine the masks using the mask with the highest score and propagate through frames.
        """
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=highest_score_idx,
            obj_id=1,
            mask=highest_score_mask,
        )

        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(inference_state):
            refined_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

            if np.sum(refined_mask) == 0:
                logging.warning(
                    f"Refined mask is empty for frame {out_frame_idx}. Using the old mask."
                )
                refined_mask = self.masks_padded[out_frame_idx]

            self.masks_refined.append(refined_mask)

    def unpad_masks_to_original_size(self):
        """
        Remove padding from refined masks to restore them to their original size.
        """
        self.masks_refined = [
            self._unpad_mask(mask, frame_idx)
            for frame_idx, mask in enumerate(self.masks_refined)
        ]

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

    def cleanup(self):
        """
        Clean up the temporary directory and clear stored data.
        """
        self._clear_temp_directory()
        self._clear_storage()

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
                    logging.error(f"Failed to delete {file_path}. Reason: {e}")

            logging.info(f"Contents of temporary directory {self.temp_dir} removed.")

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
        self.masks_refined = []
        logging.info(
            "Cleared all stored images, masks, scores, and padding information."
        )


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
        self.masks_refined = []
        self.crop_scores = []
        self.sam_scores = []

    def store_data(self, rgbs, masks, scores):
        """
        Stores the RGB images, masks, and scores.
        """
        if len(rgbs) != len(masks) or len(rgbs) != len(scores):
            raise ValueError("The number of RGB images, masks, and scores must match.")

        for idx, (rgb, mask, score) in enumerate(zip(rgbs, masks, scores)):
            self._store_data(idx, rgb, mask, score)

    def _store_data(self, idx, rgb, mask, score):
        """
        Store RGB, mask, and score data.
        """
        self.rgbs.append(rgb)
        self.masks.append(mask)
        self.crop_scores.append(score)

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

    def _set_image_for_predictor(self, rgb):
        """
        Preprocesses the RGB image and sets it for the SAM predictor.
        """
        # # Convert to RGB format (OpenCV loads images in BGR by default)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Set the image for the SAM predictor
        self.predictor.set_image(rgb)

    def _predict_mask_with_points(self, rgb, points):
        """
        Use sampled points to prompt SAM for mask prediction and return mask and score.
        """
        labels = np.ones(
            len(points), dtype=np.int32
        )  # All positive labels (foreground)

        # Set the current RGB image for the predictor
        self._set_image_for_predictor(rgb)

        # Perform mask prediction using the points and labels
        masks, scores, _ = self.predictor.predict(
            point_coords=points, point_labels=labels, multimask_output=False
        )

        # Convert mask to bool type if it's not already
        if masks.dtype != bool:
            masks = masks.astype(bool)

        return masks, scores

    def _predict_mask_with_points_and_bbox(self, rgb, points, bbox):
        """
        Use sampled points to prompt SAM for mask prediction and return mask and score.
        """
        labels = np.ones(
            len(points), dtype=np.int32
        )  # All positive labels (foreground)

        # Set the current RGB image for the predictor
        self._set_image_for_predictor(rgb)

        # Perform mask prediction using the points and labels
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox[None, :],
            multimask_output=False,
        )

        # Convert mask to bool type if it's not already
        if masks.dtype != bool:
            masks = masks.astype(bool)

        return masks, scores

    def ransac_mask_selection(self):
        """
        Perform RANSAC-like sampling of points and select the best mask based on SAM score.
        """
        for idx, (rgb, mask) in enumerate(zip(self.rgb, self.mask)):
            best_score = -float("inf")
            best_mask = None

            for _ in range(self.ransac_iterations):
                try:
                    # Sample points from the mask
                    sampled_points = self._sample_points_from_mask(mask)
                    bbox = self._get_bounding_box_from_mask(mask)

                    # Get mask prediction and score from SAM
                    # predicted_mask, predicted_scores = self._predict_mask_with_points(rgb, sampled_points)
                    predicted_mask, predicted_scores = (
                        self._predict_mask_with_points_and_bbox(
                            rgb, sampled_points, bbox
                        )
                    )

                    predicted_mask = predicted_mask[
                        0
                    ]  # Assuming single mask is returned

                    # Choose the mask with the highest score
                    score = predicted_scores[0]  # Assuming single mask is returned
                    # score = self._mask_score_calculation(idx, predicted_mask, predicted_scores[0])
                    if score > best_score:
                        best_score = score
                        best_mask = predicted_mask
                except ValueError as e:
                    logging.warning(f"Skipping frame {idx} due to error: {e}")
                    continue

            # Store the best mask and score for the current frame
            self.masks_refined.append(best_mask)
            self.sam_scores.append(best_score)

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

    def _mask_score_calculation(self, idx, refined_mask, sam_score):
        current_mask = self.mask[idx]

        current_area = np.sum(current_mask)
        refined_area = np.sum(refined_mask)
        areas_diff = abs(current_area - refined_area)

        areas_score = 1 / (1 + areas_diff)

        final_score = areas_score * sam_score
        return final_score

    def refine_masks(self):
        """
        Public method to trigger RANSAC-based mask refinement.
        """
        self.ransac_mask_selection()

    def cleanup(self):
        """
        Cleans up the stored data, including RGBs, masks, and scores,
        while keeping the SAM model ready for further use.
        """
        # Clear all stored data (RGBs, masks, scores)
        self.rgbs = []
        self.masks = []
        self.masks_refined = []
        self.crop_scores = []
        self.sam_scores = []

        logging.info("Cleared all stored images, masks, scores, and refined masks.")
