# TODO: Fix the problem when the image has mask (avoid using the masked regions for color correction)
# TODO: Make it a independent script for evaluation
# TODO: Remove the hardcoded paths and numbers

import argparse
from typing import List, Optional, Union
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ot

from common.scene_release import ScannetppScene_Release
from eval.nvs import get_test_images, scene_has_mask
from eval.nvs import evaluate_all
from iphone.color_correction_utils import get_concat_h, ColorCorrector, ALL_PIXELS

SUPPORT_IMAGE_FORMAT = [".JPG", ".jpg", ".png", ".PNG", ".jpeg"]


def save_array_to_image(
    array: np.ndarray,
    save_path: Union[str, Path],
):
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    image.save(save_path)


def color_correction(
    pred_dir: Union[str, Path],
    gt_dir: Union[str, Path],
    output_dir: Union[str, Path],
    image_list: List[str],
    mask_dir: Optional[Union[str, Path]] = None,
    scene_id: str = "unknown",
    verbose: bool = True,
    gt_file_format: str = ".JPG",
    cc_configs: dict = None,
):
    """
    Apply color correction to predicted images.

    Args:
        pred_dir: Path to the directory containing the predicted images.
        gt_dir: Path to the directory containing the GT images.
        image_list: List of image names to evaluate.
        upload_path: Path to upload directory.
        mask_dir: Path to the directory containing the masks. Evaluate without mask if None.
        scene_id: Scene ID for logging.
        verbose: Print the evaluation results.
        gt_file_format: File format of the GT images.
        cc_configs: Color correction configuration dictionary.

    Process:
        1. Load pair of image/target
        2. Estimate optimal transport operator
        3. Apply operator on image/Save it in path2images_after_cc
        4. Create side-by-side dir/Save it
        5. Add the path2images_after_cc

    Returns:
        dict: Dictionary containing paths to original, GT, and color-corrected images.
    """
    if cc_configs is None:
        cc_configs = {
            "method": "default",
            "option": "LinearTransport",
            "sample_size": 2764800,
            "batch_size": 1,
            "mode": "server"
        }

    # pred_after_cc_dir = Path(output_dir) / "after"
    # if not os.path.exists(pred_after_cc_dir):
    #     os.makedirs(pred_after_cc_dir)

    # pred_after_cc_dir = Path(pred_after_cc_dir) / scene_id
    # if not os.path.exists(pred_after_cc_dir):
    #     os.makedirs(pred_after_cc_dir)

    # # Path to pairs (before/after)
    # collection_dir = Path(upload_path) / "color_corrected_BA"
    # if not os.path.exists(collection_dir):
    #     os.makedirs(collection_dir)

    # collection_dir = Path(collection_dir) / scene_id
    # if not os.path.exists(collection_dir):
    #     os.makedirs(collection_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Copy color correction config
    # with open(os.path.join(collection_dir, "cc_config.txt"), "w") as config_f:
    #     config_f.write(str(cc_configs))

    color_corrector = None
    path_dict = {"pred": [], "gt": [], "cc_pred": []}

    for image_idx, image_fn in enumerate(image_list):
        image_name = image_fn.split(".")[0]
        gt_image_path = os.path.join(gt_dir, image_name + gt_file_format)
        assert os.path.exists(
            gt_image_path
        ), f"{scene_id} GT image not found: {image_fn} given path {gt_image_path}"
        gt_image = Image.open(gt_image_path)

        pred_image_path = None
        for img_format in SUPPORT_IMAGE_FORMAT:
            test_image_path = os.path.join(pred_dir, image_name + img_format)
            if os.path.exists(test_image_path):
                pred_image_path = test_image_path
                break
        assert (
            pred_image_path is not None
        ), f"{scene_id} pred image not found: {image_fn} with the following format {' '.join(SUPPORT_IMAGE_FORMAT)}"
        pred_image = Image.open(pred_image_path)

        if mask_dir is not None:
            mask_path = os.path.join(mask_dir, image_name + ".png")
            assert os.path.exists(mask_path), f"mask not found: {mask_path}"
            mask = Image.open(mask_path)
            mask = np.array(mask) > 0
            assert (
                len(mask.shape) == 2
            ), f"mask should have 2 channels (H, W) but get shape: {mask.shape}"
            assert (
                mask.shape[0] == gt_image.size[1] and mask.shape[1] == gt_image.size[0]
            ), f"mask shape {mask.shape} does not match GT image size: {gt_image.size}"
        else:
            mask = None

        if gt_image.size != pred_image.size:
            # Auto resize to match the GT image size
            pred_image = pred_image.resize(gt_image.size, Image.BICUBIC)

        gt_image = np.array(gt_image) / 255.0
        pred_image = np.array(pred_image) / 255.0
        assert len(gt_image.shape) == 3, f"GT image should have 3 channels (H, W, 3) but get shape: {gt_image.shape}"
        assert len(pred_image.shape) == 3, f"pred image should have 3 channels (H, W, 3) but get shape: {pred_image.shape}"

        image_shape = gt_image.shape
        # Initialize color_corrector
        num_samples = gt_image[0] * gt_image[1]
        color_corrector = ColorCorrector(
            method=cc_configs["method"],
            option=cc_configs["option"],
            batch_size=cc_configs["batch_size"],
            # sample_size=cc_configs["sample_size"],
            sample_size=num_samples,
            mode=cc_configs["mode"],
        )

        color_corrector.prepare_and_fit(
            train_images_source=[pred_image],
            train_images_target=[gt_image],
            train_image_masks=[mask],
            paired=True,
            sample_every=1,
        )

        pred_image_cc = color_corrector.transform_and_result(
            image_source=pred_image,
            image_target=gt_image,
            image_mask=mask,
        )

        if verbose:
            l2_loss_before = np.linalg.norm(pred_image.reshape(-1, 3) - gt_image.reshape(-1, 3), axis=1).mean()
            l2_loss_after = np.linalg.norm(pred_image_cc.reshape(-1, 3) - gt_image.reshape(-1, 3), axis=1).mean()
            print(f"Before CC L2 loss: {l2_loss_before:.03}, After CC L2 loss: {l2_loss_after:.03}")
            # diff = np.linalg.norm(pred_image.reshape(-1, 3) - pred_image_cc.reshape(-1, 3), axis=1).mean()
            # print(f"Diff (before/after): {diff:.03}")

        save_array_to_image(pred_image_cc, output_dir / f"{image_name}.png")


def color_correction_all(data_root, pred_dir, output_dir, scene_list, verbose=True):
    """
    Apply color correction to all scenes in the scene list.

    Args:
        data_root: Root directory containing scene data
        pred_dir: Directory containing predicted images
        scene_list: List of scene IDs to process
        upload_path: Path to upload directory
        verbose: Whether to print progress information

    Returns:
        dict: Dictionary mapping scene IDs to path dictionaries
    """
    for i, scene_id in enumerate(scene_list):
        if verbose:
            print(f"({i+1} / {len(scene_list)}) scene_id: {scene_id}")

        assert (
            Path(pred_dir) / scene_id
        ).exists(), f"Prediction dir of scene {scene_id} does not exist"
        num_images_pred = len(os.listdir(Path(pred_dir) / scene_id))
        assert num_images_pred > 0, f"Prediction dir of scene {scene_id} is empty"

        scene = ScannetppScene_Release(scene_id, data_root=data_root)

        # Get the list of test image names: "e.g., DSC09999.JPG"
        # but we need to refer the specific folder which contains DSLR after undistortion by iPhone intrinsic
        image_list = get_test_images(scene.dslr_nerfstudio_transform_path)  # read original transforms.json

        mask_dir = None
        if scene_has_mask(scene.dslr_nerfstudio_transform_path):
            mask_dir = scene.dslr_resized_mask_dir
            if verbose:
                print(f"Scene {scene_id} has masks. Using masks for color correction.")
        mask_dir = scene.dslr_resized_mask_dir  # change to the DSLR_undistorted_iphone mask dir

        color_correction(
            pred_dir=Path(pred_dir) / scene_id,
            gt_dir=scene.dslr_resized_dir,  # change to the DSLR_undistorted_iphone dir
            output_dir=Path(output_dir) / scene_id,
            image_list=image_list,
            mask_dir=mask_dir,
            scene_id=scene_id,
            verbose=verbose,
        )


def main(args):
    if args.scene_id is not None:
        val_scenes = [args.scene_id]
    else:
        with open(args.split, "r") as f:
            val_scenes = f.readlines()
            val_scenes = [x.strip() for x in val_scenes if len(x.strip()) > 0]

    all_images, all_psnr, all_ssim, all_lpips = evaluate_all(
        args.data_root, args.pred_dir, val_scenes, args.device
    )

    color_correction_all(
        data_root=args.data_root,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        scene_list=val_scenes,
        verbose=False,
    )

    print("After color correction:")
    all_images, all_psnr, all_ssim, all_lpips = evaluate_all(
        args.data_root, args.output_dir, val_scenes, args.device
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root", help="Data root (e.g., scannetpp/data)", required=True
    )
    p.add_argument(
        "--split",
        help="The split file containing the scenes for evaluation (e.g., scannetpp/splits/nvs_sem_val.txt)",
        default=None,
    )
    p.add_argument(
        "--scene_id",
        help="Scene ID for evaluation (e.g., 3db0a1c8f3)",
        default=None,
    )
    p.add_argument("--pred_dir", help="Model prediction", required=True)
    p.add_argument("--output_dir", help="Save output", required=True)
    p.add_argument("--device", help="Device", default="cuda")
    args = p.parse_args()
    main(args)
