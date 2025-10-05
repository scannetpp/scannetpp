import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
from PIL import Image
import torch
from torchmetrics.image import PeakSignalNoiseRatio

from common.scene_release import ScannetppScene_Release
from eval.nvs import evaluate_all, get_test_images, scene_has_mask
from iphone.color_correction_utils import ColorCorrector

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
) -> Tuple[List[str], List[str]]:
    """
    Apply color correction to predicted images.

    Args:
        pred_dir: Path to the directory containing the predicted images.
        gt_dir: Path to the directory containing the GT images.
        output_dir: Path to the directory to save color-corrected images.
        image_list: List of image names to evaluate.
        mask_dir: Path to the directory containing the masks. Evaluate without mask if None.
        scene_id: Scene ID for logging.
        verbose: Print the evaluation results.
        gt_file_format: File format of the GT images.

    Process:
        1. Load pair of image/target
        2. Estimate optimal transport (OT)
        3. Apply the OT on image and save them

    Returns:
        The lists of path of original predicted images and color-corrected images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_pred_path_list = []
    image_cc_path_list = []
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)

    for image_idx, image_fn in enumerate(image_list):
        image_name = image_fn.split(".")[0]
        gt_image_path = os.path.join(gt_dir, image_name + gt_file_format)
        assert os.path.exists(gt_image_path), f"{scene_id} GT image not found: {image_fn} given path {gt_image_path}"
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
            assert len(mask.shape) == 2, f"mask should have 2 channels (H, W) but get shape: {mask.shape}"
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

        # Initialize color_corrector
        color_corrector = ColorCorrector("LinearTransport")
        color_corrector.fit(
            train_images_source=[pred_image],
            train_images_target=[gt_image],
            train_image_masks=[mask],
            paired=True,
            sample_every=1,
        )

        pred_image_cc = color_corrector.transform(
            image_source=pred_image,
            image_target=gt_image,
            image_mask=mask,
        )

        if verbose:
            l2_loss_before = np.linalg.norm(pred_image.reshape(-1, 3) - gt_image.reshape(-1, 3), axis=1).mean()
            l2_loss_after = np.linalg.norm(pred_image_cc.reshape(-1, 3) - gt_image.reshape(-1, 3), axis=1).mean()
            psnr_before = psnr_metric(torch.from_numpy(pred_image).permute(2, 0, 1), torch.from_numpy(gt_image).permute(2, 0, 1)).item()
            psnr_after = psnr_metric(torch.from_numpy(pred_image_cc).permute(2, 0, 1), torch.from_numpy(gt_image).permute(2, 0, 1)).item()
            print(
                f"[{scene_id}] Image: {image_name}\n"
                f"    L2 Loss Before CC: {l2_loss_before:.4f}\n"
                f"    L2 Loss After  CC: {l2_loss_after:.4f}\n"
                f"    PSNR Before CC: {psnr_before:.4f}\n"
                f"    PSNR After  CC: {psnr_after:.4f}\n"
            )
            # diff = np.linalg.norm(pred_image.reshape(-1, 3) - pred_image_cc.reshape(-1, 3), axis=1).mean()
            # print(f"Diff (before/after): {diff:.03}")

        save_array_to_image(pred_image_cc, output_dir / f"{image_name}.png")

        image_pred_path_list.append(str(pred_image_path))
        image_cc_path_list.append(str(output_dir / f"{image_name}.png"))
    return image_pred_path_list, image_cc_path_list


def color_correction_all(
    data_root: Union[str, Path],
    pred_dir: Union[str, Path],
    output_dir: Union[str, Path],
    scene_list: List[str],
    verbose: bool = True,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Apply color correction to all scenes in the scene list.

    Args:
        data_root: Root directory containing scene data
        pred_dir: Directory containing predicted images
        output_dir: Directory to save color-corrected images
        scene_list: List of scene IDs to process
        verbose: Whether to print progress information

    Returns:
        Dictionary of scene_id to (pred_path_list, cc_path_list)
    """

    output_dict = {}
    for i, scene_id in enumerate(scene_list):
        if verbose:
            print(f"({i+1} / {len(scene_list)}) scene_id: {scene_id}")

        assert (
            Path(pred_dir) / scene_id
        ).exists(), f"Prediction dir of scene {scene_id} does not exist"
        num_images_pred = len(os.listdir(Path(pred_dir) / scene_id))
        assert num_images_pred > 0, f"Prediction dir of scene {scene_id} is empty"

        # Here we assume the GT images are stored in the data_root / scene_id / "dslr_undistorted_by_iphone" and not "dslr"
        scene = ScannetppScene_Release(scene_id, data_root=data_root, dslr_folder_name="dslr_undistorted_by_iphone")

        # Get the list of test image names: "e.g., DSC09999.JPG"
        # but we need to refer the specific folder which contains DSLR after undistortion by iPhone intrinsic
        image_list = get_test_images(scene.dslr_nerfstudio_transform_path)  # read original transforms.json

        mask_dir = None
        if scene_has_mask(scene.dslr_nerfstudio_transform_path):
            mask_dir = scene.dslr_resized_mask_dir
            if verbose:
                print(f"Scene {scene_id} has masks. Using masks for color correction.")
        mask_dir = scene.dslr_resized_mask_dir

        pred_path_list, cc_path_list = color_correction(
            pred_dir=Path(pred_dir) / scene_id,
            gt_dir=scene.dslr_resized_dir,
            output_dir=Path(output_dir) / scene_id,
            image_list=image_list,
            mask_dir=mask_dir,
            scene_id=scene_id,
            verbose=verbose,
        )

        output_dict[scene_id] = (pred_path_list, cc_path_list)
    return output_dict


def main(args):
    if args.scene_id is not None:
        val_scenes = [args.scene_id]
    else:
        with open(args.split, "r") as f:
            val_scenes = f.readlines()
            val_scenes = [x.strip() for x in val_scenes if len(x.strip()) > 0]

    print("Before color correction:")
    evaluate_all(args.data_root, args.pred_dir, val_scenes, args.device, verbose=True)

    output_dict = color_correction_all(
        data_root=args.data_root,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        scene_list=val_scenes,
        verbose=True,
    )

    print("After color correction:")
    evaluate_all(args.data_root, args.output_dir, val_scenes, args.device, verbose=True)


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
