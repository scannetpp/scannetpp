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


def color_correction(
    pred_dir: Union[str, Path],
    gt_dir: Union[str, Path],
    output_dir: Union[str, Path],
    image_list: List[str],
    mask_dir: Optional[Union[str, Path]] = None,
    scene_id: str = "unknown",
    verbose: bool = True,
    gt_file_format: str = ".JPG",
    device: str = "cpu",
    cc_configs: dict = None,
):

    print(pred_dir)
    print(gt_dir)
    print(output_dir)
    print(image_list)
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
        device: Device to use for computation.
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
            mask = torch.from_numpy(np.array(mask)).to(device)
            mask = (mask > 0).bool()
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
        # gt_image = gt_image.to(device)
        pred_image = np.array(pred_image) / 255.0
        # pred_image = pred_image.to(device)
        image_shape = gt_image.shape

        assert len(gt_image.shape) == 3, f"GT image should have 3 channels (H, W, 3) but get shape: {gt_image.shape}"
        assert len(pred_image.shape) == 3, f"pred image should have 3 channels (H, W, 3) but get shape: {pred_image.shape}"
        gt_image = gt_image.transpose(2, 0, 1)
        pred_image = pred_image.transpose(2, 0, 1)

        # If the mask is given and not all pixels are valid
        if mask is not None and not torch.all(mask):
            valid_gt = gt_image[:, mask]
            valid_pred = pred_image[:, mask]
        else:
            valid_gt = gt_image.reshape(3, -1)
            valid_pred = pred_image.reshape(3, -1)
        num_samples = valid_gt.shape[1]

        # print("==" * 10 + f"{_id}, {image_name}, {scene_id}" + "==" * 10)
        # print("gt: ", valid_gt.shape, gt_image_path)
        # print("pred: ", valid_pred.shape, pred_image_path)
        # if mask is not None and not torch.all(mask):
        #     print("mask: ", mask.shape, mask_path)
        # else:
        #     print("mask: not available")

        # Initialize color_corrector
        color_corrector = ColorCorrector(
            image_shape=image_shape,
            method=cc_configs["method"],
            option=cc_configs["option"],
            batch_size=cc_configs["batch_size"],
            # sample_size=cc_configs["sample_size"],
            sample_size=num_samples,
            mode=cc_configs["mode"],
        )

        # Prepare for color_correction
        (
            train_Xs,
            train_Xt,
            eval_Xs,
            eval_Xt,
            eval_masks,
            eval_path2Is,
            eval_path2It,
            _,
        ) = color_corrector.preparation(
            train_images_path_list=[pred_image_path],
            train_gt_path_list=[gt_image_path],
            train_masks_path_list=[mask_path],
            eval_images_path_list=[pred_image_path],
            eval_gt_path_list=[gt_image_path],
            eval_masks_path_list=[mask_path],
            sample_every=ALL_PIXELS // cc_configs["sample_size"],
            paired=True,
            offset=0,
            mode=cc_configs["mode"],
            train_batch_size=cc_configs["batch_size"],
            eval_batch_size=cc_configs["batch_size"],
            downsample=-1
        )

        # Color correction operator estimation
        color_corrector(Xs=train_Xs, Xt=train_Xt)

        # Apply color_correction
        transformed_Xs = color_corrector.transform_and_result(
            eval_Xs=eval_Xs,
            eval_Xt=eval_Xt,
            eval_masks=eval_masks,
            # eval_batch_size=cc_configs["batch_size"],
        )

        transform_x = transformed_Xs[0]
        # post process
        eval_source_image_path = eval_path2Is[0]
        path_dict["pred"].append(str(eval_source_image_path))
        eval_target_image_path = eval_path2It[0]
        path_dict["gt"].append(str(eval_target_image_path))

        print("eval source (test DSLR): ", eval_source_image_path)
        print("eval target (test DSLR): ", eval_target_image_path)

        im1 = Image.open(eval_source_image_path).convert("RGB")
        im2 = Image.open(eval_target_image_path).convert("RGB")
        im3 = Image.fromarray(transform_x).convert("RGB")

        npim1 = np.asarray(im1).astype(np.uint8)
        npim2 = np.asarray(im2).astype(np.uint8)
        npim3 = np.asarray(im3).astype(np.uint8)
        print("transformed x: ", npim3.shape)

        if npim1.shape != npim2.shape:
            im1 = Image.fromarray(np.array(Image.fromarray(npim1).resize((npim2.shape[1], npim2.shape[0]))).astype(np.uint8)).convert("RGB")

        # save dir of color corrected img
        iphone_fname = Path(eval_source_image_path).stem
        nn_dslr_fname = Path(eval_target_image_path).stem
        assert iphone_fname == nn_dslr_fname

        print(collection_dir / Path(eval_source_image_path).name)
        cc_source_path = collection_dir / Path(eval_source_image_path).name
        path_dict["cc_pred"].append(str(cc_source_path))

        print("path_dict[pred]: ", path_dict["pred"])
        print("path_dict[cc_pred]: ", path_dict["cc_pred"])

        # save the color_corrected_pred -> pred_after_cc_dir
        im3.save(os.path.join(pred_after_cc_dir, Path(eval_source_image_path).name))

        # save the pair (before/after) -> collection_dir
        save_im_path = os.path.join(collection_dir, iphone_fname + ".png")
        concat_im = get_concat_h(im1=im1, im2=im3)
        concat_im.save(save_im_path)

    return path_dict


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
    path_dicts = {str(scene_id): None for scene_id in scene_list}
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

        path_dict = color_correction(
            pred_dir=Path(pred_dir) / scene_id,
            gt_dir=scene.dslr_resized_dir,  # change to the DSLR_undistorted_iphone dir
            output_dir=Path(output_dir) / scene_id,
            image_list=image_list,
            mask_dir=mask_dir,
            scene_id=scene_id,
            verbose=verbose,
        )

        path_dicts[scene_id] = path_dict

    return path_dicts




def main(args):
    if args.scene_id is not None:
        val_scenes = [args.scene_id]
    else:
        with open(args.split, "r") as f:
            val_scenes = f.readlines()
            val_scenes = [x.strip() for x in val_scenes if len(x.strip()) > 0]

    color_correction_all(
        data_root=args.data_root,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        scene_list=val_scenes,
        verbose=True,
    )

    print(val_scenes)
    # if args.device == "cuda" and not torch.cuda.is_available():
    #     args.device = "cpu"
    # all_images, all_psnr, all_ssim, all_lpips = evaluate_all(
    #     args.data_root, args.pred_dir, val_scenes, args.device
    # )



# if __name__ == "__main__":
#     data_root = ""
#     pred_dir = ""
#     scene_list = []
#     upload_path = ""


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
