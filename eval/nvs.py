import argparse
from typing import List, Optional, Union
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from common.scene_release import ScannetppScene_Release


def evaluate_scene(
    pred_dir: Union[str, Path],
    gt_dir: Union[str, Path],
    image_list: List[str],
    mask_dir: Optional[Union[str, Path]] = None,
    auto_resize: bool = True,
    scene_id: str = "unknown",
    verbose: bool = False,
):
    """Evaluate a scene using PSNR, SSIM and LPIPS metrics.
    Args:
        pred_dir: Path to the predicted images.
        gt_dir: Path to the ground truth images.
        image_list: List of image filenames to evaluate.
        mask_dir: Path to the masks. If provided, the metrics will be computed only on the masked regions.
        auto_resize: If True, automatically resize the predicted images to match the GT image size.
        scene_id: Scene identifier for verbose print.
        verbose: If True, print the metrics.
    """

    # images = os.listdir(gt_dir)
    # assert len(images) > 0, f"no images found in GT dir: {gt_dir}"
    # assert len(images) == len(os.listdir(pred_dir)), f"GT and pred dirs have different number of images"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    # TODO: make sure which one should be used (VGG or Alex)
    lpip_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", normalize=True
    ).to(device)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    for image_fn in image_list:
        assert os.path.exists(
            os.path.join(gt_dir, image_fn)
        ), f"GT image not found: {image_fn}"
        gt_image = Image.open(os.path.join(gt_dir, image_fn))
        assert os.path.exists(
            os.path.join(pred_dir, image_fn)
        ), f"pred image not found: {image_fn}"
        pred_image = Image.open(os.path.join(pred_dir, image_fn))

        if mask_dir is not None:
            mask_path = os.path.join(mask_dir, image_fn.split(".")[0] + ".png")
            assert os.path.exists(mask_path), f"mask not found: {mask_path}"
            mask = Image.open(mask_path)
            mask = torch.from_numpy(np.array(mask))
            mask = (mask > 0).bool()
            assert (
                len(mask.shape) == 2
            ), f"mask should have 2 channels (H, W) but get shape: {mask.shape}"
        else:
            mask = None

        if gt_image.size != pred_image.size:
            if auto_resize:
                # Auto resized to match the GT image size
                pred_image = pred_image.resize(gt_image.size)
            else:
                raise ValueError(
                    f"GT and pred images have different sizes: {gt_image.size} != {pred_image.size}"
                )

        gt_image = torch.from_numpy(np.array(gt_image)).float() / 255.0
        pred_image = torch.from_numpy(np.array(pred_image)).float() / 255.0
        assert (
            len(gt_image.shape) == 3
        ), f"GT image should have 3 channels (H, W, 3) but get shape: {gt_image.shape}"
        assert (
            len(pred_image.shape) == 3
        ), f"pred image should have 3 channels (H, W, 3) but get shape: {pred_image.shape}"

        gt_image = gt_image.permute(2, 0, 1).unsqueeze(0)
        pred_image = pred_image.permute(2, 0, 1).unsqueeze(0)
        # pred_image += torch.rand_like(pred_image) * 1e-2
        # pred_image = torch.clamp(pred_image, 0.0, 1.0)

        if mask is not None:
            valid_gt = torch.masked_select(gt_image, mask).view(3, -1)
            valid_pred = torch.masked_select(pred_image, mask).view(3, -1)
            psnr = psnr_metric(valid_pred, valid_gt)

            # Fill the invalid region with the average color for SSIM and LPIPS
            average_color = valid_gt.mean(dim=-1)
            for i in range(3):
                pred_image[0, i, ...] = torch.masked_fill(
                    pred_image[0, i, ...], ~mask, average_color[i]
                )
                gt_image[0, i, ...] = torch.masked_fill(
                    gt_image[0, i, ...], ~mask, average_color[i]
                )
        else:
            psnr = psnr_metric(pred_image, gt_image)
        ssim = structural_similarity_index_measure(pred_image, gt_image)
        lpips = lpip_metric(pred_image.to(device), gt_image.to(device))
        # Save the images for debugging
        pred_image = pred_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_image = gt_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

        os.makedirs(os.path.join("/tmp/val_debug", scene_id, "pred"), exist_ok=True)
        os.makedirs(os.path.join("/tmp/val_debug", scene_id, "gt"), exist_ok=True)
        Image.fromarray((pred_image * 255).astype(np.uint8)).save(
            os.path.join("/tmp/val_debug", scene_id, "pred", image_fn)
        )
        Image.fromarray((gt_image * 255).astype(np.uint8)).save(
            os.path.join("/tmp/val_debug", scene_id, "gt", image_fn)
        )

        psnr_values.append(psnr.item())
        ssim_values.append(ssim.item())
        lpips_values.append(lpips.item())

    if verbose:
        print(
            f"Scene: {scene_id} PSNR: {np.mean(psnr_values):.4f} +/- {np.std(psnr_values):.4f}"
        )
        print(
            f"Scene: {scene_id} SSIM: {np.mean(ssim_values):.4f} +/- {np.std(ssim_values):.4f}"
        )
        print(
            f"Scene: {scene_id} LPIPS: {np.mean(lpips_values):.4f} +/- {np.std(lpips_values):.4f}"
        )

    return psnr_values, ssim_values, lpips_values


def get_test_images(transforms_path: str):
    with open(transforms_path, "r") as f:
        transforms = json.load(f)
    image_list = [x["file_path"] for x in transforms["test_frames"]]
    return image_list


def evaluate_all(data_root, pred_dir, scene_list):
    all_psnr = []
    all_ssim = []
    all_lpips = []

    for scene_id in scene_list:
        scene = ScannetppScene_Release(scene_id, data_root=data_root)
        image_list = get_test_images(scene.dslr_nerfstudio_transform_path)
        print(image_list)

        scene_psnr, scene_ssim, scene_lpips = evaluate_scene(
            Path(pred_dir) / scene_id,
            scene.dslr_resized_dir,
            image_list,
            scene.dslr_resized_mask_dir,
            auto_resize=True,
            scene_id=scene_id,
            verbose=True,
        )
        all_psnr += scene_psnr
        all_ssim += scene_ssim
        all_lpips += scene_lpips
    return all_psnr, all_ssim, all_lpips


def main(args):
    if args.scene_id is not None:
        val_scenes = [args.scene_id]
    else:
        with open(args.split, "r") as f:
            val_scenes = f.readlines()
            val_scenes = [x.strip() for x in val_scenes]

    all_psnr, all_ssim, all_lpips = evaluate_all(args.data_root, args.pred_dir, val_scenes)
    print(f"Overall PSNR: {np.mean(all_psnr):.4f} +/- {np.std(all_psnr):.4f}")
    print(f"Overall SSIM: {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f}")
    print(f"Overall LPIPS: {np.mean(all_lpips):.4f} +/- {np.std(all_lpips):.4f}")


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
    args = p.parse_args()
    main(args)

    # bcd2436daf f3685d06a9 have mask
    # python -m eval.nvs --data_root /home/data --scene_id 3db0a1c8f3 --pred_dir val_pred
