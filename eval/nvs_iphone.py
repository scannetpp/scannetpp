import argparse

import numpy as np
import torch

from eval.nvs import evaluate_all
from iphone.color_correction import color_correction_all


def main(args):
    if args.scene_id is not None:
        val_scenes = [args.scene_id]
    else:
        with open(args.split, "r") as f:
            val_scenes = f.readlines()
            val_scenes = [x.strip() for x in val_scenes if len(x.strip()) > 0]
    print(val_scenes)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    print("Starting color correction")
    color_correction_all(
        args.data_root,
        args.pred_dir,
        args.output_dir,
        val_scenes,
        verbose=True,
    )

    print("Starting evaluation before color_correction")
    _, psnr_orig, ssim_orig, lpips_orig = evaluate_all(
        args.data_root,
        args.pred_dir,
        val_scenes,
        args.device,
        verbose=False,
        custom_dslr_folder_name="dslr_undistorted_by_iphone",
    )
    psnr_orig = np.concatenate(psnr_orig)
    ssim_orig = np.concatenate(ssim_orig)
    lpips_orig = np.concatenate(lpips_orig)
    print(f"Overall PSNR before color correction: {np.mean(psnr_orig):.4f} +/- {np.std(psnr_orig):.4f}")
    print(f"Overall SSIM before color correction: {np.mean(ssim_orig):.4f} +/- {np.std(ssim_orig):.4f}")
    print(f"Overall LPIPS before color correction: {np.mean(lpips_orig):.4f} +/- {np.std(lpips_orig):.4f}")

    print("Starting evaluation after color_correction")
    _, psnr_cc, ssim_cc, lpips_cc = evaluate_all(
        args.data_root,
        args.output_dir,
        val_scenes,
        args.device,
        verbose=False,
        custom_dslr_folder_name="dslr_undistorted_by_iphone",
    )
    print(f"Overall PSNR after color correction: {np.mean(psnr_cc):.4f} +/- {np.std(psnr_cc):.4f}")
    print(f"Overall SSIM after color correction: {np.mean(ssim_cc):.4f} +/- {np.std(ssim_cc):.4f}")
    print(f"Overall LPIPS after color correction: {np.mean(lpips_cc):.4f} +/- {np.std(lpips_cc):.4f}")


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
    p.add_argument("--output_dir", help="Place to store all the color corrected images", required=True)
    p.add_argument("--device", help="Device", default="cuda")
    args = p.parse_args()
    main(args)

    # python -m eval.nvs_iphone --data_root /home/data --scene_id 3db0a1c8f3 --pred_dir val_pred --output_dir val_pred_cc
