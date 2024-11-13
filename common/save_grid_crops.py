import argparse
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2
from tqdm import tqdm

import numpy as np

from common.utils.utils import load_yaml_munch, read_txt_list
from common.render_crops_utils import plot_2x2_grid_images


def load_crops_data(data_dir):
    crops_data = dict()

    rgb_dir = data_dir / "rgbs"
    mask_dir = data_dir / "masks"
    metadata_dir = data_dir / "metadata"

    metadata_files = natsorted(metadata_dir.glob("*.json"))

    for metadata_file in metadata_files:
        obj_id = metadata_file.stem.split("_")[0]
        obj_id_int = int(obj_id)

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        crops_data[obj_id_int] = {
            "rgbs": [],
            "masks": [],
            "scores": metadata["scores"],
            "label": metadata["label"],
        }

        rgb_files = natsorted(rgb_dir.glob(f"{obj_id}_*_rgb.png"))
        mask_files = natsorted(mask_dir.glob(f"{obj_id}_*_mask.npy"))

        for rgb_file, mask_file in zip(rgb_files, mask_files):
            rgb_image = cv2.imread(str(rgb_file))
            rgb_image = cv2.cvtColor(
                rgb_image, cv2.COLOR_BGR2RGB
            )  # Convert to RGB format

            mask = np.load(mask_file)

            crops_data[obj_id_int]["rgbs"].append(rgb_image)
            crops_data[obj_id_int]["masks"].append(mask)

    return crops_data


def get_data_from_crops(crops_data, obj_id):
    rgbs = crops_data[obj_id]["rgbs"]
    masks = crops_data[obj_id]["masks"]
    scores = crops_data[obj_id]["scores"]
    label = crops_data[obj_id]["label"]

    return rgbs, masks, scores, label


def main(args):
    cfg = load_yaml_munch(args.config_file)

    if cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)

    output_dir = cfg.get("output_dir")
    if output_dir is None:
        output_dir = Path(cfg.data_root) / "data"
    output_dir = Path(output_dir)

    render_devices = []
    if cfg.get("render_dslr", False):
        render_devices.append("dslr")
        raise Exception("This code has not been tested with the DSLR data.")
    if cfg.get("render_iphone", False):
        render_devices.append("iphone")

    # Assuming one one device for simplicity
    device = render_devices[0]

    for scene_id in tqdm(scene_ids):
        # Set path for crops data
        crops_data_dir = output_dir / scene_id / device / "crops_data"
        crops_data = load_crops_data(crops_data_dir)
        obj_ids = list(crops_data.keys())

        # Directory to save rendered crops
        plot_save_dir = crops_data_dir.parent / "render_crops_2x2"
        plot_save_dir.mkdir(parents=True, exist_ok=True)

        for obj_id in tqdm(obj_ids):
            rgbs, masks, _, _ = get_data_from_crops(crops_data, obj_id)

            plot_2x2_grid_images(rgbs, masks)
            plt.savefig(plot_save_dir / f"{str(obj_id).zfill(5)}.jpg")
            plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "config_file", help="Path to config file", default="configs/render_crops.yaml"
    )
    args = p.parse_args()

    main(args)
