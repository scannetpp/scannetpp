import argparse
from pathlib import Path
import json
from copy import deepcopy

import numpy as np
import cv2
from tqdm import tqdm

from common.scene_release import ScannetppScene_Release
from common.utils.utils import load_yaml_munch, load_json, read_txt_list


def compute_resize_intrinsic(K, height, width, scale_factor):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    new_K = np.array(
        [
            [fx * scale_factor, 0, cx * scale_factor],
            [0, fy * scale_factor, cy * scale_factor],
            [0, 0, 1],
        ]
    )
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    return new_K, new_height, new_width


def downscale_frames(
    frames,
    K,
    height,
    width,
    downscale_factor,
    input_image_dir,
    input_mask_dir,
    out_image_dir,
    out_mask_dir,
):
    scale_factor = 1 / downscale_factor
    new_K, new_height, new_width = compute_resize_intrinsic(
        K, height, width, scale_factor
    )

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame["file_path"]
        image = cv2.imread(str(image_path))
        resized_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
        )
        out_image_path = Path(out_image_dir) / frame["file_path"]
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_image_path), resized_image)

        mask_path = Path(input_mask_dir) / frame["mask_path"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        resized_mask = cv2.resize(
            mask, (new_width, new_height), interpolation=cv2.INTER_CUBIC
        )
        # Filter the mask valid: 255, invalid: 0
        resized_mask[resized_mask < 255] = 0
        out_mask_path = Path(out_mask_dir) / frame["mask_path"]
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_mask_path), resized_mask)
    return new_K, new_height, new_width


def update_transforms_json(transforms, new_K, new_height, new_width):
    new_transforms = deepcopy(transforms)
    new_transforms["h"] = new_height
    new_transforms["w"] = new_width
    new_transforms["fl_x"] = new_K[0, 0]
    new_transforms["fl_y"] = new_K[1, 1]
    new_transforms["cx"] = new_K[0, 2]
    new_transforms["cy"] = new_K[1, 2]
    return new_transforms


def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get the scenes to process
    if cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)

    # get the options to process
    # go through each scene
    for scene_id in tqdm(scene_ids, desc="scene"):
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")
        input_image_dir = cfg.get("input_image_dir", None)
        if input_image_dir is None:
            input_image_dir = scene.dslr_resized_dir
        else:
            input_image_dir = scene.dslr_dir / input_image_dir

        input_mask_dir = cfg.get("input_mask_dir", None)
        if input_mask_dir is None:
            input_mask_dir = scene.dslr_resized_mask_dir
        else:
            input_mask_dir = scene.dslr_dir / input_mask_dir

        input_transforms_path = cfg.get("input_transforms_path", None)
        if input_transforms_path is None:
            input_transforms_path = scene.dslr_nerfstudio_transform_path
        else:
            input_transforms_path = scene.dslr_dir / input_transforms_path

        out_image_dir = scene.dslr_dir / cfg.out_image_dir
        out_mask_dir = scene.dslr_dir / cfg.out_mask_dir
        out_transforms_path = scene.dslr_dir / cfg.out_transforms_path

        transforms = load_json(input_transforms_path)
        assert len(transforms["frames"]) > 0
        frames = transforms["frames"]
        if "test_frames" not in transforms:
            print(f"{scene_id} has no test split")
        elif not (input_image_dir / transforms["test_frames"][0]["file_path"]).exists():
            print(
                f"{scene_id} test image not found. Might due to the scene belonging to testing scenes. "
                "The resizing will skip those images."
            )
        else:
            assert len(transforms["test_frames"]) > 0
            frames += transforms["test_frames"]

        height = int(transforms["h"])
        width = int(transforms["w"])
        fx = float(transforms["fl_x"])
        fy = float(transforms["fl_y"])
        cx = float(transforms["cx"])
        cy = float(transforms["cy"])
        K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )
        downscale_factor = float(cfg.get("downscale_factor", 2))
        new_K, new_height, new_width = downscale_frames(
            frames,
            K,
            height,
            width,
            downscale_factor,
            input_image_dir,
            input_mask_dir,
            out_image_dir,
            out_mask_dir,
        )
        new_trasforms = update_transforms_json(transforms, new_K, new_height, new_width)
        out_transforms_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_transforms_path, "w") as f:
            json.dump(new_trasforms, f, indent=4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
