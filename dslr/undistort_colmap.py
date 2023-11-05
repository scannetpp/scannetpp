import argparse
import os
import tempfile
from pathlib import Path
import shutil
import json

import imageio
import numpy as np
from tqdm import tqdm

from common.utils.colmap import read_model, write_model, Image
from common.scene_release import ScannetppScene_Release
from common.utils.utils import run_command, load_yaml_munch, load_json, read_txt_list
from common.utils.nerfstudio import convert_camera


def undistort_anon_masks(
    image_dir: Path,
    input_model_dir: Path,
    output_dir: Path,
    colmap_exec: Path = "colmap",
    max_size: int = 2000,
    crop: bool = False,
):
    """Undistort masks using COLMAP.
    args:
        image_dir: Path to the directory containing the masks.
        input_model_dir: Path to the directory containing the COLMAP model.
        output_dir: Path to the directory where the undistorted masks will be saved.
        colmap_exec: Path to the COLMAP executable.
        max_size: Maximum size of the undistorted images.
        crop: Whether to crop the image borders (x-axis) during the undistortion.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cameras, images, points3D = read_model(input_model_dir, '.txt')
        new_images = {}
        # Replace the image path (ends with '.JPG') with the mask path (ends with '.png')
        for image_id, image in images.items():
            new_image = image._asdict()
            new_image["name"] = new_image["name"].replace(".JPG", ".png")
            new_images[image_id] = Image(**new_image)

        cur_model_dir = tmpdir / "mask_sparse"
        cur_model_dir.mkdir(parents=True, exist_ok=True)
        write_model(cameras, new_images, points3D, cur_model_dir, ".txt")

        undistort_dir = tmpdir / "mask_undistort"

        command = (
            f"{colmap_exec} image_undistorter"
            " --output_type COLMAP"
            f" --max_image_size {max_size}"
            f" --image_path {image_dir}"
            f" --input_path {cur_model_dir}"
            f" --output_path {undistort_dir}"
        )
        if crop:
            command += (
                f" --roi_min_x 0.125"
                f" --roi_min_y 0"
                f" --roi_max_x 0.875"
                f" --roi_max_y 1"
            )
        run_command(command)

        # # Convert model from .bin to .txt
        # run_command(
        #     f"{colmap_exec} model_converter"
        #     f" --input_path {undistort_dir}/sparse"
        #     f" --output_path {undistort_dir}/sparse"
        #     " --output_type TXT"
        # )

        # Go through all the image masks and make sure they are all 0 or 255
        cameras, images, points3D = read_model(undistort_dir / "sparse")
        for image_id, image in images.items():
            image_path = undistort_dir / "images" / image.name
            mask = imageio.imread(image_path)
            mask = np.array(mask, dtype=np.uint8)
            if (mask == 255).all():
                # The mask is all 255
                continue
            in_between = np.logical_and(mask > 0, mask < 255)
            mask[in_between] = 0
            imageio.imwrite(image_path, mask)

        shutil.move(undistort_dir / "images", output_dir / "masks")


def undistort_images(
    image_dir: Path,
    input_model_dir: Path,
    output_dir: Path,
    colmap_exec: Path = "colmap",
    max_size: int = 2000,
    crop: bool = False,
):
    """Undistort images using COLMAP.
    args:
        image_dir: Path to the directory containing the images.
        input_model_dir: Path to the directory containing the COLMAP model.
        output_dir: Path to the directory where the undistorted images will be saved.
        colmap_exec: Path to the COLMAP executable.
        max_size: Maximum size of the undistorted images.
        crop: Whether to crop the image borders (x-axis) during the undistortion.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        undistort_dir = tmpdir / "undistort"

        command = (
            f"{colmap_exec} image_undistorter"
            " --output_type COLMAP"
            f" --max_image_size {max_size}"
            f" --image_path {image_dir}"
            f" --input_path {input_model_dir}"
            f" --output_path {undistort_dir}"
        )
        if crop:
            command += (
                f" --roi_min_x 0.125"
                f" --roi_min_y 0"
                f" --roi_max_x 0.875"
                f" --roi_max_y 1"
            )
        run_command(command)

        # Convert model from .bin to .txt
        run_command(
            f"{colmap_exec} model_converter"
            f" --input_path {undistort_dir}/sparse"
            f" --output_path {undistort_dir}/sparse"
            " --output_type TXT"
        )
        os.remove(undistort_dir / "sparse" / "cameras.bin")
        os.remove(undistort_dir / "sparse" / "images.bin")
        os.remove(undistort_dir / "sparse" / "points3D.bin")
        shutil.move(undistort_dir / "images", output_dir / "images")
        shutil.move(undistort_dir / "sparse", output_dir / "colmap")


def update_transforms_json(model_path, old_json, output_json):
    cameras, images, points3D = read_model(model_path, ".txt")
    assert len(cameras) == 1, "Multiple cameras not supported"
    camera = next(iter(cameras.values()))
    transforms = load_json(old_json)
    new_transforms = transforms
    new_transforms.update(convert_camera(camera))
    print(convert_camera(camera))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(new_transforms, f, indent=4)


def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get the scenes to process
    if cfg.get('scene_ids'):
        scene_ids = cfg.scene_ids
    elif cfg.get('splits'):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / 'splits' / f'{split}.txt'
            scene_ids += read_txt_list(split_path)

    # get the options to process
    # go through each scene
    for scene_id in tqdm(scene_ids, desc='scene'):
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / 'data')
        train_test_list = load_json(scene.dslr_train_test_lists_path)
        train_list = train_test_list['train']
        test_list = train_test_list['test']
        assert len(train_list) > 0
        assert len(test_list) > 0

        output_dir = Path(cfg.output_dir) / scene_id
        output_dir.mkdir(exist_ok=True, parents=True)
        undistort_anon_masks(
            image_dir=scene.dslr_resized_mask_dir,
            input_model_dir=scene.dslr_colmap_dir,
            output_dir=output_dir,
            max_size=cfg.max_size,
            crop=cfg.crop_border,
            colmap_exec=cfg.colmap_exec,
        )
        undistort_images(
            image_dir=scene.dslr_resized_dir,
            input_model_dir=scene.dslr_colmap_dir,
            output_dir=output_dir,
            max_size=cfg.max_size,
            crop=cfg.crop_border,
            colmap_exec=cfg.colmap_exec,
        )
        update_transforms_json(
            output_dir / "colmap",
            scene.dslr_nerfstudio_transform_path,
            output_dir / "nerfstudio/transforms.json",
        )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()

    main(args)
