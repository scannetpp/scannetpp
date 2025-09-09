import argparse
import os
import tempfile
from pathlib import Path
import json
from copy import deepcopy

import numpy as np
import cv2
from tqdm import tqdm
import shutil

from common.scene_release import ScannetppScene_Release
from common.utils.utils import load_yaml_munch, load_json, read_txt_list


def compute_undistort_intrinsic(K, height, width, distortion_params):
    assert len(distortion_params.shape) == 1
    assert distortion_params.shape[0] == 4  # OPENCV_FISHEYE has k1, k2, k3, k4

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        distortion_params,
        (width, height),
        np.eye(3),
        balance=0.0,
    )
    # Make the cx and cy to be the center of the image
    new_K[0, 2] = width / 2.0
    new_K[1, 2] = height / 2.0
    return new_K


def undistort_frames(
    frames,
    K,
    dslr_K,
    dslr_distortion_params,
    height,
    width,
    distortion_params,
    input_image_dir,
    input_mask_dir,
    out_image_dir,
    out_mask_dir,
):
    """
        K: iphone intrinsic matrix
        height: iphone image height
        width: iphone image width
        distortion_params: iphone OPENCV model distortion param: [k1, k2, p1, p2]
    """
    
    # OPENCV
    new_K = K

    # get mapping by cv2.initUndistortRectifyMap() from distorted image to undistorted image
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        dslr_K, dslr_distortion_params, np.eye(3), new_K, (width, height), cv2.CV_32FC1
    )

    # undistort DSLR by given iphone intrinsic *new K which is appropriatedly converted into 
    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame["file_path"]
        image = cv2.imread(str(image_path))
        undistorted_image = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode = cv2.BORDER_CONSTANT, # borderMode=cv2.BORDER_REFLECT_101,
        )
        out_image_path = Path(out_image_dir) / frame["file_path"]
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_image_path), undistorted_image)

        # Mask
        if "mask_path" in list(frame.keys()):
            mask_path = Path(input_mask_dir) / frame["mask_path"]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if np.all(mask > 0):
                # No invalid pixels. Just use empty mask
                undistorted_mask = np.zeros((height, width), dtype=np.uint8) + 255
            else:
                undistorted_mask = cv2.remap(
                    mask,
                    map1,
                    map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255,
                )
                # Filter the mask valid: 255, invalid: 0
                undistorted_mask[undistorted_mask < 255] = 0

            out_mask_path = Path(out_mask_dir) / frame["mask_path"]
            out_mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_mask_path), undistorted_mask)
    return new_K


def update_transforms_json(transforms, new_K, new_height, new_width):
    new_transforms = deepcopy(transforms)
    new_transforms["h"] = new_height
    new_transforms["w"] = new_width
    new_transforms["fl_x"] = new_K[0, 0]
    new_transforms["fl_y"] = new_K[1, 1]
    new_transforms["cx"] = new_K[0, 2]
    new_transforms["cy"] = new_K[1, 2]
    # The undistortion will be PINHOLE and have no distortion paramaters
    new_transforms["camera_model"] = "PINHOLE"
    for key in ("k1", "k2", "k3", "k4"):
        if key in new_transforms:
            new_transforms[key] = 0.0
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
    for scene_id in tqdm(scene_ids, desc='scene'):
        print("scene_id: ", scene_id)
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / 'data')

        # path to save the undistorted DSLR frames by iPhone 
        output_dir = Path(cfg.data_root) / cfg.folder_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        output_scene_dir = os.path.join(output_dir, scene_id)
        if not os.path.exists(output_scene_dir):
            os.makedirs(output_scene_dir, exist_ok=True)

        output_dslr_dir = os.path.join(output_scene_dir, "dslr")
        if not os.path.exists(output_dslr_dir):
            os.makedirs(output_dslr_dir, exist_ok=True)

        output_colmap_dir = None
        output_colmap_dir = os.path.join(output_dslr_dir, "colmap")
        if not os.path.exists(output_colmap_dir):
            os.makedirs(output_colmap_dir, exist_ok=True)

        print("output_colmap_dir: ", output_colmap_dir)
        # copy and paste scene.dslr_dir/colmap
        new_colmap_path = shutil.copytree(os.path.join(scene.dslr_dir, "colmap"), output_colmap_dir, dirs_exist_ok=True)

        out_image_dir = None
        out_mask_dir = None
        out_transforms_path = None

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

        out_image_dir = Path(output_dslr_dir) / cfg.out_image_dir
        out_mask_dir = Path(output_dslr_dir) / cfg.out_mask_dir
        out_transforms_path = Path(output_dslr_dir) / cfg.out_transforms_path

        print(out_image_dir)
        print(out_mask_dir)
        print(out_transforms_path)

        transforms = load_json(input_transforms_path)
        assert len(transforms["test_frames"]) > 0
        frames = deepcopy(transforms["test_frames"])

        iphone_transforms = load_json(scene.iphone_nerfstudio_transform_path)

        height = int(iphone_transforms["h"])
        width = int(iphone_transforms["w"])
        distortion_params = np.array(
            [
                float(iphone_transforms["k1"]),
                float(iphone_transforms["k2"]),
                float(iphone_transforms["p1"]),
                float(iphone_transforms["p2"]),
            ]
        )
        fx = float(iphone_transforms["fl_x"])
        fy = float(iphone_transforms["fl_y"])
        cx = float(iphone_transforms["cx"])
        cy = float(iphone_transforms["cy"])
        K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )


        d_fx = float(transforms["fl_x"])
        d_fy = float(transforms["fl_y"])
        d_cx = float(transforms["cx"])
        d_cy = float(transforms["cy"])
        dslr_K = np.array(
            [
                [d_fx, 0, d_cx],
                [0, d_fy, d_cy],
                [0, 0, 1],
            ]
        )

        d_distortion_params = np.array(
            [
                float(transforms["k1"]),
                float(transforms["k2"]),
                float(transforms["k3"]),
                float(transforms["k4"]),
            ]
        )

        new_K = undistort_frames(
            frames,
            K,
            dslr_K,
            d_distortion_params,
            height,
            width,
            distortion_params,
            input_image_dir,
            input_mask_dir,
            out_image_dir,
            out_mask_dir,
        )
        new_trasforms = update_transforms_json(transforms, new_K, height, width)
        out_transforms_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_transforms_path, "w") as f:
            json.dump(new_trasforms, f, indent=4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
