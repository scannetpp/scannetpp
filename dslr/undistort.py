import argparse
import os
import tempfile
from pathlib import Path
import json
from copy import deepcopy

import numpy as np
import cv2
from tqdm import tqdm

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
    height,
    width,
    distortion_params,
    input_image_dir,
    input_depth_dir,
    input_normal_dir,
    input_semantic_dir,
    input_mask_dir,
    out_image_dir,
    out_depth_dir,
    out_normal_dir,
    out_semantic_dir, 
    out_mask_dir,
    semantic_palette
):
    new_K = compute_undistort_intrinsic(K, height, width, distortion_params)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, distortion_params, np.eye(3), new_K, (width, height), cv2.CV_32FC1
    )

    for frame in tqdm(frames, desc="frame"):
        # Undistort RGB as default
        image_path = Path(input_image_dir) / frame["file_path"]
        image = cv2.imread(str(image_path))
        undistorted_image = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        out_image_path = Path(out_image_dir) / frame["file_path"]
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_image_path), undistorted_image)

        # Undistort depth as default
        depth_path = Path(input_depth_dir) / frame["file_path"].replace(".JPG", ".png")
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        undistorted_depth = cv2.remap(
            depth,
            map1,
            map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        out_depth_path = Path(out_depth_dir) / frame["file_path"].replace(".JPG", ".png")
        out_depth_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_depth_path), undistorted_depth)
        # Depth_vis
        undistorted_depth_vis = (undistorted_depth.astype(np.float32) / 1000.0)*50
        undistorted_depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(undistorted_depth_vis), cv2.COLORMAP_JET)
        out_depth_vis_path = Path(str(out_depth_dir)+'_vis') / frame["file_path"].replace(".JPG", ".png")
        out_depth_vis_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_depth_vis_path), undistorted_depth_vis)
        
        # Undistort normals as optional
        if input_normal_dir and (~ os.path.exists(Path(input_normal_dir))):
            # todo: undistort normals
            pass

        # Undistort semantics as optional
        if input_semantic_dir and (~ os.path.exists(Path(input_semantic_dir))):
            semantic_path = Path(input_semantic_dir) / frame["file_path"].replace(".JPG", ".png")
            semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
            undistorted_semantic = cv2.remap(
                semantic,
                map1,
                map2,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            out_semantic_path = Path(out_semantic_dir) / frame["file_path"].replace(".JPG", ".png")
            out_semantic_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_semantic_path), undistorted_semantic)
            # Semantic_vis
            viz_color = np.zeros((semantic.shape[0], semantic.shape[1], 3)).astype(np.uint8)
            valid_labels = undistorted_semantic != 65535
            viz_color[valid_labels] = semantic_palette[undistorted_semantic[valid_labels] % len(semantic_palette)]
            out_semantic_vis_path = Path(str(out_semantic_dir)+'_vis') / frame["file_path"].replace(".JPG", ".png")
            out_semantic_vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_semantic_vis_path), viz_color[...,::-1])

        # Undistort mask as default
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

        input_depth_dir = cfg.get("input_depth_dir", None)
        if input_depth_dir is None:
            input_depth_dir = scene.dslr_dir / 'render_depth'
        else:
            input_depth_dir = scene.dslr_dir / input_depth_dir

        input_normal_dir = cfg.get("input_normal_dir", None)
        if input_normal_dir is None:
            pass # no need to undistort semantic
        else:
            input_normal_dir = scene.dslr_dir / input_normal_dir
        
        input_semantic_dir = cfg.get("input_semantic_dir", None)
        if input_semantic_dir is None:
            pass # no need to undistort semantic
        else:
            input_semantic_dir = scene.dslr_dir / input_semantic_dir
        
        out_image_dir = scene.dslr_dir / cfg.out_image_dir
        out_mask_dir = scene.dslr_dir / cfg.out_mask_dir
        out_depth_dir = scene.dslr_dir / cfg.out_depth_dir
        out_normal_dir = scene.dslr_dir / cfg.out_normal_dir
        out_semantic_dir = scene.dslr_dir / cfg.out_semantic_dir
        out_transforms_path = scene.dslr_dir / cfg.out_transforms_path

        semantic_palette = np.loadtxt(cfg['palette_path'], dtype=np.uint8)

        transforms = load_json(input_transforms_path)
        assert len(transforms["frames"]) > 0
        frames = deepcopy(transforms["frames"])
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
        distortion_params = np.array(
            [
                float(transforms["k1"]),
                float(transforms["k2"]),
                float(transforms["k3"]),
                float(transforms["k4"]),
            ]
        )
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

        new_K = undistort_frames(
            frames,
            K,
            height,
            width,
            distortion_params,
            input_image_dir,
            input_depth_dir,
            input_normal_dir,
            input_semantic_dir,
            input_mask_dir,
            out_image_dir,
            out_depth_dir,
            out_normal_dir,
            out_semantic_dir,
            out_mask_dir,
            semantic_palette
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
