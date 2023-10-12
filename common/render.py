import argparse
import os
import sys
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm
try:
    import renderpy
except ImportError:
    print("renderpy not installed. Please install renderpy from https://github.com/liu115/renderpy")
    sys.exit(1)

from common.utils.colmap import read_model, write_model, Image
from common.scene_release import ScannetppScene_Release
from common.utils.utils import run_command, load_yaml_munch, load_json, read_txt_list


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

    output_dir = cfg.get("output_dir")
    if output_dir is None:
        # default to data folder in data_root
        output_dir = Path(cfg.data_root) / "data"
    output_dir = Path(output_dir)

    render_devices = []
    if cfg.get("render_dslr", False):
        render_devices.append("dslr")
    if cfg.get("render_iphone", False):
        render_devices.append("iphone")

    # go through each scene
    for scene_id in tqdm(scene_ids, desc="scene"):
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")
        render_engine = renderpy.Render()
        render_engine.setupMesh(str(scene.scan_mesh_path))
        for device in render_devices:
            if device == "dslr":
                cameras, images, points3D = read_model(scene.dslr_colmap_dir, ".txt")
            else:
                cameras, images, points3D = read_model(scene.iphone_colmap_dir, ".txt")
            assert len(cameras) == 1, "Multiple cameras not supported"
            camera = next(iter(cameras.values()))

            fx, fy, cx, cy = camera.params[:4]
            params = camera.params[4:]
            camera_model = camera.model
            render_engine.setupCamera(
                camera.height, camera.width,
                fx, fy, cx, cy,
                camera_model,
                params,      # Distortion parameters np.array([k1, k2, k3, k4]) or np.array([k1, k2, p1, p2])
            )

            near = cfg.get("near", 0.05)
            far = cfg.get("far", 20.0)
            rgb_dir = Path(cfg.output_dir) / scene_id / device / "render_rgb"
            depth_dir = Path(cfg.output_dir) / scene_id / device / "render_depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            for image_id, image in tqdm(images.items(), f"Rendering {device} images"):
                world_to_camera = image.world_to_camera
                rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
                rgb = rgb.astype(np.uint8)
                # Make depth in mm and clip to fit 16-bit image
                depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
                imageio.imwrite(rgb_dir / image.name, rgb)
                depth_name = image.name.split(".")[0] + ".png"
                imageio.imwrite(depth_dir / depth_name, depth)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
