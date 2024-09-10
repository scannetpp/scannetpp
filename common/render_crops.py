import argparse
import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt

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

from common.render_crops_utils import vert_to_obj_lookup, CropHeap, crop_rgb_mask, plot_grid_images



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
        raise Exception("This code is has not been tested with the DSLR data.")
    if cfg.get("render_iphone", False):
        render_devices.append("iphone")

    # go through each scene
    for scene_id in tqdm(scene_ids, desc="scene"):
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")
        render_engine = renderpy.Render()
        render_engine.setupMesh(str(scene.scan_mesh_path))

        # Load annotations
        segments_anno = json.load(open(scene.scan_anno_json_path, "r"))
        n_objects = len(segments_anno["segGroups"])
        crop_heaps = [CropHeap(max_size=4) for _ in range(n_objects + 1)] 
        instance_colors = np.random.randint(low=0, high=256, size=(n_objects + 1, 3), dtype=np.uint8)
        instance_colors[0] = 255 # White bg
        vert_to_obj = vert_to_obj_lookup(segments_anno)

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
            rgb_dir = Path(cfg.output_dir) / scene_id / device / "render_rgb"
            depth_dir = Path(cfg.output_dir) / scene_id / device / "render_depth"
            crop_dir = Path(cfg.output_dir) / scene_id / device / "render_crops"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            crop_dir.mkdir(parents=True, exist_ok=True)

            temp = 0
            for image_id, image in tqdm(images.items(), f"Rendering object crops using {device} images"):
                world_to_camera = image.world_to_camera

                rgb_rendered, _, vert_indices = render_engine.renderAll(world_to_camera, near, far)

                iphone_rgb_path = Path(scene.iphone_rgb_dir) / image.name
                rgb = np.asarray(imageio.imread(iphone_rgb_path))

                vert_instance = vert_to_obj[vert_indices]
                pix_instance = vert_instance[:, :, 0] # Some triangles actually belong to different objects. I don't think it will matter for crops.

                # Visualize instances
                # instance_rgb = instance_colors[pix_instance]
                # imageio.imwrite(rgb_dir / image.name, instance_rgb)

                objs = np.unique(pix_instance)

                for obj in objs:
                    mask = pix_instance == obj
                    crop = crop_rgb_mask(rgb, mask, inflate_px=50)
                    crop_heaps[obj].push(crop)

                temp += 1
                if temp == 30:
                    break


                # instance_rgb = instance_rgb.astype(np.uint8)
                # # Make depth in mm and clip to fit 16-bit image
                # depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
                # depth_name = image.name.split(".")[0] + ".png"
                # imageio.imwrite(depth_dir / depth_name, depth)

        for i, heap in tqdm(enumerate(crop_heaps), f"Rendering image grids"):
            if len(heap):
                crops = heap.get_sorted()
                rgbs = [c.rgb for c in crops]
                masks = [c.mask for c in crops]
                plot_grid_images(rgbs + masks, grid_width=len(rgbs))
                plt.savefig(crop_dir / f"{str(i).zfill(5)}.jpg")
                plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
