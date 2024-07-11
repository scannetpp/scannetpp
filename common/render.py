import argparse
import os
import sys
from pathlib import Path

import imageio
import scipy
import cv2
import open3d as o3d
import numpy as np
import pandas as pd
from tqdm import tqdm
from plyfile import PlyData

from semantic.prep.map_semantic import filter_map_classes

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

        plydata = PlyData.read(str(scene.scan_sem_mesh_path))
        semantic_mesh = plydata['vertex']['label']
        
        pcd = o3d.io.read_point_cloud(str(scene.scan_mesh_path))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_normals = np.asarray(pcd.normals)
        
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

            # labels mapping-reference
            semantic_classes = np.genfromtxt(cfg['semantic_classes'], dtype=str, delimiter='\n')
            label_mapping_file = pd.read_csv(cfg['mapping_file'])
            mapped_classes, label_mapping = filter_map_classes(label_mapping_file, thresh=cfg.get('mapping_thresh', 0),
                                count_type='count', mapping_type='semantic')
            
            # mapping labels to specific classes (e.g. NYU-40)
            with open(cfg['labels_path']) as f: 
                class_names = f.read().splitlines()
            mapping = {label: ndx for (ndx, label) in enumerate(class_names)}
            # todo: NYU-40 mapping
            from labels_mapping import raw_to_nyu40
            mapping.update(raw_to_nyu40)
            mapping_dict = {}

            rgb_dir = Path(cfg.output_dir) / scene_id / device / "render_rgb"
            depth_dir = Path(cfg.output_dir) / scene_id / device / "render_depth"
            depth_vis_dir = Path(cfg.output_dir) / scene_id / device / "render_depth_vis"
            normal_dir = Path(cfg.output_dir) / scene_id / device / "render_normal"
            normal_vis_dir = Path(cfg.output_dir) / scene_id / device / "render_normal_vis"
            semantic_dir = Path(cfg.output_dir) / scene_id / device / cfg.render_semantics_dir
            semantic_vis_dir = Path(cfg.output_dir) / scene_id / device / (cfg.render_semantics_dir+'_vis')

            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True); depth_vis_dir.mkdir(parents=True, exist_ok=True)
            if cfg.get("render_normals", False):
                normal_dir.mkdir(parents=True, exist_ok=True); normal_vis_dir.mkdir(parents=True, exist_ok=True)
            if cfg.get("render_semantics", False):
                semantic_dir.mkdir(parents=True, exist_ok=True); semantic_vis_dir.mkdir(parents=True, exist_ok=True)
            
            for image_id, image in tqdm(images.items(), f"Rendering {device} images"):
                world_to_camera = image.world_to_camera
                rgb, depth, vert_indices = render_engine.renderAll(world_to_camera, near, far)
                
                # RGB image
                rgb = rgb.astype(np.uint8)
                imageio.imwrite(rgb_dir / image.name, rgb)
                
                # Depth image
                # Make depth in mm and clip to fit 16-bit image
                depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
                depth_vis = (depth.copy().astype(np.float32) / 1000.0)*50
                depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_vis), cv2.COLORMAP_JET)

                depth_name = image.name.split(".")[0] + ".png"
                imageio.imwrite(depth_dir / depth_name, depth)
                imageio.imwrite(depth_vis_dir / depth_name, depth_vis[...,::-1])
                
                # todo: Normal images
                # select the normlas of the first  of vertices or the mean of 3 vertices
                if cfg.get("render_normals", False):
                    vert_normals = np.where((vert_indices>0)[:, :, :, np.newaxis].repeat(3, axis=3), 
                                            pcd_normals[vert_indices], 
                                            np.nan)
                    normal = np.nanmean(vert_normals, axis=-2)
                    # normal = vert_normals[..., 0, :]

                    valid_mask = np.isfinite(normal)[..., 0]
                    normal[~valid_mask] = np.array([1, 1, 1])

                    normal_vis = ( normal/np.linalg.norm(normal, axis=-1, keepdims=True) + 1)*0.5
                    normal_vis[~valid_mask] = np.array([1, 1, 1])

                    normal_name = image.name.split(".")[0] + ".png"
                    cv2.imwrite(normal_dir / normal_name, normal)
                    imageio.imwrite(normal_vis_dir / normal_name, (normal_vis*255).astype(np.uint8))

                # Semantics images
                # Select vertices' semantic labels using majority vote
                if cfg.get("render_semantics", False):
                    vert_labels = np.where(vert_indices>0, semantic_mesh[vert_indices], -100) # Unlabeled vertices have value -100
                    semantic = scipy.stats.mode(vert_labels, axis=-1).mode 
                    # semantic = vert_labels[...,0] # Mode is quite time expensive, to be faster you can just select the first, although you will lose accuracy
                    
                    # labels re-mapping
                    (h, w) = semantic.shape
                    semantic_view = semantic.reshape(-1)
                    old_label_lis = [semantic_classes[semantic_idx] if semantic_idx>=0 else 'Void' for semantic_idx in semantic_view ]
                    new_label_lis = [label_mapping.get(old_label_idx, 'Void') for old_label_idx in old_label_lis]
                    # Semantic mapping
                    semantic_new = []
                    for new_labels in new_label_lis:
                        semantic_new_ndx = mapping.get(new_labels, cfg.ignore_label)
                        semantic_new.append(semantic_new_ndx)
                        if new_labels not in mapping_dict:
                            mapping_dict[new_labels] = [semantic_new_ndx, class_names[semantic_new_ndx] if semantic_new_ndx>-1 else 'Void']
                        
                    semantic = np.array(semantic_new).reshape(h, w)
                    # visualize semantic labels
                    palette = np.loadtxt(cfg['palette_path'], dtype=np.uint8)
                    viz_color = np.zeros((semantic.shape[0], semantic.shape[1], 3)).astype(np.uint8)
                    valid_labels = semantic != cfg.ignore_label
                    viz_color[valid_labels] = palette[semantic[valid_labels] % len(palette)]
                    
                    semantic = np.where(semantic>=0, semantic, 65535).astype(np.uint16) 
                    semantic_vis = viz_color

                    # save semantic labels as png to undistort using the same script than for depth
                    semantic_name = image.name.split(".")[0] + ".png"
                    imageio.imwrite(semantic_dir / semantic_name, semantic)
                    imageio.imwrite(semantic_vis_dir / semantic_name, semantic_vis)
            
            if cfg.get("render_semantics", False):
                with open(semantic_dir / "mapping.json", 'w') as f:
                    import json
                    json.dump(mapping_dict, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
