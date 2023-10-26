'''
Read training data from pth and visualize it
'''

import argparse
from pathlib import Path
import os
from common.scene_release import ScannetppScene_Release

import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
from common.file_io import load_yaml_munch, read_txt_list


def main(args):
    cfg = load_yaml_munch(args.config_file)

    pth_dir = Path(cfg['pth_dir'])
    
    palette = np.loadtxt(cfg['palette_path'], dtype=np.uint8)

    if 'scene_ids' in cfg:
        print('Viz scenes from list')
        scene_ids = cfg.scene_ids
    elif 'scene_list' in cfg:   
        scene_ids = read_txt_list(cfg['scene_list'])
    else:
        print('Viz all scenes')
        scene_ids = [Path(pth_file).stem for pth_file in os.listdir(pth_dir)]

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # use labels on sampled points or on vertices
    prop_type = 'sampled_' if cfg.use_sampled else 'vtx_'

    for scene_id in tqdm(scene_ids):
        pth_file = f'{scene_id}.pth'
        pth_path = pth_dir / pth_file
        if not pth_path.is_file():
            continue
        pth_data = torch.load(pth_path)
        vtx = pth_data[f'{prop_type}coords']

        scene = ScannetppScene_Release(scene_id, data_root=cfg.data_root)

        if cfg.viz_mesh:
            mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path))

        if cfg.viz_semantic:
            labels = pth_data[f'{prop_type}labels']
            
            viz_color = np.ones((len(labels), 3)) * 255 * 0.8
            valid_labels = labels != cfg.ignore_label
            viz_color[valid_labels] = palette[labels[valid_labels] % len(palette)]
            
            suffix = cfg.output_suffix.semantic
            out_fname = f'{scene_id}{suffix}.ply'

            if cfg.viz_pc:
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(vtx)
                pc.colors = o3d.utility.Vector3dVector(viz_color / 255.0)
                o3d.io.write_point_cloud(str(out_dir / out_fname), pc)
            if cfg.viz_mesh:
                mesh.vertex_colors = o3d.utility.Vector3dVector(viz_color / 255.0)
                o3d.io.write_triangle_mesh(str(out_dir / out_fname), mesh)
        
        if cfg.viz_instances:
            # viz instances
            inst_labels = pth_data[f'{prop_type}instance_labels']
            
            viz_inst_color = np.ones((len(inst_labels), 3)) * 255 * 0.8
            valid_inst_labels = inst_labels != cfg.ignore_label
            viz_inst_color[valid_inst_labels] = palette[inst_labels[valid_inst_labels] % len(palette)]
            
            suffix = cfg.output_suffix.instance
            out_fname = f'{scene_id}{suffix}.ply'

            if cfg.viz_pc:
                inst_pc = o3d.geometry.PointCloud()
                inst_pc.points = o3d.utility.Vector3dVector(vtx)
                inst_pc.colors = o3d.utility.Vector3dVector(viz_inst_color / 255.0)
                o3d.io.write_point_cloud(str(out_dir / out_fname), inst_pc)
            if cfg.viz_mesh:
                mesh.vertex_colors = o3d.utility.Vector3dVector(viz_inst_color / 255.0)
                o3d.io.write_triangle_mesh(str(out_dir / out_fname), mesh)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()
    main(args)