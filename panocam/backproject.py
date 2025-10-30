'''
Load the panocam RGB, depth, azimuth and elev maps and backproject to 3D
'''

import hydra
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os

from common.scene_release import ScannetppScene_Release
from common.file_io import load_json

def backproject(pano_rgb, pano_depth, azimuth, elevation):
    # Convert depth image to point cloud
    height, width = pano_depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    mask = pano_depth > 0
    u = u[mask > 0]
    v = v[mask > 0]

    dist = pano_depth[v, u]

    # From angular (theta, phi) from spherical coordinate and the distance to x, y, z.
    azimuth = azimuth[v, u]
    elevation = elevation[v, u]

    x = dist * np.sin(elevation) * np.cos(azimuth)
    y = dist * np.sin(elevation) * np.sin(azimuth)
    z = dist * np.cos(elevation)
    xyz = np.stack((x, y, z), axis=-1)
    
    rgb = pano_rgb[v, u]   # (N, 3)
    
    return xyz, rgb

@hydra.main(version_base=None, config_path="configs", config_name="backproject")
def main(cfg):
    print(f'Config: {cfg}')

    save_dir = Path(cfg.save_dir)

    for scene_id in tqdm(cfg.scene_ids, desc="scene"):
        print(f'Processing scene: {scene_id}')

        scene = ScannetppScene_Release(scene_id, data_root=cfg.data_dir)

        scan_poses = load_json(scene.scan_transformed_poses_path)
        scan_poses = np.array(scan_poses)

        # get the list of images
        image_list = os.listdir(scene.pano_rgb_dir)
        # get the corresponding scan ids
        scan_ids = [img.split('.')[0] for img in image_list]

        for scan_id in tqdm(scan_ids, desc="scan"):
            if cfg.pano_type == 'original':
                rgb_dir = scene.pano_rgb_dir
                depth_dir = scene.pano_depth_dir
                azim_dir = scene.pano_azim_dir
                elev_dir = scene.pano_elev_dir
            elif cfg.pano_type == 'resized':
                rgb_dir = scene.pano_resized_rgb_dir
                depth_dir = scene.pano_resized_depth_dir
                azim_dir = scene.pano_resized_azim_dir
                elev_dir = scene.pano_resized_elev_dir
            else:
                raise ValueError(f'Invalid pano type: {cfg.pano_type}')

            # convert BGR to RGB
            rgb = cv2.cvtColor(cv2.imread(str(rgb_dir / f'{scan_id}.jpg')), cv2.COLOR_BGR2RGB)
            # get the depth map: 16 bit PNG
            depth = cv2.imread(str(depth_dir / f'{scan_id}.png'), cv2.IMREAD_UNCHANGED)
            # get the azimuth map: 16 bit PNG
            azimuth_16bit_scaled = cv2.imread(str(azim_dir / f'{scan_id}.png'), cv2.IMREAD_UNCHANGED)
            # get the elevation map: 16 bit PNG
            elevation_16bit_scaled = cv2.imread(str(elev_dir / f'{scan_id}.png'), cv2.IMREAD_UNCHANGED)

            # NOTE: scale the depth, azim and elev back
            # depth is in meters
            depth = depth.astype(np.float32) / 1000.0
            azimuth = azimuth_16bit_scaled.astype(np.float32) / 1000.0
            elevation = elevation_16bit_scaled.astype(np.float32) / 1000.0

            xyz, rgb = backproject(rgb, depth, azimuth, elevation)

            # scan pose in the mesh space
            scan_pose = scan_poses[int(scan_id)]

            # apply pc to mesh transform to point cloud
            xyz = (xyz @ scan_pose[:3, :3].T) + scan_pose[:3, 3]

            if cfg.get('downsample_factor'):
                xyz = xyz[::cfg.downsample_factor]
                rgb = rgb[::cfg.downsample_factor]

            out_path = save_dir / scene_id / f'{scan_id}.ply'
            out_path.parent.mkdir(parents=True, exist_ok=True)

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(xyz)
            pc.colors = o3d.utility.Vector3dVector(rgb / 255.0)
            o3d.io.write_point_cloud(str(out_path), pc)
            


if __name__ == "__main__":
    main()