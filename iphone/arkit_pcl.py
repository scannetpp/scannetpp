"""
The script reads the iphone RGB, depth images, and the corresponding camera poses and intrinsics, and backproject them into a point cloud.s
"""
from typing import List, Tuple, Dict, Optional, Any
import argparse
import json
import time
import os

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from plyfile import PlyData, PlyElement
import open3d as o3d

from common.scene_release import ScannetppScene_Release


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, default="02455b3d20")
    parser.add_argument("--data_root", type=str, required=True, help="The root directory of the data.")
    parser.add_argument("--output", type=str, default="pcl.ply", help="The output filename (PLY format).")
    parser.add_argument("--sample_rate", type=int, default=50, help="Sample rate of the frames.")
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--grid_size", type=float, default=0.05, help="Grid size for voxel downsampling.")
    return parser.parse_args()


def voxel_down_sample(xyz, rgb, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    return xyz, rgb


def outlier_removal(xyz, rgb, nb_points, radius):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd, _ = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    return xyz, rgb


def backproject(
    image: np.ndarray,
    depth: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    use_point_subsample: bool = True,
    point_subsample_rate: int = 4,
    use_voxel_subsample: bool = True,
    voxel_grid_size: float = 0.02,
):
    """Backproject RGB-D image into a point cloud.
    The resolution of RGB and depth are not be the same (the aspect ratio should be the smae).
    Therefore, we need to scale the RGB image and the intrinsic matrix to match the depth.
    """
    scale_factor = depth.shape[1] / image.shape[1]
    image = cv2.resize(image, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Scale the intrinsic matrix
    intrinsic = intrinsic.copy()
    intrinsic[0, 0] *= scale_factor
    intrinsic[1, 1] *= scale_factor
    intrinsic[0, 2] *= scale_factor
    intrinsic[1, 2] *= scale_factor

    yy, xx = np.meshgrid(np.arange(0, depth.shape[0]), np.arange(0, depth.shape[1]), indexing='ij')
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    z = depth[yy, xx]
    valid_mask = np.logical_not((z < min_depth) | (z > max_depth) | np.isnan(z) | np.isinf(z))
    x = xx[valid_mask]
    y = yy[valid_mask]
    uv_one = np.stack([x, y, np.ones_like(x)], axis=0)
    xyz = np.linalg.inv(intrinsic) @ uv_one * z[valid_mask]
    xyz_one = np.concatenate([xyz, np.ones_like(xyz[:1, :])], axis=0)
    xyz_one = camera_to_world @ xyz_one
    xyz = xyz_one[:3, :].T
    rgb = image[y, x]

    xyz, rgb = outlier_removal(xyz, rgb, nb_points=20, radius=voxel_grid_size)

    if use_point_subsample:
        sample_indices = np.random.choice(np.arange(len(xyz)), len(xyz) // point_subsample_rate, replace=False)
        xyz = xyz[sample_indices]
        rgb = rgb[sample_indices]
    if use_voxel_subsample:
        xyz, rgb = voxel_down_sample(xyz, rgb, voxel_size=voxel_grid_size)
    return xyz, rgb


def save_point_cloud(
    filename: str,
    points: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    binary: bool = True,
    verbose: bool = True,
):
    """Save an RGB point cloud as a PLY file.
    Args:
        filename: The output filename.
        points: Nx3 matrix where each row is a point.
        rgb: Nx3 matrix where each row is the RGB value of the corresponding point. If not provided, use gray color for all the points.
        binary: Whether to save the PLY file in binary format.
        verbose: Whether to print the output filename.
    """
    if rgb is None:
        rgb = np.tile(np.array([128], dtype=np.uint8), (points.shape[0], 3))
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points.shape[0]):
            vertices.append(tuple(points[row_idx, :]) + tuple(rgb[row_idx, :]))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # Write
        PlyData([el]).write(filename)
    else:
        # PlyData([el], text=True).write(filename)
        with open(filename, 'w') as f:
            f.write('ply\n'
                    'format ascii 1.0\n'
                    'element vertex %d\n'
                    'property float x\n'
                    'property float y\n'
                    'property float z\n'
                    'property uchar red\n'
                    'property uchar green\n'
                    'property uchar blue\n'
                    'end_header\n' % points.shape[0])
            for row_idx in range(points.shape[0]):
                X, Y, Z = points[row_idx]
                R, G, B = rgb[row_idx]
                f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
    if verbose is True:
        print('Saved point cloud to: %s' % filename)


def main():
    args = parse_args()
    scene = ScannetppScene_Release(args.scene_id, args.data_root)
    iphone_rgb_dir = scene.iphone_rgb_dir
    iphone_depth_dir = scene.iphone_depth_dir

    with open(scene.iphone_pose_intrinsic_imu_path, "r") as f:
        json_data = json.load(f)
    frame_data = [(frame_id, data) for frame_id, data in json_data.items()]
    frame_data.sort()

    all_xyz = []
    all_rgb = []
    for frame_id, data in tqdm(frame_data[::args.sample_rate]):
        camera_to_world = np.array(data["aligned_pose"]).reshape(4, 4)
        intrinsic = np.array(data["intrinsic"]).reshape(3, 3)
        rgb = np.array(Image.open(os.path.join(iphone_rgb_dir, frame_id + ".jpg")), dtype=np.uint8)
        depth = np.array(Image.open(os.path.join(iphone_depth_dir, frame_id + ".png")), dtype=np.float32) / 1000.0

        xyz, rgb = backproject(
            rgb,
            depth,
            camera_to_world,
            intrinsic,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            use_point_subsample=False,
            use_voxel_subsample=True,
            voxel_grid_size=args.grid_size,
        )
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)
    # Voxel downsample again
    all_xyz, all_rgb = outlier_removal(all_xyz, all_rgb, nb_points=10, radius=0.1)
    all_xyz, all_rgb = voxel_down_sample(all_xyz, all_rgb, voxel_size=args.grid_size)

    save_point_cloud(
        filename=args.output,
        points=all_xyz,
        rgb=all_rgb,
        binary=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
