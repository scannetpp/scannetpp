'''
rasterize batch of scenes
'''
from copy import copy
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import hydra
from omegaconf import DictConfig
import wandb
import torch
import numpy as np

from tqdm import tqdm
import open3d as o3d

from codetiming import Timer

from pytorch3d.structures import Meshes

from common.utils.colmap import camera_to_intrinsic
from common.file_io import read_txt_list
from common.scene_release import ScannetppScene_Release
from common.utils.colmap import get_camera_images_poses
from common.utils.rasterize import get_fisheye_cameras_batch, get_opencv_cameras_batch, prep_pt3d_inputs, rasterize_mesh

def adjust_intrinsic_matrix(intrinsic, factor):
    # divide fx, fy, cx, cy by factor
    intrinsic /= factor
    intrinsic[2, 2] = 1
    return intrinsic

device = torch.device("cuda:0")

@hydra.main(version_base=None, config_path="../configs", config_name="rasterize")
def main(cfg : DictConfig) -> None:
    print('Config:', cfg)

    if not cfg.no_log:
        wandb.init(project='caption3d_datagen', 
                   group=cfg.wandb_group, config=OmegaConf.to_container(cfg, resolve=True), notes=cfg.wandb_notes)

    # get scene list
    scene_list = read_txt_list(cfg.scene_list_file)
    print('Scenes in list:', len(scene_list))

    if cfg.get('filter_scenes'):
        scene_list = [s for s in scene_list if s in cfg.filter_scenes]
        print('Filtered scenes:', len(scene_list))

    # keep iphone and dslr data separate
    rasterout_dir = Path(cfg.rasterout_dir) / cfg.image_type
    rasterout_dir.mkdir(parents=True, exist_ok=True)

    # go through scenes
    for scene_id in tqdm(scene_list, desc='scene'):
        print(f'Rasterizing: {scene_id}')
        scene = ScannetppScene_Release(scene_id, data_root=cfg.data_root)
        # read mesh
        mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path))
        verts, faces, _ = prep_pt3d_inputs(mesh)

        # get the list of iphone/dslr images and poses
        colmap_camera, image_list, poses, distort_params = get_camera_images_poses(scene, cfg.subsample_factor, cfg.image_type)
        # unreleased data has this prefix, remove it 
        if image_list[0].startswith('video/'):
            image_list = [i.split('video/')[-1] for i in image_list]
        if image_list[0].startswith('dslr/'):
            image_list = [i.split('dslr/')[-1] for i in image_list]

        intrinsic_mat = camera_to_intrinsic(colmap_camera)
        img_height, img_width = colmap_camera.height, colmap_camera.width

        if cfg.image_downsample_factor:
            img_height = img_height // cfg.image_downsample_factor
            img_width = img_width // cfg.image_downsample_factor
            intrinsic_mat = adjust_intrinsic_matrix(intrinsic_mat, cfg.image_downsample_factor)

        # create batches of images and poses with cfg.batch_size in each batch
        batch_start_indices = list(range(0, len(image_list), cfg.batch_size))
        
        (rasterout_dir / scene_id).mkdir(parents=True, exist_ok=True)

        for batch_ndx, batch_start_ndx in enumerate(tqdm(batch_start_indices, desc='batch', leave=False)):
            if cfg.limit_batches == batch_ndx:
                print(f'Done with {cfg.limit_batches} batches, finish')
                break

            batch_image_list = image_list[batch_start_ndx:batch_start_ndx+cfg.batch_size]
            batch_poses = torch.Tensor(np.array(poses[batch_start_ndx:batch_start_ndx+cfg.batch_size]))
            rasterout_paths_batch = [rasterout_dir / scene_id / f'{image_name}.pth' for image_name in batch_image_list]

            if cfg.skip_existing:
                # keep only the ones that don't exist
                rasterout_paths_orig = copy(rasterout_paths_batch)
                rasterout_paths_batch = [p for p in rasterout_paths_batch if not p.exists()]

                exists = set(rasterout_paths_orig) - set(rasterout_paths_batch)

                # if nothing left, skip batch
                if len(rasterout_paths_batch) == 0:
                    print('Skipping batch')
                    continue

                if exists:
                    print(f'Skip existing files: {list(exists)}')

                batch_image_list = [image_name for image_name, p in zip(batch_image_list, rasterout_paths_orig) if not p.exists()]
                batch_poses = torch.Tensor(np.array([pose for pose, p in zip(batch_poses, rasterout_paths_orig) if not p.exists()]))
            
            print(f'Images in batch: {batch_image_list}')

            # create batch of camera poses
            if cfg.image_type == 'dslr':
                # get fisheye cameras with distortion
                cameras_batch = get_fisheye_cameras_batch(batch_poses, img_height, img_width, intrinsic_mat, distort_params)
            elif cfg.image_type == 'iphone':
                # opencv cameras
                cameras_batch = get_opencv_cameras_batch(batch_poses, img_height, img_width, intrinsic_mat)

            bsize = len(batch_image_list)

            # repeat meshes N times for N images in the batch
            meshes_verts = torch.Tensor(np.array([verts for _ in range(bsize)]))
            meshes_faces = torch.Tensor(np.array([faces for _ in range(bsize)]))
            meshes_batch = Meshes(verts=meshes_verts, faces=meshes_faces).to(device)

            with Timer():
                # NOTE!!: with bsize>1, output indices are into the batch mesh, subtract to get back indices into mesh
                raster_out_dict = rasterize_mesh(meshes_batch, img_height, img_width, cameras_batch)
                # save each to file in the correct order
                keep_keys = 'pix_to_face', 'zbuf'
                for sample_ndx, raster_out_path in enumerate(rasterout_paths_batch):
                    print(f'Saving to: {raster_out_path}')
                    sample_data = {k: raster_out_dict[k][sample_ndx].squeeze() for k in keep_keys}
                    # subtract #faces from valid faces (pix_to_face) to get the correct face index
                    pix_to_face = sample_data['pix_to_face']
                    valid_pix_to_face =  pix_to_face[:, :] != -1
                    num_faces = meshes_faces.size(1)
                    pix_to_face[valid_pix_to_face] -= (num_faces * sample_ndx)
                    sample_data['pix_to_face'] = pix_to_face
                    torch.save(sample_data, raster_out_path)
                        

if __name__ == "__main__":
    main()