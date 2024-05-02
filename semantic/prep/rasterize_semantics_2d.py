'''
Get 3d semantics onto dslr images by rasterizing the mesh
'''

import argparse
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch

from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer,  
    fisheyecameras
)


from scannetpp.common.scene_release import ScannetppScene_Release
from scannetpp.common.file_io import load_json, load_yaml_munch, read_txt_list

from semantic.utils.colmap_utils import read_cameras_text, read_images_text, camera_to_intrinsic

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def main(args):
    # read cfg 
    cfg = load_yaml_munch(args.config_file)

    scene_list = read_txt_list(cfg.scene_list)

    pth_data_dir = Path(cfg.pth_data_dir)
    semantic_classes = read_txt_list(cfg.semantic_labels_path)

    for scene_id in tqdm(scene_list, desc='scene'):
        # print('Scene:', scene_id)
        scene = ScannetppScene_Release(scene_id, data_root=cfg.data_root)
        # read mesh
        mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path)) 
        # read annotation
        pth_data = torch.load(pth_data_dir / f'{scene_id}.pth')

        # list of dslr images
        dslr_names_all = load_json(scene.dslr_train_test_lists_path)['train']
        # pick every nth dslr image and corresponding camera pose
        dslr_indices = list(range(0, len(dslr_names_all), cfg.dslr_subsample_factor))
        dslr_names = [dslr_names_all[i] for i in dslr_indices]

        # read camera intrinsics
        intrinsics_file = scene.dslr_colmap_dir / 'cameras.txt'
        # there is only 1 camera model, get it
        colmap_camera = list(read_cameras_text(intrinsics_file).values())[0]
        # params [0,1,2,3] give the intrinsic
        intrinsic_mat = camera_to_intrinsic(colmap_camera)
        # rest are the distortion params
        # need 6 radial params
        distort_params = list(colmap_camera.params[4:]) + [0, 0]

        extrinsics_file = scene.dslr_colmap_dir / 'images.txt'
        all_extrinsics = read_images_text(extrinsics_file)
        # get the extrinsics for the selected images into a dict with filename as key
        all_extrinsics_dict = {v.name: v.to_transform_mat() for v in all_extrinsics.values()}

        # create meshes object
        verts = torch.Tensor(np.array(mesh.vertices))
        faces = torch.Tensor(np.array(mesh.triangles))
        meshes = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])
        
        # go through dslr images
        for _, image_name in enumerate(tqdm(dslr_names, desc='image')):
            # draw the camera frustum on the mesh
            camera_pose = all_extrinsics_dict[image_name]

            # read image and get dims
            image_path = scene.dslr_resized_dir / image_name
            image = plt.imread(image_path)
            # get h, w from image
            img_height, img_width = image.shape[:2]

            raster_settings = RasterizationSettings(image_size=(img_height, img_width), 
                                                blur_radius=0.0, 
                                                faces_per_pixel=1,
                                                cull_to_frustum=True)
            rasterizer = MeshRasterizer(
                raster_settings=raster_settings
            )

            camera_lines = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=img_width,
                view_height_px=img_height,
                intrinsic=intrinsic_mat,
                extrinsic=camera_pose
            )
            
            # get 2d-3d mapping of this image by rasterizing, add a dimension in the beginning
            R = torch.Tensor(camera_pose[:3, :3]).unsqueeze(0)
            T = torch.Tensor(camera_pose[:3, 3]).unsqueeze(0)

            # create camera with opencv function
            image_size = torch.Tensor((img_height, img_width))
            image_size_repeat = torch.tile(image_size.reshape(-1, 2), (1, 1))
            intrinsic_repeat = torch.Tensor(intrinsic_mat).unsqueeze(0).expand(1, -1, -1)
            
            opencv_cameras = cameras_from_opencv_projection(
                # N, 3, 3
                R=R,
                # N, 3
                tvec=T,
                # N, 3, 3
                camera_matrix=intrinsic_repeat,
                # N, 2 h,w
                image_size=image_size_repeat
            )

            # apply the same transformation for fisheye cameras 
            # transpose R, then negate 1st and 2nd columns
            fisheye_R = R.mT
            fisheye_R[:, :, :2] *= -1
            # negate x and y in the transformation T
            # negate everything
            fisheye_T = -T
            # negate z back
            fisheye_T[:, -1] *= -1

            # focal, center, radial_params, R, T, use_radial
            fisheye_cameras = fisheyecameras.FishEyeCameras(
                focal_length=opencv_cameras.focal_length,
                principal_point=opencv_cameras.principal_point,
                radial_params=torch.Tensor([distort_params]),
                use_radial=True,
                R=fisheye_R,
                T=fisheye_T,
                image_size=image_size_repeat,
                # need to specify world coordinates, otherwise camera coordinates
                world_coordinates=True
            )  

            # rasterize
            with torch.no_grad():
                raster_out = rasterizer(meshes, cameras=fisheye_cameras.to(device))
                # H, W
                pix_to_face = raster_out.pix_to_face.squeeze().cpu().numpy()

            valid_pix_to_face =  pix_to_face[:, :] != -1
            face_ndx = pix_to_face[valid_pix_to_face]
            mesh_faces_np = np.array(mesh.triangles) 
            faces_in_img = mesh_faces_np[face_ndx]

            # get the set of vertices visible from this image 
            img_verts = np.unique(faces_in_img)

            # keep only obj ids that are visible in this image
            # use the original object IDs from the annotation JSON
            obj_ids = np.unique(pth_data['vtx_instance_anno_id'][img_verts])
            # discard negative obj ids
            obj_ids = obj_ids[obj_ids >= 0]

            obj_bboxes, obj_labels = [], []

            for obj_id in obj_ids:
                # get the vertices occupied by the object
                obj_mask = pth_data['vtx_instance_anno_id'] == obj_id
                # get vertices as numbers
                obj_vert_ndx = np.where(obj_mask)[0]
                # get the vertex coordinates and bbox of this object
                obj_verts = np.array(mesh.vertices)[obj_vert_ndx]
                obj_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(obj_verts))
                # change bbox color to red
                obj_bbox.color = (1, 0, 0)
                # get the object label
                semantic_id = pth_data['vtx_labels'][obj_mask][0]
                obj_label = semantic_classes[semantic_id]
                # store the bbox and label
                obj_bboxes.append(obj_bbox)
                obj_labels.append(obj_label)
            

            pix_inst_ids = np.zeros_like(pix_to_face)
            # get instance ids on pixels
            pix_inst_ids[valid_pix_to_face] = pth_data['vtx_instance_anno_id'][mesh_faces_np[pix_to_face[valid_pix_to_face]][:, 0]]
            # get semantic labels on pixels, initialize to -1
            pix_sem_ids = np.ones_like(pix_to_face) * -1
            # get semantic labels on pixels
            pix_sem_ids[valid_pix_to_face] = pth_data['vtx_labels'][mesh_faces_np[pix_to_face[valid_pix_to_face]][:, 0]]

            # get object 2d bboxes
            obj_bboxes_2d = {}
            
            for obj_id in obj_ids:
                # get a binary image indicating the location of this obj_id
                obj_mask_2d = pix_inst_ids == obj_id
                # get the bounding box of these pixels
                # get the indices of the non-zero pixels
                nonzero_inds = np.nonzero(obj_mask_2d)
                # get the min and max of these indices
                bbox_min = np.min(nonzero_inds, axis=1)
                bbox_max = np.max(nonzero_inds, axis=1)
                # store the bbox as x,y,w,h
                bbox = np.concatenate([bbox_min, bbox_max - bbox_min])
                # store the bbox in a list
                obj_bboxes_2d[int(obj_id)] = bbox.tolist()

            if cfg.viz:
                # display the mesh with obj bboxes
                geoms = []
                geoms += obj_bboxes
                # display the mesh and camera
                geoms.append(camera_lines)
                geoms.append(mesh)

                o3d.visualization.draw_geometries(geoms)

                # read the image and show it
                image_path = scene.dslr_resized_dir / image_name
                image = plt.imread(image_path)
                # show the image
                plt.imshow(image)
                # display 2d object bboxes
                ax = plt.gca()

                for bbox in obj_bboxes_2d.values():
                    # draw the bbox
                    rect = plt.Rectangle((bbox[1], bbox[0]), bbox[3], bbox[2], fill=False, edgecolor='r', linewidth=2)
                    ax.add_patch(rect)

                plt.axis('off')
                plt.show()



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()
    main(args)