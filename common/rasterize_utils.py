

from scannetpp.common.utils.colmap import read_cameras_text, read_images_text
import torch
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer,  
)

def get_vtx_prop_on_2d(pix_to_face, vtx_prop, mesh):
    '''
    pix_to_face: output of rasterization
    vtx_prop: some property on the vertices
    mesh: open3d mesh

    TODO: supports only scalar values on pixels
    allow storing n-dim features

    output: -1 -> no property, otherwise the property from 3D onto 2D
    '''
    valid_pix_to_face =  pix_to_face[:, :] != -1

    mesh_faces_np = np.array(mesh.triangles)

    # pix to obj id
    pix_vtx_prop = np.ones_like(pix_to_face, dtype=vtx_prop.dtype) * -1
    pix_vtx_prop[valid_pix_to_face] = vtx_prop[mesh_faces_np[pix_to_face[valid_pix_to_face]][:, 0]]

    return pix_vtx_prop.squeeze()

def get_opencv_cameras(pose, img_height, img_width, intrinsic_mat):
    # get 2d-3d mapping of this image by rasterizing, add a dimension in the beginning
    R = torch.Tensor(pose[:3, :3]).unsqueeze(0)
    T = torch.Tensor(pose[:3, 3]).unsqueeze(0)

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

    return opencv_cameras

def prep_pt3d_inputs(mesh, device='cuda'):
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.triangles)
    meshes = Meshes(verts=[torch.Tensor(verts).to(device)], faces=[torch.Tensor(faces).to(device)])

    return verts, faces, meshes

def get_camera_images_poses(scene, image_type, subsample_factor=1):
    # TODO: unify iphone and dslr code

    if image_type == 'iphone':
        # read camera intrinsics
        intrinsics_file = scene.iphone_colmap_dir / 'cameras.txt'
        # there is only 1 camera model, get it
        colmap_camera = list(read_cameras_text(intrinsics_file).values())[0]
        # params [0,1,2,3] give the intrinsic
        extrinsics_file = scene.iphone_colmap_dir / 'images.txt'
        # dict with key 0,1,2, value has the same "id"
        all_extrinsics = read_images_text(extrinsics_file)
        # sort by id and get list of objects
        all_extrinsics = [all_extrinsics[k] for k in sorted(all_extrinsics.keys())]
        # subsample with cfg.subsample_factor
        subsampled_extrinsics = all_extrinsics[::subsample_factor]
        image_names = [e.name for e in subsampled_extrinsics]
        poses = [e.to_transform_mat() for e in subsampled_extrinsics]
    else:
        raise NotImplementedError

    return colmap_camera, image_names, poses


def rasterize_mesh_and_cache(meshes, img_height, img_width, opencv_cameras, rasterout_path):
    if rasterout_path.exists():
        raster_out_dict = torch.load(rasterout_path)
    else:
        # rasterize mesh onto image and get mapping
        raster_out_dict = rasterize_mesh(meshes, img_height, img_width, opencv_cameras)
        torch.save(raster_out_dict, rasterout_path)

    return raster_out_dict

def rasterize_mesh(meshes, img_height, img_width, cameras, device='cuda'):
    raster_settings = RasterizationSettings(image_size=(img_height, img_width), 
                                                    blur_radius=0.0, 
                                                    faces_per_pixel=1,
                                                    cull_to_frustum=True)
    rasterizer = MeshRasterizer(
        raster_settings=raster_settings
    )

    with torch.no_grad():
        raster_out = rasterizer(meshes, cameras=cameras.to(device))

    raster_out_dict = {
        'pix_to_face': raster_out.pix_to_face.cpu(),
        'zbuf': raster_out.zbuf.cpu(),
        'bary_coords': raster_out.bary_coords.cpu(),
        'dists': raster_out.dists.cpu(),
    }

    return raster_out_dict
