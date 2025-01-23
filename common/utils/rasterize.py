try:
    # rasterization functions need these
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection
    from pytorch3d.renderer import (
        RasterizationSettings, 
        MeshRasterizer,  
        fisheyecameras
    )
except:
    pass
try:
    # some functions need only torch
    import torch
    device = torch.device("cuda:0")
except:
    pass
import numpy as np
import cv2


def undistort_rasterization(pix_to_face, zbuf, undistort_map1, undistort_map2):
    # apply undistortion to rasterization (nearest neighbor), zbuf (linear) and image (linear)
    pix_to_face = cv2.remap(pix_to_face, undistort_map1, undistort_map2, 
        interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
    )
    # zbuf is tensor
    zbuf = torch.Tensor(cv2.remap(zbuf, undistort_map1, undistort_map2,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    ))

    return pix_to_face, zbuf

def upsample_rasterization(pix_to_face, zbuf, img_height, img_width):
    # upsample pixtoface and zbuf
    pix_to_face = torch.nn.functional.interpolate(pix_to_face.unsqueeze(0).unsqueeze(0).float(),
                                                    size=(img_height, img_width), mode='nearest').squeeze().squeeze().long()
    zbuf = torch.nn.functional.interpolate(zbuf.unsqueeze(0).unsqueeze(0).float(),
                                                    size=(img_height, img_width), mode='nearest').squeeze().squeeze()

    return pix_to_face, zbuf

def get_opencv_cameras_batch(poses, img_height, img_width, intrinsic_mat):
    R = torch.Tensor(poses[:, :3, :3])
    T = torch.Tensor(poses[:, :3, 3])

    bsize = R.shape[0]

    # create camera with opencv function
    image_size = torch.Tensor((img_height, img_width))
    image_size_repeat = torch.tile(image_size.reshape(-1, 2), (bsize, 1))
    intrinsic_repeat = torch.Tensor(intrinsic_mat).unsqueeze(0).expand(bsize, -1, -1)
    
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

def get_fisheye_cameras(pose, img_width, img_height, intrinsic_mat, distort_params):
    opencv_cameras = get_opencv_cameras(pose, img_width, img_height, intrinsic_mat)

    # get 2d-3d mapping of this image by rasterizing, add a dimension in the beginning
    R = torch.Tensor(pose[:3, :3]).unsqueeze(0)
    T = torch.Tensor(pose[:3, 3]).unsqueeze(0)

    # create camera with opencv function
    image_size = torch.Tensor((img_height, img_width))
    image_size_repeat = torch.tile(image_size.reshape(-1, 2), (1, 1))

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

    return fisheye_cameras

def get_fisheye_cameras_batch(poses, img_width, img_height, intrinsic_mat, distort_params):
    opencv_cameras = get_opencv_cameras_batch(poses, img_width, img_height, intrinsic_mat)

    R = torch.Tensor(poses[:, :3, :3])
    T = torch.Tensor(poses[:, :3, 3])
    bsize = R.shape[0]

    # create camera with opencv function
    image_size = torch.Tensor((img_height, img_width))
    image_size_repeat = torch.tile(image_size.reshape(-1, 2), (bsize, 1))

    # apply the same transformation for fisheye cameras 
    # transpose R, then negate 1st and 2nd columns
    fisheye_R = R.mT
    fisheye_R[:, :, :2] *= -1
    # negate x and y in the transformation T
    # negate everything
    fisheye_T = -T
    # negate z back
    fisheye_T[:, -1] *= -1

    radial_params_repeat = torch.Tensor([distort_params]) #.expand(bsize, -1)

    # focal, center, radial_params, R, T, use_radial
    fisheye_cameras = fisheyecameras.FishEyeCameras(
        # N,1
        focal_length=opencv_cameras.focal_length,
        # N,2
        principal_point=opencv_cameras.principal_point,
        # N, num radial
        radial_params=radial_params_repeat,
        use_radial=True,
        # N, 3, 3
        R=fisheye_R,
        # N, 3
        T=fisheye_T,
        # N, 2 h,w
        image_size=image_size_repeat,
        # need to specify world coordinates, otherwise camera coordinates
        world_coordinates=True
    )  

    return fisheye_cameras

def rasterize_mesh(meshes, img_height, img_width, cameras):
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

def prep_pt3d_inputs(mesh):
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.triangles)
    meshes = Meshes(verts=[torch.Tensor(verts).to(device)], faces=[torch.Tensor(faces).to(device)])

    return verts, faces, meshes

def rasterize_mesh_and_cache(meshes, img_height, img_width, opencv_cameras, rasterout_path):
    if rasterout_path.exists():
        print('Loading rasterization from cache:', rasterout_path)
        raster_out_dict = torch.load(rasterout_path)
    else:
        # rasterize mesh onto image and get mapping
        raster_out_dict = rasterize_mesh(meshes, img_height, img_width, opencv_cameras)
        print('Saving rasterization to cache:', rasterout_path)
        # keep only pix_to_face and zbuf
        raster_out_dict_to_save = {k: v for k, v in raster_out_dict.items() if k in ['pix_to_face', 'zbuf']}
        torch.save(raster_out_dict_to_save, rasterout_path)

    return raster_out_dict
