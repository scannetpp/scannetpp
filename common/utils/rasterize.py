try:
    # rasterization functions need these
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection
    from pytorch3d.renderer import (
        RasterizationSettings, 
        MeshRasterizer,  
        fisheyecameras
    )
except Exception as e:
    print(f'Error importing pytorch3d: {e}')

try:
    # some functions need only torch
    import torch
    device = torch.device("cuda:0")
except:
    print(f'Error importing torch: {e}')

try:
    import nvdiffrast.torch as dr
except Exception as e:
    print(f'Error importing nvdiffrast: {e}')

import numpy as np
import cv2
from tqdm import tqdm


def undistort_pix_to_face(pix_to_face, undistort_map1, undistort_map2):
    pix_to_face = cv2.remap(pix_to_face, undistort_map1, undistort_map2, 
        interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
    )
    return pix_to_face

def undistort_rasterization(pix_to_face, zbuf, undistort_map1, undistort_map2):
    # apply undistortion to rasterization (nearest neighbor), zbuf (linear) and image (linear)
    pix_to_face = cv2.remap(pix_to_face, undistort_map1, undistort_map2, 
        interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
    )
    if torch.is_tensor(zbuf):
        # convert to np array
        zbuf = zbuf.cpu().numpy()
    # zbuf is tensor
    zbuf = torch.Tensor(cv2.remap(zbuf, undistort_map1, undistort_map2,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    ))

    return pix_to_face, zbuf

def upsample_pix_to_face(pix_to_face, img_height, img_width):
    pix_to_face = torch.nn.functional.interpolate(pix_to_face.unsqueeze(0).unsqueeze(0).float(),
                                                    size=(img_height, img_width), mode='nearest').squeeze().squeeze().long()
    return pix_to_face

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

    radial_params_repeat = torch.Tensor([distort_params])

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
        'fragments': raster_out # everything
    }

    return raster_out_dict

def project_fisheye_single(v_world, pose, intrinsic, distort_params, img_height, img_width):
    """
    Transforms world-space vertices into Nvdiffrast-compatible clip space for one pose.
    """
    # 1. World to Camera Space
    # v_world: [N, 3], pose: [4, 4]
    v_homo = torch.cat([v_world, torch.ones_like(v_world[:, :1])], dim=-1)
    # mat = pose
    v_cam = (v_homo @ pose.T)[:, :3]
    
    x, y, z = v_cam[:, 0], v_cam[:, 1], v_cam[:, 2]
    
    # 2. Apply Fisheye (Kannala-Brandt model)
    r = torch.sqrt(x**2 + y**2) + 1e-10
    theta = torch.atan2(r, z)
    
    t2 = theta**2
    k1, k2, k3, k4 = distort_params[:4]
    theta_d = theta * (1 + k1*t2 + k2*(t2**2) + k3*(t2**3) + k4*(t2**4))
    
    # 3. Project to Image Plane
    scale = theta_d / r
    u = scale * x * intrinsic[0, 0] + intrinsic[0, 2]
    v = scale * y * intrinsic[1, 1] + intrinsic[1, 2]
    
    # 4. Map to Nvdiffrast Clip Space [-1, 1]
    u_clip = (2.0 * u / img_width) - 1.0
    # flip Y to go to nvdiffrast convention
    v_clip = 1.0 - (2.0 * v / img_height)
    
    z_min = z.min()
    z_max = z.max()

    # normalize Z 0-1
    z_clip = (z - z_min) / (z_max - z_min)

    # Inside project_fisheye_single
    mask = (z < 0.01) | (theta > (np.pi / 2.0))

    # set to inf in clip space to avoid rasterization
    u_clip = torch.where(mask, torch.tensor(float('inf'), device=z.device), u_clip)
    v_clip = torch.where(mask, torch.tensor(float('inf'), device=z.device), v_clip)
    z_clip = torch.where(mask, torch.tensor(float('inf'), device=z.device), z_clip)
    
    # Return [1, N, 4] as Nvdiffrast expects a batch dimension
    return torch.stack([u_clip, v_clip, z_clip, torch.ones_like(z)], dim=-1).unsqueeze(0)

def project_batch(v_world, poses, intrinsic, distort_params, img_height, img_width):
    """
    Transforms world-space vertices into Nvdiffrast-compatible clip space for a batch of poses.
    
    Args:
        v_world: [N, 3] world-space vertices
        poses: [B, 4, 4] batch of camera poses
        intrinsic: [3, 3] camera intrinsic matrix
        distort_params: [4] distortion parameters (k1, k2, k3, k4) or None
        img_height: int, image height
        img_width: int, image width
    
    Returns:
        v_clip: [B, N, 4] vertices in clip space for each pose
    """
    device = v_world.device if torch.is_tensor(v_world) else torch.device("cuda:0")
    if not torch.is_tensor(v_world):
        v_world = torch.Tensor(v_world).to(device)
    if not torch.is_tensor(poses):
        poses = torch.Tensor(poses).to(device)
    if not torch.is_tensor(intrinsic):
        intrinsic = torch.Tensor(intrinsic).to(device)
    
    B = poses.shape[0]
    N = v_world.shape[0]
    
    # 1. World to Camera Space for all poses
    # v_world: [N, 3], poses: [B, 4, 4]
    v_homo = torch.cat([v_world, torch.ones(N, 1, device=device)], dim=-1)  # [N, 4]
    # Expand v_homo to [B, N, 4] and poses to [B, 4, 4]
    v_homo_batch = v_homo.unsqueeze(0).expand(B, -1, -1)  # [B, N, 4]
    # Transform: [B, N, 4] @ [B, 4, 4].transpose(-1, -2) -> [B, N, 4]
    v_cam = torch.bmm(v_homo_batch, poses.transpose(-1, -2))[:, :, :3]  # [B, N, 3]
    
    x, y, z = v_cam[:, :, 0], v_cam[:, :, 1], v_cam[:, :, 2]  # each [B, N]
    
    if distort_params is not None:
        # 2. Apply Fisheye (Kannala-Brandt model) for all poses
        r = torch.sqrt(x**2 + y**2) + 1e-10  # [B, N]
        theta = torch.atan2(r, z)  # [B, N]
        
        t2 = theta**2  # [B, N]
        k1, k2, k3, k4 = distort_params[:4]
        theta_d = theta * (1 + k1*t2 + k2*(t2**2) + k3*(t2**3) + k4*(t2**4))  # [B, N]
        
        # 3. Project to Image Plane
        scale = theta_d / r  # [B, N]
    else:
        # Standard pinhole projection: divide by z
        scale = 1.0 / (z + 1e-10)  # [B, N] - add small epsilon to avoid division by zero
        
    u = scale * x * intrinsic[0, 0] + intrinsic[0, 2]  # [B, N]
    v = scale * y * intrinsic[1, 1] + intrinsic[1, 2]  # [B, N]
    
    # 4. Map to Nvdiffrast Clip Space [-1, 1]
    u_clip = (2.0 * u / img_width) - 1.0  # [B, N]
    # flip Y to go to nvdiffrast convention
    v_clip = 1.0 - (2.0 * v / img_height)  # [B, N]
    
    # Normalize Z per pose (0-1 range)
    z_min = z.min(dim=1, keepdim=True)[0]  # [B, 1]
    z_max = z.max(dim=1, keepdim=True)[0]  # [B, 1]
    z_range = z_max - z_min
    z_range = torch.where(z_range < 1e-6, torch.ones_like(z_range), z_range)  # avoid division by zero
    z_clip = (z - z_min) / z_range  # [B, N]
    
    # Mask invalid points (behind camera or outside FOV)
    mask = (z < 0.01)  # [B, N]
    if distort_params is not None:
        mask = mask | (theta > (np.pi / 2.0))
    
    # Set to inf in clip space to avoid rasterization
    inf_val = torch.tensor(float('inf'), device=device)
    u_clip = torch.where(mask, inf_val, u_clip)
    v_clip = torch.where(mask, inf_val, v_clip)
    z_clip = torch.where(mask, inf_val, z_clip)
    
    # Return [B, N, 4] as Nvdiffrast expects
    return torch.stack([u_clip, v_clip, z_clip, torch.ones(B, N, device=device)], dim=-1)

def rasterize_mesh_nvdiffrast(mesh, img_height, img_width, pose, intrinsic, distort_params, img=None):
    """
    Rasterization supporting both single pose and batch of poses.
    
    Args:
        mesh: open3d mesh
        img_height, img_width: int
        pose: [4, 4] single pose or [B, 4, 4] batch of poses with R and t
        intrinsic: [3, 3] camera intrinsic matrix with fx, fy, cx, cy
        distort_params: [4] array of distortion parameters
        img: optional, not used (kept for backward compatibility)
    
    Returns:
        raster_out_dict: dict with 'pix_to_face' of shape [H, W] for single pose or [B, H, W] for batch
    """
    device = torch.device("cuda:0")

    # Convert pose to tensor and check if it's a batch
    if not torch.is_tensor(pose):
        pose = torch.Tensor(pose)
    pose = pose.to(device)
    
    # Check if batch (3D tensor) or single (2D tensor)
    is_batch = len(pose.shape) == 3
    if not is_batch:
        pose = pose.unsqueeze(0)  # [1, 4, 4]
        B = 1
    else:
        B = pose.shape[0]

    v_pos = torch.Tensor(np.array(mesh.vertices)).to(device)
    faces = torch.Tensor(np.array(mesh.triangles)).to(device).to(torch.int32)
    img_dims = (img_height, img_width)

    ctx = dr.RasterizeCudaContext()

    with torch.no_grad():
        if B == 1:
            # Use single pose function for backward compatibility
            v_clip = project_fisheye_single(
                v_pos, pose[0], torch.Tensor(intrinsic).to(device), 
                torch.Tensor(distort_params).to(device), img_height, img_width
            )
        else:
            # Use batch function for efficiency
            v_clip = project_fisheye_batch(
                v_pos, pose, torch.Tensor(intrinsic).to(device), 
                torch.Tensor(distort_params).to(device), img_height, img_width
            )

        # rasterize mesh into image
        # v_clip: [B, N, 4] clipped vertices
        # faces: ntriangles, 3
        # img_dims: height, width
        # grad_db: False, no gradients
        rast, _ = dr.rasterize(
            ctx, v_clip, faces, img_dims, 
            grad_db=False
        )

        # Extract face IDs (0-indexed, -1 for background)
        # (u, v, z/w, triangle_id) -> [B, H, W]
        pixel_to_face = rast[..., 3].int() - 1

    # flip the rows for each image in the batch
    pixel_to_face = torch.flip(pixel_to_face, dims=[1])  # flip along height dimension

    # Remove batch dimension if single pose for backward compatibility
    if not is_batch:
        pixel_to_face = pixel_to_face[0]  # [H, W]

    raster_out_dict = {
        'pix_to_face': pixel_to_face.cpu(),
    }

    return raster_out_dict

def rasterize_mesh_nvdiffrast_large_batch(mesh, img_height, img_width, poses_list, intrinsic, distort_params=None, batch_size=16):
    """
    Memory-efficient batch rasterization for large lists of camera poses.
    Processes poses in smaller batches to avoid GPU memory issues.
    
    Args:
        mesh: open3d mesh
        img_height, img_width: int
        poses_list: list of [4, 4] camera poses (can be numpy arrays or tensors)
        intrinsic: [3, 3] camera intrinsic matrix with fx, fy, cx, cy
        distort_params: [4] array of distortion parameters
        batch_size: int, number of poses to process at once on GPU
    
    Returns:
        raster_out_dict: dict with 'pix_to_face' of shape [N, H, W] where N is len(poses_list)
    """
    device = torch.device("cuda:0")
    
    # Convert all poses to tensors once upfront
    if isinstance(poses_list, (list, tuple)):
        num_poses = len(poses_list)
        # Convert each pose to tensor and stack into [N, 4, 4]
        poses_tensor = torch.stack([
            torch.Tensor(pose) if not torch.is_tensor(pose) else pose 
            for pose in poses_list
        ])  # [N, 4, 4] on CPU
    else:
        # Assume it's a numpy array or tensor
        if not torch.is_tensor(poses_list):
            poses_tensor = torch.Tensor(poses_list)  # [N, 4, 4]
        else:
            poses_tensor = poses_list
        num_poses = poses_tensor.shape[0]
    
    # Pre-allocate CPU memory for all results
    # pix_to_face will be int32 tensor of shape [num_poses, img_height, img_width]
    all_pix_to_face = torch.zeros((num_poses, img_height, img_width), dtype=torch.int32, device='cpu')
    
    # Prepare mesh data once (will be reused for all batches)
    v_pos = torch.Tensor(np.array(mesh.vertices)).to(device)
    faces = torch.Tensor(np.array(mesh.triangles)).to(device).to(torch.int32)
    img_dims = (img_height, img_width)
    
    # Convert intrinsic and distort_params to tensors once
    intrinsic_tensor = torch.Tensor(intrinsic).to(device)
    if distort_params is not None:
        distort_params_tensor = torch.Tensor(distort_params).to(device)
    else:
        distort_params_tensor = None
    
    ctx = dr.RasterizeCudaContext()
    
    # Process in batches
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_poses, batch_size), desc='Batch rasterization'):
            batch_end = min(batch_start + batch_size, num_poses)
            
            # Slice the pre-converted poses tensor and move to GPU
            batch_poses_tensor = poses_tensor[batch_start:batch_end].to(device)  # [B, 4, 4]
            
            # Project vertices to clip space for this batch
            # v_clip shape: [B, N, 4] where B is batch_end - batch_start
            v_clip = project_batch(
                v_pos, batch_poses_tensor, intrinsic_tensor, 
                distort_params_tensor, img_height, img_width
            )
            
            # Rasterize this batch
            rast, _ = dr.rasterize(
                ctx, v_clip, faces, img_dims, 
                grad_db=False
            )
            # rast shape: [B, H, W, 4] where B is batch_end - batch_start
            
            # Extract face IDs (0-indexed, -1 for background)
            pixel_to_face = rast[..., 3].int() - 1  # [B, H, W]
            
            # Flip the rows for each image in the batch
            pixel_to_face = torch.flip(pixel_to_face, dims=[1])  # flip along height dimension
            
            # Transfer to CPU immediately and store in pre-allocated tensor
            all_pix_to_face[batch_start:batch_end] = pixel_to_face.cpu()
            
            # Clear GPU memory for this batch
            del v_clip, rast, pixel_to_face, batch_poses_tensor
            torch.cuda.empty_cache()
    
    raster_out_dict = {
        'pix_to_face': all_pix_to_face,
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
