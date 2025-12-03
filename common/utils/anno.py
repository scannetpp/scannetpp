'''
utils related to 3d semantic+instance annotations
'''
try:
    # not required for all functions
    from torch_scatter import scatter_mean
    import torch
    import open3d as o3d
except ImportError:
    print('torch_scatter not found')
    pass

from collections import defaultdict
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from common.utils.colmap import camera_to_intrinsic, get_camera_images_poses
from common.utils.dslr import compute_undistort_intrinsic, get_undistort_maps
from common.utils.rasterize import undistort_rasterization, upsample_rasterization
from common.file_io import write_json, load_json

def get_top_images_from_visibility(obj_id, visibility_data):
    '''
    obj_id: object id
    visibility_data: visibility data for a scene

    return: sorted list of image names in order of max to min visible vertices
    '''
    images_visibilites = []
    # go through each image
    for i_name in visibility_data['images']:
        # convert obj keys to str always!
        # check if this object is visible in this image
        if str(obj_id) in visibility_data['images'][i_name]['objects'] and 'visible_vertices_frac' in visibility_data['images'][i_name]['objects'][str(obj_id)]:
            # (image, visible frac)
            images_visibilites.append((i_name, visibility_data['images'][i_name]['objects'][str(obj_id)]['visible_vertices_frac']))
    # sort descending by visibility
    images_visibilites.sort(key=lambda x: x[1], reverse=True)
    top_images = [i_name for i_name, _ in images_visibilites]

    return top_images

def viz_sem_ids_2d(pix_sem_ids, semantic_colors, out_path):
    '''
    pix_sem_ids: -1 = no label, 0-N = semantic id
    semantic_colors: color palette, wrap label if it exceeds the palette size
    out_path: save image here
    '''
    viz_sem_img = np.zeros((pix_sem_ids.shape[0], pix_sem_ids.shape[1], 3), dtype=np.uint8)
    for sem_id in np.unique(pix_sem_ids):
        if sem_id == -1:
            continue
        color = semantic_colors[sem_id % len(semantic_colors)]
        viz_sem_img[pix_sem_ids == sem_id] = color

    o3d.io.write_image(str(out_path), o3d.geometry.Image(viz_sem_img))


def get_sem_ids_on_2d(pix_obj_ids, anno, semantic_classes, ignore_label=-1):
    '''
    pix obj ids: obj ids on pixels
    anno: raw annotation with labels
    semantic_classes: list of classes to generate 0-N semantic IDs

    return: same size array as pix_obj_ids with semantic ids, -1 indicates no semantic label
    '''
    unique_obj_ids = np.unique(pix_obj_ids)
    # exclude <= 0
    unique_obj_ids = unique_obj_ids[unique_obj_ids > 0]

    # fill all with ignore label initially, could be +ve or -ve 
    pix_sem_ids = np.ones_like(pix_obj_ids) * ignore_label

    for obj_id in unique_obj_ids:
        obj_label = anno['objects'][obj_id]['label']
        if obj_label in semantic_classes:
            sem_id = semantic_classes.index(obj_label)
        else:
            sem_id = ignore_label
        pix_sem_ids[pix_obj_ids == obj_id] = sem_id

    return pix_sem_ids

def get_visiblity_from_cache(scene, raster_dir, cache_dir, image_type, subsample_factor, undistort_dslr=None, limit_images=None, anno=None):
    cached_path = Path(cache_dir) / f'{scene.scene_id}.json'
    if cached_path.exists():
        print(f'Loading visibility data from cache: {cached_path}')
        visiblity_data = load_json(cached_path)
    else:
        if anno is None:
            anno = load_anno_wrapper(scene)
        visiblity_data = compute_visiblity(scene, anno, raster_dir, image_type=image_type, subsample_factor=subsample_factor, undistort_dslr=undistort_dslr,
                                           limit_images=limit_images)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving visibility data to cache: {cached_path}')
        write_json(cached_path, visiblity_data)

    return visiblity_data

def get_best_views_from_cache(scene, cache_dir, rasterout_dir, image_type, subsample_factor, undistort_dslr):
    cached_path = Path(cache_dir) / f'{scene.scene_id}.pth'
    if cached_path.exists():
        best_view_data = torch.load(cached_path)
    else:
        best_view_data = compute_best_views(scene, rasterout_dir, image_type, subsample_factor, undistort_dslr)
        torch.save(best_view_data, cached_path)
    return best_view_data

def compute_best_views(scene, raster_dir, image_type, subsample_factor, undistort_dslr):
    '''
    order the images based on how many new vertices are visible after adding each one
    first create list of images and images that see a vertex/face, for each vertex/face in the scene
    then pick image that sees the most faces, set it to -1, and repeat
    output = list of image names in order of max to min added visiblity
    '''
    mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path)) 
    faces = np.array(mesh.triangles)

    # get list of images
    # get the list of iphone/dslr images and poses
    colmap_camera, image_list, _, distort_params = get_camera_images_poses(scene, subsample_factor, image_type)
    # keep first 4 elements
    distort_params = distort_params[:4]

    intrinsic = camera_to_intrinsic(colmap_camera)
    img_height, img_width = colmap_camera.height, colmap_camera.width

    if image_type == 'dslr' and undistort_dslr:
        undistort_intrinsic = compute_undistort_intrinsic(intrinsic, img_height, img_width, distort_params)
        undistort_map1, undistort_map2 = get_undistort_maps(intrinsic, distort_params, undistort_intrinsic, img_height, img_width)

    # num faces, num images
    face_seen_ids = np.ones((len(faces), len(image_list)), dtype=np.int32) * -1

    best_views = []

    # for each image
    for image_ndx, image_name in enumerate(tqdm(image_list, desc='image')):
        rasterout_path = Path(raster_dir) / scene.scene_id / f'{image_name}.pth'
        raster_out_dict = torch.load(rasterout_path)

        pix_to_face = raster_out_dict['pix_to_face'].squeeze().cpu()
        zbuf = raster_out_dict['zbuf'].squeeze().cpu()

        rasterized_dims = list(pix_to_face.shape)

        if rasterized_dims != [img_height, img_width]: # upsample
            pix_to_face, zbuf = upsample_rasterization(pix_to_face, zbuf, img_height, img_width)

        pix_to_face = pix_to_face.numpy()

        if image_type == 'dslr' and undistort_dslr: # undistort
            pix_to_face, zbuf = undistort_rasterization(pix_to_face, zbuf.numpy(), undistort_map1, undistort_map2)
                
        valid_pix_to_face =  pix_to_face[:, :] != -1
        face_ndx = pix_to_face[valid_pix_to_face]

        face_seen_ids[face_ndx, image_ndx] = image_ndx

    for _ in tqdm(range(len(image_list)), desc='find_next_image'):
        # find the image that currently sees the most faces
        face_seen_counts = np.sum(face_seen_ids != -1, axis=0)
        best_image_ndx = np.argmax(face_seen_counts)
        best_views.append(image_list[best_image_ndx])
        # set the faces seen by this image to -1, for all images! dont need to see them again
        faces_seen = face_seen_ids[:, best_image_ndx] != -1 # faces seen by the current best image
        face_seen_ids[faces_seen, :] = -1

    return best_views


def compute_visiblity(scene, anno, raster_dir, image_type, subsample_factor, undistort_dslr=True, limit_images=None):
    '''
    dict for 1 scene

    objects
        obj_id
            num_total_vertices: total number of vertices of this object
    images:
        image_name 
            objects: obj_id
                bbox_2d: x,y,w,h
                num_visible_vertices: number of vertices of this object visible in the image
                visible_vertices_frac: fraction of vertices of this object visibile in the image
                num_visible_pixels: number of pixels that this obj covers in the image
                visible_pixels_frac: frac of pixels that this obj covers in the image
                zbuf_min: min zbuf value of the object in the image
                zbuf_max: max zbuf value of the object in the image
    '''
    print(f'Computing visibility for {scene.scene_id}')

    visibility_data = {
        'scene_id': scene.scene_id,
        'objects': defaultdict(dict),
        'images': defaultdict(dict)
    }

    mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path)) 
    faces = np.array(mesh.triangles)

    # get list of images
    # get the list of iphone/dslr images and poses
    colmap_camera, image_list, _, distort_params = get_camera_images_poses(scene, subsample_factor, image_type)
    print(f'Num images for visibility check: {len(image_list)}')
    if limit_images is not None:
        image_list = image_list[:limit_images]
        print(f'Limiting viz check to {limit_images} images')
    # keep first 4 elements
    distort_params = distort_params[:4]

    intrinsic = camera_to_intrinsic(colmap_camera)
    img_height, img_width = colmap_camera.height, colmap_camera.width

    if image_type == 'dslr' and undistort_dslr:
        undistort_intrinsic = compute_undistort_intrinsic(intrinsic, img_height, img_width, distort_params)
        undistort_map1, undistort_map2 = get_undistort_maps(intrinsic, distort_params, undistort_intrinsic, img_height, img_width)

    # for each image
    for image_name in tqdm(image_list, desc='image'):
        visibility_data['images'][image_name]['objects'] = defaultdict(dict)

        # load rasterization
        rasterout_path = raster_dir / scene.scene_id / f'{image_name}.pth'
        raster_out_dict = torch.load(rasterout_path)

        pix_to_face = raster_out_dict['pix_to_face'].squeeze().cpu()
        zbuf = raster_out_dict['zbuf'].squeeze().cpu()

        rasterized_dims = list(pix_to_face.shape)

        # upsample rasterization to the image dims
        if rasterized_dims != [img_height, img_width]: # upsample
            pix_to_face, zbuf = upsample_rasterization(pix_to_face, zbuf, img_height, img_width)

        pix_to_face = pix_to_face.numpy()

        if image_type == 'dslr' and undistort_dslr: # undistort
            pix_to_face, zbuf = undistort_rasterization(pix_to_face, zbuf, undistort_map1, undistort_map2)
                
        valid_pix_to_face =  pix_to_face[:, :] != -1
        face_ndx = pix_to_face[valid_pix_to_face]
        # get obj ids on 2d image
        pix_obj_ids = get_vtx_prop_on_2d(pix_to_face, anno['vertex_obj_ids'], mesh)
        # get objid -> bbox x,y,w,h 
        bboxes_2d = get_bboxes_2d(pix_obj_ids)

        faces_in_img = faces[face_ndx]
        # get the set of vertices visible from this image 
        img_verts = np.unique(faces_in_img)

        # get all required stats per object visible in the image
        for (obj_id, obj_bbox) in tqdm(bboxes_2d.items(), desc='obj', leave=False):
            if obj_id <= 0:
                continue

            obj_mask_3d = anno['vertex_obj_ids'] == obj_id
            obj_verts_ndx = np.where(obj_mask_3d)[0] # indices of vertices in this object
            if len(obj_verts_ndx) == 0:
                continue
            # store total #vertices of object
            visibility_data['objects'][str(obj_id)]['num_total_vertices'] = len(obj_verts_ndx)

            # faces in this image -> vertices in this image
            faces_in_img = faces[face_ndx]
            img_verts = np.unique(faces_in_img)
            # obj verts in this image
            intersection = np.intersect1d(obj_verts_ndx, img_verts)
            # frac of obj vertices visible in this image
            visible_frac = len(intersection) / len(obj_verts_ndx)

            obj_pixel_mask = pix_obj_ids == obj_id
            num_obj_pixels = np.sum(obj_pixel_mask)

            obj_zbuf = zbuf.squeeze()[obj_pixel_mask.squeeze()]
            # keep only the >= 0 values
            obj_zbuf = obj_zbuf[obj_zbuf >= 0]

            zbuf_min = obj_zbuf.min().item() if len(obj_zbuf) > 0 else -1
            zbuf_max = obj_zbuf.max().item() if len(obj_zbuf) > 0 else -1

            # string keys for json
            visibility_data['images'][image_name]['objects'][str(obj_id)] = {
                'bbox_2d': list(obj_bbox),
                'num_visible_vertices': len(intersection),
                'visible_vertices_frac': float(visible_frac),
                'num_visible_pixels': int(num_obj_pixels),
                'visible_pixels_frac': float(num_obj_pixels / (img_height * img_width)),
                'zbuf_min': float(zbuf_min),
                'zbuf_max': float(zbuf_max)
            }

    return visibility_data


def load_anno_wrapper(scene):
    anno = load_annotation(scene.scan_anno_json_path, bboxes_only=True, 
                            segments_path=scene.scan_mesh_segs_path, 
                            return_vertex_obj_ids=True)
    
    return anno

def get_bboxes_2d(pix_to_objid):
    '''
    pix_to_objid: 2d array of obj ids for each pixel
    '''
    obj_ids = np.unique(pix_to_objid)
    # discard objid 0 and negative
    obj_ids = obj_ids[obj_ids > 0]

    obj_bboxes_2d = {}

    for obj_id in obj_ids:
        # get a binary image indicating the location of this obj_id
        obj_mask_2d = pix_to_objid == obj_id
        # get the bounding box of these pixels
        # get the indices of the non-zero pixels
        nonzero_inds = np.nonzero(obj_mask_2d)
        # get the min and max of these indices
        bbox_min = np.min(nonzero_inds, axis=1)
        bbox_max = np.max(nonzero_inds, axis=1)
        # store the bbox as x,y,w,h
        bbox = np.concatenate([bbox_min, bbox_max - bbox_min])
        # store the bbox in a list
        obj_bboxes_2d[obj_id] = bbox.tolist()

    return obj_bboxes_2d

def load_annotation(anno_path, bboxes_only=False, segments_path=None, return_vertex_obj_ids=False):
    '''
    load annotation from json file
    '''
    with open(anno_path) as f:
        anno = json.load(f)

    ret_dict = {}

    # json['segGroups'] has id, objectId, segments, label, obb
    if bboxes_only:
        # everything except segments, gets the 3d bbox
        ret_dict['objects'] = {segGroup['objectId']: {k: v for k, v in segGroup.items() if k != 'segments'} for segGroup in anno['segGroups']}
    else:   
        # everything
        ret_dict['objects'] = {segGroup['objectId']: {k: v for k, v in segGroup.items()} for segGroup in anno['segGroups']}

    if return_vertex_obj_ids:
        # load the segments json file
        with open(segments_path) as f:
            segments = json.load(f)['segIndices']
        # empty array to hold obj ids on vertices
        vertex_obj_ids = np.zeros(len(segments), dtype=np.int32)

        # get object ids for each vertex
        for obj in anno['segGroups']:
            # segment ids for this object
            seg_ids = np.array(obj['segments'], dtype=np.int32)
            # assume each segment is a vertex -> use segment id as vertex id
            vertex_obj_ids[seg_ids] = obj['objectId']

        ret_dict['vertex_obj_ids'] = vertex_obj_ids
        
    return ret_dict

def get_pixel_props_on_vtx(pix_to_face, pix_prop, mesh):
    '''
    pix_to_face: output of rasterization, h,w (292,438 for 1/4 downsampled`)
    pix_prop: some property on the pixels: h,w,dim
    mesh: open3d mesh

    allow storing n-dim features

    return:
        num_vertices,dim tensor with features
        vtx_ndx = the indices of the vertices that have features
    '''
    valid_pix_to_face =  pix_to_face[:, :] != -1

    mesh_faces_np = np.array(mesh.triangles) # F,3
    # TODO: interpolate features onto vertices using bary coords + aggregate from multiple faces
    face_indices = pix_to_face[valid_pix_to_face] # faces that are mapped to
    vtx_ndx = mesh_faces_np[face_indices][:, 0] # corresponding vtx indices
    # each vertex gets the average of features mapped onto it, use scatter mean
    face_features = pix_prop[valid_pix_to_face] # features on faces

    all_face_features = torch.zeros(len(mesh_faces_np), face_features.shape[1], device=pix_prop.device)
    all_face_features[face_indices] = face_features

    vtx_props = scatter_mean(face_features, torch.LongTensor(vtx_ndx).to(pix_prop.device), dim=0, dim_size=len(mesh.vertices))
    
    # unique vertices that got features now
    unique_vtx_ndx = np.unique(vtx_ndx)

    return vtx_props, unique_vtx_ndx


def get_vtx_prop_on_2d(pix_to_face, vtx_prop, mesh):
    '''
    pix_to_face: output of rasterization
    vtx_prop: some property on the vertices
    mesh: open3d mesh

    allow storing n-dim features
    '''
    valid_pix_to_face =  pix_to_face[:, :] != -1

    mesh_faces_np = np.array(mesh.triangles)

    # pix to obj id
    pix_vtx_prop = np.zeros_like(pix_to_face)
    # pick the first vertex of each face
    pix_vtx_prop[valid_pix_to_face] = vtx_prop[mesh_faces_np[pix_to_face[valid_pix_to_face]][:, 0]]

    return pix_vtx_prop.squeeze()
