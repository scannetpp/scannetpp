'''
Get 3D semantics onto 2D images using precomputed rasterization
'''

from common.utils.image import get_img_crop, load_image, save_img, viz_ids, viz_obj_ids_txt
from omegaconf import DictConfig
from pathlib import Path
import hydra
from omegaconf import DictConfig

from tqdm import tqdm
import open3d as o3d
import torch
import numpy as np
import cv2

from common.utils.dslr import compute_undistort_intrinsic
from common.utils.colmap import get_camera_images_poses, camera_to_intrinsic
from common.utils.anno import get_bboxes_2d, get_sem_ids_on_2d, get_visiblity_from_cache, get_vtx_prop_on_2d, load_anno_wrapper, viz_sem_ids_2d
from common.file_io import read_txt_list
from common.scene_release import ScannetppScene_Release

from common.utils.rasterize import get_fisheye_cameras_batch, get_opencv_cameras_batch, prep_pt3d_inputs, rasterize_mesh


@hydra.main(version_base=None, config_path="../configs", config_name='semantics_2d')
def main(cfg : DictConfig) -> None:
    print('Config:', cfg)

    # get scene list
    scene_list = read_txt_list(cfg.scene_list_file)
    print('Scenes in list:', len(scene_list))

    if cfg.get('filter_scenes'):
        scene_list = [s for s in scene_list if s in cfg.filter_scenes]
        print('Filtered scenes:', len(scene_list))
    if cfg.get('exclude_scenes'):
        scene_list = [s for s in scene_list if s not in cfg.exclude_scenes]
        print('After excluding scenes:', len(scene_list))

    # root + runname + savedir
    save_dir = Path(cfg.save_dir_root) / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    print('Save to dir:', save_dir)

    semantics_dir = save_dir / 'semantics'
    semantics_viz_dir = save_dir / 'semantics_viz'
    img_crop_dir =  save_dir / 'img_crops'
    img_crop_nobg_dir = save_dir / 'img_crops_nobg'
    img_crop_mask_dir = save_dir / 'img_crops_mask'
    bbox_img_dir =  save_dir / 'img_bbox'
    viz_obj_ids_dir = save_dir / 'viz_obj_ids'
    viz_obj_ids_txt_dir = save_dir / 'viz_obj_ids_txt'
    objid_gt_2d_dir = save_dir / 'obj_ids'
    undistorted_dir = save_dir / 'undistorted'
    obj_pcs_dir = save_dir / 'obj_pcs'
    obj_meshes_dir = save_dir / 'obj_meshes'
    

    for dir in [img_crop_dir, img_crop_nobg_dir, img_crop_mask_dir, bbox_img_dir, viz_obj_ids_dir, viz_obj_ids_txt_dir, 
        objid_gt_2d_dir, undistorted_dir, obj_pcs_dir, obj_meshes_dir, semantics_dir, 
        semantics_viz_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    rasterout_dir = Path(cfg.rasterout_dir) / cfg.image_type

    if cfg.save_semantic_gt_2d:
        semantic_classes = read_txt_list(cfg.semantic_classes_file)
        semantic_colors = np.loadtxt(cfg.semantic_2d_palette_path, dtype=np.uint8)
        print(f'Num semantic classes: {len(semantic_classes)}')

    # go through scenes
    for scene_id in tqdm(scene_list, desc='scene'):
        print(f'Running on scene: {scene_id}')
        scene = ScannetppScene_Release(scene_id, data_root=cfg.data_root)
        # get object ids on the mesh vertices
        anno = load_anno_wrapper(scene)

        if cfg.get('filter_obj_ids') and cfg.get('filter_objs_global'):
            # anno['objects'] only for the given ids
            anno['objects'] = {obj_id: anno['objects'][obj_id] for obj_id in cfg.filter_obj_ids}
            valid_vtx_mask = np.isin(anno['vertex_obj_ids'], cfg.filter_obj_ids)
            anno['vertex_obj_ids'][~valid_vtx_mask] = -1

        if cfg.check_visibility:
            # create visibility cache to pick topk images where an object is visible
            visibility_data = get_visiblity_from_cache(scene, rasterout_dir, cfg.visiblity_cache_dir, 
                                                       cfg.image_type, 
                                                       cfg.subsample_factor, 
                                                       cfg.undistort_dslr, 
                                                       limit_images=cfg.limit_images,
                                                       anno=anno)
        if cfg.create_visiblity_cache_only:
            print(f'Created visibility cache for {scene_id}')
            continue

        vtx_obj_ids = anno['vertex_obj_ids']
        # read mesh
        mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path)) 

        obj_ids = np.unique(vtx_obj_ids)
        # remove <= 0
        obj_ids = sorted(obj_ids[obj_ids > 0])

        obj_id_locations = {obj_id: anno['objects'][obj_id]['obb']['centroid'] for obj_id in obj_ids}
        obj_id_dims = {obj_id: anno['objects'][obj_id]['obb']['axesLengths'] for obj_id in obj_ids}

        # get the list of iphone/dslr images and poses
        # NOTE: should be the same as during rasterization
        if cfg.get('filter_images'):
            print(f'>>>>> Filtering images, subsample_factor is ignored')
            subsample_factor = 1
        else:
            subsample_factor = cfg.subsample_factor
        colmap_camera, image_list, poses, distort_params_orig = get_camera_images_poses(scene, subsample_factor, cfg.image_type)
        print('Num images:', len(image_list))
        # keep first 4 elements
        distort_params = distort_params_orig[:4]

        intrinsic = camera_to_intrinsic(colmap_camera)
        img_height, img_width = colmap_camera.height, colmap_camera.width

        undistort_map1, undistort_map2 = None, None
        if cfg.image_type == 'dslr' and cfg.undistort_dslr:
            undistort_intrinsic = compute_undistort_intrinsic(intrinsic, img_height, img_width, distort_params)
            undistort_map1, undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                intrinsic, distort_params, np.eye(3), undistort_intrinsic, (img_width, img_height), cv2.CV_32FC1
            )

        if cfg.get('limit_images'):
            print(f'Limiting to {cfg.limit_images} images')
            image_list = image_list[:cfg.limit_images]

        # go through list of images
        for image_ndx, image_name in enumerate(tqdm(image_list, desc='image', leave=False)):
            if cfg.get('filter_images') and image_name not in cfg.filter_images:
                continue
            if cfg.image_type == 'iphone':
                image_dir = scene.iphone_rgb_dir
            elif cfg.image_type == 'dslr':
                image_dir = scene.dslr_resized_dir
            # load the image H, W, 3
            img_path = str(image_dir / image_name)
            if not Path(img_path).exists():
                print(f'Image not found: {img_path}, skipping')
                continue
            
            try:
                print(f'Loading image: {img_path}')
                img = load_image(img_path) 
            except:
                print(f'Error loading image: {img_path}, skipping')
                continue

            rasterout_path = rasterout_dir / scene_id / f'{image_name}.pth'
            if not rasterout_path.is_file():
                # might not have rasterization, esp when specifying filter_images
                print(f'Rasterization not found for {image_name}, rasterizing..')
                _, _, meshes_batch = prep_pt3d_inputs(mesh)
                # create batch of camera poses
                if cfg.image_type == 'dslr':
                    # add batch dimension
                    poses_batch = torch.Tensor(poses[image_ndx]).unsqueeze(0)
                    # get fisheye cameras with distortion
                    cameras_batch = get_fisheye_cameras_batch(poses_batch, img_height, img_width, intrinsic, distort_params_orig)
                    raster_out_dict = rasterize_mesh(meshes_batch, img_height, img_width, cameras_batch)
                else:
                    raise NotImplementedError(f'Image type {cfg.image_type} not supported')
            else:
                raster_out_dict = torch.load(rasterout_path)

            # if dimensions dont match, raster is from downsampled image
            # upsample using nearest neighbor
            pix_to_face = raster_out_dict['pix_to_face'].squeeze().cpu()
            rasterized_dims = list(pix_to_face.shape)

            if rasterized_dims != [img_height, img_width]:
                # upsample pixtoface and zbuf
                pix_to_face = torch.nn.functional.interpolate(pix_to_face.unsqueeze(0).unsqueeze(0).float(),
                                                              size=(img_height, img_width), mode='nearest').squeeze().squeeze().long()
            pix_to_face = pix_to_face.numpy()

            if undistort_map1 is not None and undistort_map2 is not None:
                # apply undistortion to rasterization (nearest neighbor), zbuf (linear) and image (linear)
                pix_to_face = cv2.remap(pix_to_face, undistort_map1, undistort_map2, 
                    interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101,
                )
                # img is np
                img = cv2.remap(img, undistort_map1, undistort_map2,
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
            # get object IDs on image
            try:
                pix_obj_ids = get_vtx_prop_on_2d(pix_to_face, vtx_obj_ids, mesh)
            except IndexError: # something wrong with the rasterization
                print(f'Rasterization error in {scene_id}/{image_name}, skipping')
                continue

            if cfg.get('filter_obj_ids'):
                print(f'Filtering obj ids: {cfg.filter_obj_ids}')
                pix_obj_ids = np.where(np.isin(pix_obj_ids, cfg.filter_obj_ids), pix_obj_ids, -1)

            if cfg.viz_obj_ids: # save viz to file
                out_path = viz_obj_ids_dir / scene_id / f'{image_name}.png'
                viz_ids(img, pix_obj_ids, out_path)

            if cfg.viz_obj_ids_txt:
                # viz obj ids in different colors and write the obj id in the center of the bbox of the object
                out_path = viz_obj_ids_txt_dir / scene_id / f'{image_name}.png'
                viz_obj_ids_txt(img, pix_obj_ids, out_path)

            if cfg.save_objid_gt_2d: # save obj ids to pth file
                out_path = objid_gt_2d_dir / scene_id / f'{image_name}.pth'
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pix_obj_ids, out_path)

            if cfg.save_undistorted_images:
                out_path = undistorted_dir / scene_id / f'{image_name}.jpg'
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_img(img, out_path)

            # create semantics GT and of semantic ids on vertices, -1 = no semantic label
            if cfg.save_semantic_gt_2d:
                out_path = semantics_dir / scene_id / f'{image_name}.png'
                if cfg.skip_existing_semantic_gt_2d and out_path.exists():
                    print(f'File exists: {out_path}, skipping')
                    continue
                # use 255 so that it can be saved as a PNG!
                pix_sem_ids = get_sem_ids_on_2d(pix_obj_ids, anno, semantic_classes, ignore_label=255)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # save to png file, smaller
                print(f'Saving 2d semantic anno to {out_path}')
                cv2.imwrite(str(out_path), pix_sem_ids)

                if cfg.viz_semantic_gt_2d:
                    out_path = semantics_viz_dir / scene_id / f'{image_name}_viz.png'
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f'Saving 2d semantic viz to {out_path}')
                    viz_sem_ids_2d(pix_sem_ids, semantic_colors, out_path)
                continue # do only semantics, nothing else

            # get objid -> bbox x,y,w,h after upsampling rasterization, all the objs in this image
            bboxes_2d = get_bboxes_2d(pix_obj_ids)

            if cfg.process_each_object:
                # go through each object that has a bbox 
                for _, (obj_id, obj_bbox) in enumerate(tqdm(bboxes_2d.items(), desc='obj', leave=False)):
                    if obj_id == 0:
                        continue

                    if cfg.check_visibility:
                        # no viz data for this image -> skip
                        if image_name not in visibility_data['images']:
                            print(f'No viz data for {image_name}, skipping')
                            continue

                        # have viz data for this object, otherwise dont process this object
                        if str(obj_id) not in visibility_data['images'][image_name]['objects']:
                            print(f'No viz data for {obj_id} in {image_name}, skipping')
                            continue

                        # enough of the object is seen
                        if visibility_data['images'][image_name]['objects'][str(obj_id)].get('visible_vertices_frac', 0) < cfg.obj_visible_thresh:
                            continue
                        
                        # check if obj occupies enough % of the image
                        if visibility_data['images'][image_name]['objects'][str(obj_id)].get('visible_pixels_frac', 0) < cfg.obj_pixel_thresh:
                            continue

                        if visibility_data['images'][image_name]['objects'][str(obj_id)].get('zbuf_min', 9999) > cfg.obj_dist_thresh:
                            # object is too far away from camera
                            continue
                        
                        if cfg.visibility_topk is not None:
                            images_visibilites = []
                            for i_name in visibility_data['images']:
                                # convert obj keys to str always!
                                if str(obj_id) in visibility_data['images'][i_name]['objects'] and 'visible_vertices_frac' in visibility_data['images'][i_name]['objects'][str(obj_id)]:
                                    images_visibilites.append((i_name, visibility_data['images'][i_name]['objects'][str(obj_id)]['visible_vertices_frac']))
                            # sort descending by visibility
                            images_visibilites.sort(key=lambda x: x[1], reverse=True)
                            top_images = [i_name for i_name, _ in images_visibilites][:cfg.visibility_topk]
                            # dont consider this object in this image
                            if image_name not in top_images: 
                                continue

                    img_crop = get_img_crop(img, obj_bbox, cfg.bbox_expand_factor, expand_bbox=True)
                    
                    # crop the object from the image
                    if cfg.save_obj_crop:
                        # TODO: save jpgs, use less space!
                        img_crop_path = img_crop_dir / scene_id / f'{image_name}_{obj_id}.png'
                        save_img(img_crop, img_crop_path)

                    if cfg.save_obj_crop_nobg or cfg.save_obj_crop_mask:
                        # crop pix_obj_ids to the same size
                        pix_obj_ids_crop = get_img_crop(pix_obj_ids, obj_bbox, cfg.bbox_expand_factor, expand_bbox=True)

                    # save image crop without background
                    if cfg.save_obj_crop_nobg:
                        # make copy of img_crop
                        img_crop_nobg = img_crop.copy()
                        # set background to black in regions not the current object
                        img_crop_nobg[pix_obj_ids_crop != obj_id] = 0
                        # save
                        img_crop_nobg_path = img_crop_nobg_dir / scene_id / f'{image_name}_{obj_id}.png'
                        save_img(img_crop_nobg, img_crop_nobg_path)

                    if cfg.save_obj_crop_mask:
                        # make copy of img_crop
                        img_crop_mask = img_crop.copy()
                        # set background to 0 in regions not the current object
                        img_crop_mask[pix_obj_ids_crop != obj_id] = 0
                        # set foreground to white
                        img_crop_mask[pix_obj_ids_crop == obj_id] = 255
                        # save
                        img_crop_mask_path = img_crop_mask_dir / scene_id / f'{image_name}_{obj_id}.png'
                        save_img(img_crop_mask, img_crop_mask_path)

                    # draw a bbox around the object and save the full image
                    # create new image with bbox of the object draw on the full image
                    img_copy = img.copy()
                    x, y, w, h = obj_bbox
                    # convert image RGB to BGR
                    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(img_copy, (y, x), (y+h, x+w), (0, 0, 255), 2)
                    # convert back to RGB
                    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                    global_prompt_img_path = bbox_img_dir / scene_id / f'{image_name}_{obj_id}.png'
                    # save it to file
                    save_img(img_copy, global_prompt_img_path)

                    # other useful info for this object
                    obj_location_3d = np.round(obj_id_locations[obj_id], 2).tolist()
                    x, y, w, h = obj_bbox
                    # center of the bbox
                    obj_location_2d = np.round([x + w/2, y + h/2]).tolist()
                    obj_dims_3d = obj_id_dims[obj_id]
                    # semantic label
                    obj_label = anno['objects'][obj_id]['label']
                    # vertices in this object
                    obj_mask_3d = vtx_obj_ids == obj_id

                    # save the object point cloud
                    if cfg.save_obj_pcs:
                        out_path = obj_pcs_dir / scene_id / f'{obj_id}.ply'
                        # same for all images
                        if not out_path.exists():
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            # get the pc for this object from the mesh, with colors
                            obj_pc = o3d.geometry.PointCloud()
                            obj_pc.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[obj_mask_3d])
                            obj_pc.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[obj_mask_3d])
                            o3d.io.write_point_cloud(str(out_path), obj_pc)

                    if cfg.save_obj_meshes:
                        # crop the mesh to the object 3d bbox
                        bbox_rot = np.array(anno['objects'][obj_id]['obb']['normalizedAxes']).reshape(3, 3).T
                        obj_dims_3d = np.array(obj_dims_3d)
                        if cfg.get('expand_mesh_factor'):
                            obj_dims_3d = obj_dims_3d * cfg.expand_mesh_factor
                        bbox_3d = o3d.geometry.OrientedBoundingBox(obj_location_3d, bbox_rot, obj_dims_3d)
                        # crop the mesh to the 3d bbox
                        obj_mesh = mesh.crop(bbox_3d)
                        # save the mesh
                        out_path = obj_meshes_dir / scene_id / f'{obj_id}.ply'
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        o3d.io.write_triangle_mesh(str(out_path), obj_mesh)


if __name__ == "__main__":
    main()