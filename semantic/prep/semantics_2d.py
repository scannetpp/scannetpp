'''
Get 3D semantics onto 2D images using precomputed rasterization
'''

from common.utils.image import get_img_crop, load_image, save_img, viz_ids
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

    img_crop_dir =  save_dir / 'img_crops'
    bbox_img_dir =  save_dir / 'img_bbox'
    viz_obj_ids_dir = save_dir / 'viz_obj_ids'

    for dir in [img_crop_dir, bbox_img_dir, viz_obj_ids_dir]:
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

        if cfg.check_visibility:
            # create visibility cache to pick topk images where an object is visible
            visibility_data = get_visiblity_from_cache(scene, rasterout_dir, cfg.visiblity_cache_dir, cfg.image_type, cfg.subsample_factor, cfg.undistort_dslr, anno=anno)
            if cfg.create_visiblity_cache_only:
                print(f'Created visibility cache for {scene_id}')
                continue

        vtx_obj_ids = anno['vertex_obj_ids']
        # read mesh
        mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path)) 

        obj_ids = np.unique(vtx_obj_ids)
        # remove 0
        obj_ids = sorted(obj_ids[obj_ids != 0])

        obj_id_locations = {obj_id: anno['objects'][obj_id]['obb']['centroid'] for obj_id in obj_ids}
        obj_id_dims = {obj_id: anno['objects'][obj_id]['obb']['axesLengths'] for obj_id in obj_ids}

        # get the list of iphone/dslr images and poses
        # NOTE: should be the same as during rasterization
        colmap_camera, image_list, _, distort_params = get_camera_images_poses(scene, cfg.subsample_factor, cfg.image_type)
        # keep first 4 elements
        distort_params = distort_params[:4]

        intrinsic = camera_to_intrinsic(colmap_camera)
        img_height, img_width = colmap_camera.height, colmap_camera.width

        undistort_map1, undistort_map2 = None, None
        if cfg.image_type == 'dslr' and cfg.undistort_dslr:
            undistort_intrinsic = compute_undistort_intrinsic(intrinsic, img_height, img_width, distort_params)
            undistort_map1, undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                intrinsic, distort_params, np.eye(3), undistort_intrinsic, (img_width, img_height), cv2.CV_32FC1
            )

        # go through list of images
        for _, image_name in enumerate(tqdm(image_list, desc='image', leave=False)):
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
                print(f'Rasterization not found for {image_name}')
                continue
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

            if cfg.dbg.viz_obj_ids: # save viz to file
                out_path = viz_obj_ids_dir / scene_id / f'{image_name}.png'
                viz_ids(img, pix_obj_ids, out_path)

            # create semantics GT and of semantic ids on vertices, -1 = no semantic label
            if cfg.save_semantic_gt_2d:
                out_path = save_dir / scene_id / f'{image_name}.png'
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
                    out_path = save_dir / scene_id / f'{image_name}_viz.png'
                    print(f'Saving 2d semantic viz to {out_path}')
                    viz_sem_ids_2d(pix_sem_ids, semantic_colors, out_path)
                continue # do only semantics, nothing else

            # get objid -> bbox x,y,w,h after upsampling rasterization, all the objs in this image
            bboxes_2d = get_bboxes_2d(pix_obj_ids)

            # go through each object that has a bbox 
            for _, (obj_id, obj_bbox) in enumerate(tqdm(bboxes_2d.items(), desc='obj', leave=False)):
                if obj_id == 0:
                    continue

                if cfg.check_visibility:
                    # enough of the object is seen
                    if visibility_data['images'][image_name]['objects'][obj_id].get('visible_vertices_frac', 0) < cfg.obj_visible_thresh:
                        continue
                    
                    # check if obj occupies enough % of the image
                    if visibility_data['images'][image_name]['objects'][obj_id].get('visible_pixels_frac', 0) < cfg.obj_pixel_thresh:
                        continue

                    if visibility_data['images'][image_name]['objects'][obj_id].get('zbuf_min', 9999) > cfg.obj_dist_thresh:
                        # object is too far away from camera
                        continue
                    
                    if cfg.visibility_topk is not None:
                        images_visibilites = []
                        for i_name in visibility_data['images']:
                            if obj_id in visibility_data['images'][i_name]['objects'] and 'visible_vertices_frac' in visibility_data['images'][i_name]['objects'][obj_id]:
                                images_visibilites.append((i_name, visibility_data['images'][i_name]['objects'][obj_id]['visible_vertices_frac']))
                        # sort descending by visibility
                        images_visibilites.sort(key=lambda x: x[1], reverse=True)
                        top_images = [i_name for i_name, _ in images_visibilites][:cfg.visibility_topk]
                        # dont consider this object in this image
                        if image_name not in top_images: 
                            continue

                # crop the object from the image
                img_crop = get_img_crop(img, obj_bbox, cfg.bbox_expand_factor, expand_bbox=True)
                img_crop_path = img_crop_dir / scene_id / f'{image_name}_{obj_id}.png'
                save_img(img_crop, img_crop_path)

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
                obj_dims_3d = np.round(obj_id_dims[obj_id], 2).tolist()
                # semantic label
                obj_label = anno['objects'][obj_id]['label']
                # vertices in this object
                obj_mask_3d = vtx_obj_ids == obj_id

if __name__ == "__main__":
    main()