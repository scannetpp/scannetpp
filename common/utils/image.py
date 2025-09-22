import open3d as o3d
import numpy as np
import cv2


def get_img_crop(img, bbox, bbox_expand_factor, expand_bbox=True):
    # x is along the width, y is along the height
    x, y, w, h = bbox
    if expand_bbox and bbox_expand_factor > 0:
        # keep the new x, y, w, h within the image 
        x = max(0, int(bbox[0] - bbox_expand_factor*bbox[2]))
        y = max(0, int(bbox[1] - bbox_expand_factor*bbox[3]))
        w = min(img.shape[0] - x, int(bbox[2] + 2*bbox_expand_factor*bbox[2]))
        h = min(img.shape[1] - y, int(bbox[3] + 2*bbox_expand_factor*bbox[3]))
    return img[x:x+w, y:y+h]


def save_img(img, path):
    # make parent if necessary
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d_img = o3d.geometry.Image(img.astype(np.uint8))
    o3d.io.write_image(str(path), o3d_img)

def load_image(img_path):
    return np.asarray(o3d.io.read_image(img_path))

def viz_ids(img, img_ids, out_path):
    viz_img = np.zeros_like(img)
    unique_ids = np.unique(img_ids)
    # ignore negative and 0
    unique_ids = unique_ids[unique_ids > 0]

    for obj_id in unique_ids:
        mask = img_ids == obj_id
        color = np.random.randint(0, 255, size=3)
        viz_img[mask] = color
    
    save_img(viz_img, out_path)

def viz_obj_ids_txt(img, img_ids, out_path):
    viz_img = np.zeros_like(img)
    unique_ids = np.unique(img_ids)
    # ignore negative and 0
    unique_ids = unique_ids[unique_ids > 0]
    
    # first add all the object colors, then add text labels
    # otherwise colors overwrite the text labels
    for obj_ids in unique_ids:
        mask = img_ids == obj_ids
        color = np.random.randint(0, 255, size=3)
        viz_img[mask] = color

    for obj_ids in unique_ids:
        # add a small text label in the center of the bbox with the objid
        mask = img_ids == obj_ids
        obj_max_coords = np.max(np.argwhere(mask), axis=0)
        obj_min_coords = np.min(np.argwhere(mask), axis=0)
        # draw black box
        # exchange x and y to go to opencv format
        viz_img = cv2.rectangle(viz_img, (obj_min_coords[1], obj_min_coords[0]), (obj_max_coords[1], obj_max_coords[0]), (255, 255, 255), 1)
        # convert to int
        # and write the obj id in the top left corner of the box, add a small offset to get the text inside the box
        # (location is the bottom left of the text)
        text_location = (obj_min_coords[1], obj_min_coords[0] + 10)
        viz_img = cv2.putText(viz_img, str(obj_ids), text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    save_img(viz_img, out_path)
