
import json
import numpy as np

def get_vtx_obj_ids(scene):
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
        # everything except segments
        ret_dict['objects'] = {segGroup['objectId']: {k: v for k, v in segGroup.items() if k != 'segments'} for segGroup in anno['segGroups']}
    else:   
        # everything
        ret_dict['objects'] = anno['segGroups']

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
    pix_vtx_prop[valid_pix_to_face] = vtx_prop[mesh_faces_np[pix_to_face[valid_pix_to_face]][:, 0]]

    return pix_vtx_prop.squeeze()
