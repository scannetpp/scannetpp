from semantic.transforms.common import Compose
from semantic.transforms.mesh import AddSegmentIDs, AddVertexNormals, MapLabelToIndex, SamplePointsOnMesh, \
             GetLabelsOnVertices, AddMeshVertices


def get_transform(data_cfg):
    transforms_list = data_cfg.transforms
    
    transforms = []

    if 'add_mesh_vertices' in transforms_list:
        transforms.append(AddMeshVertices())

    # map string label to 0..N
    if 'map_label_to_index' in transforms_list:
        transforms.append(MapLabelToIndex(data_cfg.labels_path, data_cfg.ignore_label,
                                        count_thresh=data_cfg.get('count_thresh', 0),
                                        mapping_file=data_cfg.mapping_file,
                                        keep_classes=data_cfg.get('keep_classes')))

    if 'get_labels_on_vertices' in transforms_list:
        # get 0..N labels for each vertex
        transforms.append(GetLabelsOnVertices(data_cfg.ignore_label, data_cfg.get('multilabel'),
                                              use_instances=data_cfg.use_instances,
                                              instance_labels_path=data_cfg.instance_labels_path))
        
    # add segment ID for each vertex from the segments file
    if 'add_segment_ids' in transforms_list:
        transforms.append(AddSegmentIDs())
        
    # add vtx_normals from the o3d mesh
    if 'add_normals' in transforms_list:
        transforms.append(AddVertexNormals())
        
    if 'sample_points_on_mesh' in transforms_list :
        # sample points -> new coordinates with colors and labels
        transforms.append(SamplePointsOnMesh(data_cfg['sample_factor']))

    t = Compose(transforms)             

    return t
