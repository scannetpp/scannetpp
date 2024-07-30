
import numpy as np
from common.file_io import read_txt_list
from scipy.spatial import KDTree
import pandas as pd

from semantic.prep.map_semantic import filter_map_classes


class MapLabelToIndex:
    '''
    map anno labels such as 'chair' to 0..N indices
    '''
    def __init__(self, labels_path, ignore_label, count_thresh=None, mapping_file=None, keep_classes=None):
        with open(labels_path) as f: 
            labels = f.read().splitlines()

        # keep only these classes
        if keep_classes:
            self.class_names, self.label_mapping = keep_classes, None
        # use a mapping file
        elif mapping_file:
            self.class_names = labels
            mapping = pd.read_csv(mapping_file)
            print('Classes before mapping:', len(mapping))
            mapped_classes, self.label_mapping = filter_map_classes(mapping, count_thresh, 
                                    count_type='count', mapping_type='semantic')
            print('Classes after mapping:', len(mapped_classes))
        else:
            self.class_names, self.label_mapping = labels, None

        self.ignore_label = ignore_label
        # map class name to index 0..N in the same order
        self.mapping = {label: ndx for (ndx, label) in enumerate(self.class_names)}

    def get_mapping(self):
        return self.mapping

    def get_class_names(self):
        return self.class_names

    def __call__(self, sample):
        for ndx, anno in enumerate(sample['anno']['segGroups']):
            label = anno['label']
            
            # store original label
            sample['anno']['segGroups'][ndx]['label_orig'] = label

            # need to remap labels? eg. books->book
            if self.label_mapping is not None:
                label = self.label_mapping.get(label, None)
                # in case label is remapped - put the new label into the anno dict
                sample['anno']['segGroups'][ndx]['label'] = label

            # name -> 0..N, else ignore label
            label_ndx = self.mapping.get(label, self.ignore_label)

            sample['anno']['segGroups'][ndx]['label_ndx'] = label_ndx 

        return sample

class AddSegmentIDs:
    def __init__(self):
        pass
    
    def __call__(self, sample):
        seg_indices = np.array(sample['segments']['segIndices'], dtype=np.uint32)
        sample['vtx_segment_ids'] = seg_indices
        
        return sample

class AddVertexNormals:
    def __init__(self):
        pass
    
    def __call__(self, sample):
        if not sample['o3d_mesh'].has_vertex_normals():
            sample['o3d_mesh'].compute_vertex_normals()
            
        sample['vtx_normals'] = np.asarray(sample['o3d_mesh'].vertex_normals)
        
        return sample

class GetLabelsOnVertices:
    '''
    label per segment group -> get label_ndx per vertex
    '''
    def __init__(self, ignore_label, multilabel_cfg=None, multilabel_max=3, use_instances=None,
                        instance_labels_path=None):
        self.ignore_label = ignore_label
        self.multilabel_cfg = multilabel_cfg
        # max number of multilabels that can be assigned to a vertex
        self.multilabel_max = multilabel_max
        # use instances or not
        self.use_instances = use_instances
        if self.use_instances:
            self.instance_labels = read_txt_list(instance_labels_path)

    def __call__(self, sample):
        seg_indices = np.array(sample['segments']['segIndices'], dtype=np.uint32)
        num_verts = len(seg_indices)

        multilabel = self.multilabel_cfg is not None

        # first store multilabels into array
        # if using single label, keep the label of the smallest instance for each vertex
        # else, keep everything
        if multilabel:
            max_gt = self.multilabel_cfg['max_gt']
        else:
            max_gt = self.multilabel_max
            
        # semantic multilabels
        multilabels = np.ones((num_verts, max_gt), dtype=np.int16) * self.ignore_label
        # how many labels are used per vertex? initially 0
        # increment each time a new label is added
        # 0, 1, 2 eg. if max_gt is 3
        labels_used = np.zeros(num_verts, dtype=np.int16)
        # keep track of the size of the instance (#vertices) assigned to each vertex
        # later, keep the label of the smallest instance for multilabeled vertices
        # store inf initially so that we can pick the smallest instance
        instance_size = np.ones((num_verts, max_gt), dtype=np.int16) * np.inf
        
        # all instance labels, including multilabels
        instance_multilabels = None
        # the final instance labels
        instance_labels = None
        instance_anno_id_multi = None
        
        if self.use_instances:
            # keep all instance labels initially
            # then pick only the ones required
            # same ignore label for instances
            # used for unannotated regions and non-instance classes

            # new instance IDs from 0..N
            instance_multilabels = np.ones((num_verts, max_gt), dtype=np.int16) * self.ignore_label      
            # object id from the annotation to link back to the JSON, could be different from instance_ndx
            instance_anno_id_multi = np.ones((num_verts, max_gt), dtype=np.int16) * self.ignore_label      
        
        for instance_ndx, instance in enumerate(sample['anno']['segGroups']):
            if instance['label_ndx'] == self.ignore_label:
                continue
            # get all the vertices with segment index in this instance
            # and max number of labels not yet applied
            inst_mask = np.isin(seg_indices, instance['segments']) & (labels_used < max_gt)
            
            num_vertices = inst_mask.sum()
            if num_vertices == 0:
                continue
            
            # get the position to add the label - 0, 1, 2
            new_label_position = labels_used[inst_mask]
            multilabels[inst_mask, new_label_position] = instance['label_ndx']
            
            # add instance label only for instance classes
            if self.use_instances and instance['label'] in self.instance_labels:
                instance_multilabels[inst_mask, new_label_position] = instance_ndx
                # store the object ID from the annotation
                instance_anno_id_multi[inst_mask, new_label_position] = instance['objectId']
                
            # store number of vertices in this instance
            instance_size[inst_mask, new_label_position] = num_vertices
            labels_used[inst_mask] += 1
            
        # if single label: keep only the smallest instance for each vertex
        # else, keep everything
        if not multilabel: 
            labels = multilabels[:, 0]
            # vertices which have multiple labels
            has_multilabel = labels_used > 1
            # get the label of the smallest instance for multilabeled vertices
            smallest_instance_ndx = np.argmin(instance_size[has_multilabel], axis=1)
            labels[has_multilabel] = multilabels[has_multilabel, smallest_instance_ndx] 
            
            if instance_multilabels is not None:
                # pick the 1st label for everything
                instance_labels = instance_multilabels[:, 0]
                # pick the label of the smallest instance for multilabeled vertices
                instance_labels[has_multilabel] = instance_multilabels[has_multilabel, smallest_instance_ndx]
                # repeat for anno id
                instance_anno_id = instance_anno_id_multi[:, 0]
                instance_anno_id[has_multilabel] = instance_anno_id_multi[has_multilabel, smallest_instance_ndx]
        else:
            labels = multilabels
            instance_labels = instance_multilabels
            instance_anno_id = instance_anno_id_multi
        
        if multilabel and self.multilabel_cfg['multilabel_only']:
            # keep only labels on vertices with >= 2 labels
            multilabels[labels_used <= 1] = self.ignore_label
            labels_used[labels_used <= 1] = 0
            labels = multilabels
            
            if instance_multilabels:
                instance_multilabels[labels_used <= 1] = self.ignore_label
                instance_labels = instance_multilabels
                instance_anno_id[labels_used <= 1] = self.ignore_label

        sample['vtx_num_labels'] = labels_used
        sample['vtx_labels'] = labels
        sample['vtx_instance_labels'] = instance_labels  
        if self.use_instances:
            sample['vtx_instance_anno_id'] = instance_anno_id         

        return sample


class AddMeshVertices:
    '''
    get coords, colors from mesh
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['vtx_coords'] = np.array(sample['o3d_mesh'].vertices)
        sample['vtx_colors'] = np.array(sample['o3d_mesh'].vertex_colors)

        return sample

class SamplePointsOnMesh:
    '''
    mesh with vertices, faces, colors -> points on mesh, colors
    '''
    def __init__(self, sample_factor=1):
        self.sample_factor = sample_factor

    def __call__(self, sample):
        mesh = sample['o3d_mesh']

        # keep only sampled properties, not vertex properties 
        new_sample = {'scene_id': sample['scene_id']}

        pc = mesh.sample_points_uniformly(int(self.sample_factor * len(sample['vtx_coords'])))	

        # coords and colors of sampled points
        new_sample['sampled_coords'] = np.array(pc.points)
        new_sample['sampled_colors'] = np.array(pc.colors)

        tree = KDTree(mesh.vertices)
        # for each sampled point, get the nearest original vertex
        _, ndx = tree.query(new_sample['sampled_coords'])

        # any vtx properties other than coords and colors
        # get these on the sampled points
        # rename the property to sample_<property>
        for k, v in sample.items():
            if k.startswith('vtx_') and k not in ['vtx_coords', 'vtx_colors']:
                new_sample['sampled_' + k[4:]] = v[ndx]
        return new_sample