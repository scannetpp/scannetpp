import os
import json

import numpy as np
from types import SimpleNamespace


def load_ids(filename):
    ids = np.loadtxt(filename, dtype=np.int32)
    return ids

# ------------ Instance Utils ------------ #

 # store all label related info in namespace
def get_label_info(semantic_class_list, instance_class_list):
    label_info = SimpleNamespace()
    # all semantic classes
    label_info.all_class_labels = semantic_class_list
    # indices of semantic classes not present in instance class list
    label_info.ignore_classes = [i for i in range(len(label_info.all_class_labels)) if label_info.all_class_labels[i] not in instance_class_list]
    # ids of all semantic classes
    label_info.all_class_ids = list(range(len(label_info.all_class_labels)))
    # ids of instance classes (all semantic classes - ignored classes)
    label_info.class_labels = [label_info.all_class_labels[i] for i in label_info.all_class_ids if i not in label_info.ignore_classes]
    # ids of instance classes
    label_info.valid_class_ids = [i for i in label_info.all_class_ids if i not in label_info.ignore_classes]

    label_info.id_to_label = {}
    label_info.label_to_id = {}

    for i in range(len(label_info.valid_class_ids)):
        # class id -> class name
        label_info.id_to_label[label_info.valid_class_ids[i]] = label_info.class_labels[i]
        # class name -> class id
        label_info.label_to_id[label_info.class_labels[i]] = label_info.valid_class_ids[i]

    return label_info

class Instance_Eval_Opts:
    overlaps             = np.append(np.arange(0.5,0.95,0.05), 0.25)
    # minimum region size for evaluation [verts]
    min_region_sizes     = np.array( [ 100 ] )
    # distance thresholds [m]
    distance_threshes    = np.array( [  float('inf') ] )
    # distance confidences
    distance_confs       = np.array( [ -float('inf') ] )

class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        # mesh_vert_instances: instance ids for all verts
        # instance_id: id of the current instance = sem * 1000 + inst
        if (instance_id == -1):
            return
        # instance id -> same as provided = sem*1000 + inst
        self.instance_id     = int(instance_id)
        # label id -> remainder when divided by 1000 = ID of this instance
        self.label_id    = int(self.get_label_id(instance_id))
        # number of vertices in this instance
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        # divide by 1000 to get the sem class ID
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"]    = self.label_id
        dict["vert_count"]  = self.vert_count
        dict["med_dist"]    = self.med_dist
        dict["dist_conf"]   = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id     = int(data["instance_id"])
        self.label_id        = int(data["label_id"])
        self.vert_count      = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist    = float(data["med_dist"])
            self.dist_conf   = float(data["dist_conf"])

    def __str__(self):
        return "("+str(self.instance_id)+")"

def read_instance_prediction_file(filename, pred_path):
    lines = open(filename).read().splitlines()
    instance_info = {}
    abs_pred_path = os.path.abspath(pred_path)
    
    for line in lines:
        parts = line.split(' ')
        assert len(parts) == 3, f'Invalid instance prediction file {filename}. Expected (per line): [rel path prediction] [label id prediction] [confidence prediction]'
        assert not os.path.isabs(parts[0]), f'Invalid instance prediction file {filename}. First entry in line must be a relative path'

        mask_file = os.path.join(os.path.dirname(filename), parts[0])
        mask_file = os.path.abspath(mask_file)

        # check that mask_file lives inside prediction path
        assert os.path.commonprefix([mask_file, abs_pred_path]) == abs_pred_path, \
            'fPredicted mask {mask_file} in prediction text file {filename} points outside of prediction path'
        info            = {}
        info["label_id"] = int(float(parts[1]))
        info["conf"]    = float(parts[2])
        instance_info[mask_file]  = info
    return instance_info


def get_instances(ids, valid_class_ids, valid_class_labels, id2label):
    instances = {}
    # each class name
    for label in valid_class_labels:
        # instances in this class
        instances[label] = []
    # unique instance IDs
    instance_ids = np.unique(ids)
    # ignore instance ID 0!
    for id in instance_ids:
        if id == 0:
            continue
        # create new instance object
        inst = Instance(ids, id)
        if inst.label_id in valid_class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    # list of instances as dicts
    return instances
            


