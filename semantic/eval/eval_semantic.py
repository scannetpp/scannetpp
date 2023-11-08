

import argparse
from common.file_io import read_txt_list, load_yaml_munch
from common.scene_release import ScannetppScene_Release
from semantic.utils.confmat import ConfMat
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch

import warnings


def eval_semantic(scene_list, pred_dir, gt_dir, data_root, num_classes, ignore_label, 
            top_k_pred=[1, 3], eval_against_gt=False):
    # check if all files exist
    for scene_id in tqdm(scene_list):
        assert (Path(pred_dir) / f'{scene_id}.txt').is_file(), f'Pred file {scene_id}.txt not found'

    # create one confmat for each k
    confmats = {k: 
        ConfMat(num_classes, top_k_pred=k, ignore_label=ignore_label)
        for k in top_k_pred
    }

    max_k = max(top_k_pred)

    # go through each scene
    for scene_id in tqdm(scene_list):
        # read the N,3 GT
        gt = np.loadtxt(Path(gt_dir) / f'{scene_id}.txt', dtype=np.int32, delimiter=',')
        # read the predictions N, or N,3 (usually)
        pred = np.loadtxt(Path(pred_dir) / f'{scene_id}.txt', dtype=np.int32, delimiter=',')

        assert pred.shape[0] == gt.shape[0], f'Prediction for {scene_id} does not match #GT vertices: pred={pred.shape}, GT={gt.shape}'

        # evaluating against preds -> make sure there are no negative preds
        # when evaluating against GT, we allow negative preds
        if not eval_against_gt:
            # assert min of preds is >= 0
            assert pred.min() >= 0, f'Prediction for {scene_id} contains negative labels: {pred.min()}'
        # assert max of preds is  <= (num_classes-1)
        assert pred.max() <= (num_classes-1), f'Prediction for {scene_id} contains labels > {num_classes-1}: {pred.max()}'

        if eval_against_gt:
            pred = pred[:, 0]

        # convert to torch tensors
        pred = torch.LongTensor(pred)

        # prediction should be N, or N,k_max currently, assert this
        assert len(pred.shape) == 1 or pred.shape[1] == max_k, f'Prediction shape {pred.shape} not supported'

        # single prediction? repeat to make it N, max_k
        if len(pred.shape) == 1:
            print(f'Found single prediction for {scene_id}, repeating {max_k} times')
            pred = pred.reshape(-1, 1).repeat(1, max_k)

        gt = torch.LongTensor(gt)

        # create scene object to get the mesh mask
        scene = ScannetppScene_Release(scene_id, data_root=data_root)
        # vertices to ignore for eval
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=1)
            ignore_vtx = torch.LongTensor(np.loadtxt(scene.scan_mesh_mask_path, dtype=np.int32))

        # dont eval on masked regions
        # keep all preds and gt except masked regions
        vtx_ndx = torch.arange(len(gt))
        # vertices to keep
        keep_vtx = ~torch.isin(vtx_ndx, ignore_vtx)

        for _, confmat in confmats.items():
            confmat.update(pred[keep_vtx], gt[keep_vtx])
        
    return confmats


def main(args):
    cfg = load_yaml_munch(args.config_file)

    scene_ids = read_txt_list(cfg.scene_list_file)
    semantic_classes = read_txt_list(cfg.classes_file) 
    num_classes = len(semantic_classes)

    if cfg.preds_dir == cfg.gt_dir:
        print('Evaluating against GT')
        eval_against_gt = True
    else:
        eval_against_gt = False

    confmats = eval_semantic(scene_ids, cfg.preds_dir, cfg.gt_dir, cfg.data_root,
                            num_classes, -100, [1, 3], eval_against_gt=eval_against_gt)
    
    for k, confmat in confmats.items():
        print(f'Top {k} mIOU: {confmat.miou}')

        for class_name, class_iou in zip(semantic_classes, confmat.ious):
            print(f'{class_name: <25}: {class_iou}')

        print('----------------------------------------------------')
    

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()
    main(args)