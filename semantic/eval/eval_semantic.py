

import argparse
from common.file_io import read_txt_list, load_yaml_munch
from common.scene_release import ScannetppScene_Release
from semantic.utils.confmat import ConfMat
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch


def eval_semantic(scene_list, pred_dir, gt_dir, data_root, num_classes, ignore_label, 
            top_k_pred=1):
    confmat = ConfMat(num_classes, top_k_pred=top_k_pred, ignore_label=ignore_label)

    for scene_id in tqdm(scene_list):
        gt = np.loadtxt(Path(gt_dir) / f'{scene_id}.txt', dtype=np.int32)
        pred = np.loadtxt(Path(pred_dir) / f'{scene_id}.txt', dtype=np.int32)

        pred = torch.LongTensor(pred.reshape(-1, top_k_pred))
        gt = torch.LongTensor(gt)

        scene = ScannetppScene_Release(scene_id, data_root=data_root)
        ignore_vtx = torch.LongTensor(np.loadtxt(scene.scan_mesh_mask_path, dtype=np.int32))

        # dont eval on masked regions
        # keep all preds and gt except masked regions
        vtx_ndx = torch.arange(len(gt))
        keep_vtx = ~torch.isin(vtx_ndx, ignore_vtx)

        confmat.update(pred[keep_vtx], gt[keep_vtx])
        
    return confmat


def main(args):
    cfg = load_yaml_munch(args.config_file)

    scene_ids = read_txt_list(cfg.scene_list_file)
    num_classes = len(read_txt_list(cfg.classes_file))

    confmat = eval_semantic(scene_ids, cfg.preds_dir, cfg.gt_dir, cfg.data_root,
                            num_classes, -100, 1)
    
    print(confmat.ious)
    

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()
    main(args)