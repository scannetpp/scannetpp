

import argparse
from common.file_io import read_txt_list, load_yaml_munch
from common.scene_release import ScannetppScene_Release
from semantic.utils.confmat import ConfMat
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch


def eval_semantic(scene_list, pred_dir, gt_dir, data_root, num_classes, ignore_label, 
            top_k_pred=[1, 3]):

    # create one confmat for each k
    confmats = {k: 
        ConfMat(num_classes, top_k_pred=k, ignore_label=ignore_label)
        for k in top_k_pred
    }

    # go through each scene
    for scene_id in tqdm(scene_list):
        # read the N,3 GT
        gt = np.loadtxt(Path(gt_dir) / f'{scene_id}.txt', dtype=np.int32, delimiter=',')
        # read the predictions N, or N,3 (usually)
        pred = np.loadtxt(Path(pred_dir) / f'{scene_id}.txt', dtype=np.int32, delimiter=',')

        # convert to torch tensors
        pred = torch.LongTensor(pred)

        # single prediction? repeat to make it N, max(top_k_pred)
        if len(pred.shape) == 1:
            pred = pred.reshape(-1, 1).repeat(1, max(top_k_pred))

        gt = torch.LongTensor(gt)

        # create scene object to get the mesh mask
        scene = ScannetppScene_Release(scene_id, data_root=data_root)
        # vertices to ignore for eval
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
    num_classes = len(read_txt_list(cfg.classes_file))

    confmats = eval_semantic(scene_ids, cfg.preds_dir, cfg.gt_dir, cfg.data_root,
                            num_classes, -100, [1, 3])
    
    for k, confmat in confmats.items():
        print(f'Top {k} mIOU: {confmat.miou}')
        print('All IoUs:', confmat.ious)

    

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()
    main(args)