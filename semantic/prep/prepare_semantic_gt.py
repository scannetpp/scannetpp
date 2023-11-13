'''
Read training pth files and save only the ground truth labels as npy/txt files
'''

from pathlib import Path
from common.file_io import write_json
from common.utils.rle import rle_encode
import torch
from tqdm import tqdm
import numpy as np
from common.file_io import load_yaml_munch, read_txt_list
import argparse

'''
ScanNet++ format for semantic GT and predictions
---------------------------------------------
Format for 3D Semantic Label Prediction
Both for the ScanNet and ScanNet200 3D semantic label prediction task, 
results must be provided as class labels per vertex of the corresponding 
3D scan mesh, i.e., for each vertex in the order provided by the *_vh_clean_2.ply mesh. 
Each prediction file should contain one line per vertex, with each line containing 
the integer label id of the predicted class. 
E.g., a file could look like:
10
10
2
2
2
⋮
39
A submission must contain a .txt prediction file image for each test scan, 
named <scene_id>.txt with the corresponding ScanNet++ scene ID, e.g.:
unzip_root/
 |-- sceneid1.txt
 |-- sceneid2.txt
 |-- sceneid3.txt
     ⋮
 |-- sceneidN.txt

Scannet++ format for instance predictions - 
------------------------------------------
Results must be provided as a text file for each test scan. Each text file should contain a line for each instance, containing the relative path to a binary mask of the instance, the predicted label id, and the confidence of the prediction. The result text files must be named according to the corresponding test scan, as scene%04d_%02d.txt with the corresponding ScanNet scan name. Predicted .txt files listing the instances of each scan must live in the root of the unzipped submission. Predicted instance mask files must live in a subdirectory of the unzipped submission. For instance, a submission should look like:
unzip_root/
 |-- sceneid1.txt
 |-- sceneid2.txt
     ⋮
 |-- sceneidN.txt
 |-- predicted_masks/
    |-- sceneid1_000.txt
    |-- sceneid2_001.txt
         ⋮
Each prediction file for a scan should contain a list of instances, where an instance is:
 (1) the relative path to the predicted mask file, 
 (2) the integer class label id, 
 (3) the float confidence score. 
 Each line in the prediction file should correspond to one instance, 
 and the three values above separated by spaces. 
 Thus, the filenames in the prediction files must not contain spaces.
The predicted instance mask file should provide a mask over the vertices of the scan mesh, 
i.e., for each vertex in the order provided by the mesh. 
Each instance mask file should contain one line per vertex,
 with each line containing an integer value, with non-zero values indicating 
 part of the instance. E.g., sceneid1.txt should be of the format:
predicted_masks/sceneid1_000.txt 10 0.7234
predicted_masks/sceneid1_001.txt 36 0.9038
     ⋮
and predicted_masks/sceneid1_000.txt could look like:
0
0
0
1
1
⋮
0
'''

def main(args):
    cfg = load_yaml_munch(args.config_file)

    scene_list = read_txt_list(cfg.scene_list)
    
    if cfg.save_semantic:
        sem_out_dir = Path(cfg.sem_out_dir)
        sem_out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_instance:
        if cfg.inst_gt_format:
            inst_gtformat_out_dir = Path(cfg.inst_gtformat_out_dir)
            inst_gtformat_out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.inst_preds_format:
            inst_predsformat_out_dir = Path(cfg.inst_predsformat_out_dir)
            inst_predsformat_out_dir.mkdir(parents=True, exist_ok=True)

    for scene_id in tqdm(scene_list):
        pth_path = Path(cfg.pth_dir) / f'{scene_id}.pth'

        if not pth_path.is_file():
            raise FileNotFoundError(f'pth file {pth_path} not found')

        data = torch.load(pth_path)
        
        # scannet format for semantic GT/predictions
        if cfg.save_semantic:
            gt = data['vtx_labels']

            if cfg.save_npy:
                np.save(sem_out_dir / f'{scene_id}.npy', gt)     
            if cfg.save_txt:
                np.savetxt(sem_out_dir / f'{scene_id}.txt', gt, fmt='%d', delimiter=',')

        if cfg.save_instance:
            # convert to int32 to handle more than 20 classes
            sem_gt = data['vtx_labels'].astype(np.int32)
            inst_gt = data['vtx_instance_labels'].astype(np.int32)

            # one file per scene
            # with semantic * 1000 + instance ID
            if cfg.inst_gt_format:
                # remove wall, ceiling, floor
                # sem_gt = sem_gt - 3 + 1
                # sem_gt[sem_gt < 0] = 0
                
                ignore_inds = inst_gt <= 0
                inst_out = sem_gt * 1000 + inst_gt
                inst_out[ignore_inds] = 0

                if cfg.save_npy:
                    np.save(inst_gtformat_out_dir / f'{scene_id}.npy', inst_out)     
                if cfg.save_txt:
                    np.savetxt(inst_gtformat_out_dir / f'{scene_id}.txt', inst_out, fmt='%d', delimiter=',')
                    
            # same as the ScanNet format described above
            if cfg.inst_preds_format:
                # create main txt file
                main_txt_file = inst_predsformat_out_dir / f'{scene_id}.txt'
                # get the unique and valid instance IDs in inst_gt 
                # (ignore invalid IDs)
                inst_ids = np.unique(inst_gt)
                inst_ids = inst_ids[inst_ids > 0]
                # main txt file lines
                main_txt_lines = []

                # create the dir for the instance masks
                inst_masks_dir = inst_predsformat_out_dir / 'predicted_masks'
                inst_masks_dir.mkdir(parents=True, exist_ok=True)

                # for each instance
                for inst_ndx, inst_id in enumerate(tqdm(sorted(inst_ids))):
                # get the mask for the instance
                    inst_mask = inst_gt == inst_id
                    # get the semantic label for the instance
                    inst_sem_label = sem_gt[inst_mask][0]
                    # add a line to the main file with relative path
                    # predicted_masks <semantic label> <confidence=1>
                    mask_path_relative = f'predicted_masks/{scene_id}_{inst_ndx:03d}.json'
                    main_txt_lines.append(f'{mask_path_relative} {inst_sem_label} 1.0')
                    # save the instance mask to a file in the predicted_masks dir
                    mask_path = inst_predsformat_out_dir / mask_path_relative
                    write_json(mask_path, rle_encode(inst_mask))

                # save the main txt file
                with open(main_txt_file, 'w') as f:
                    f.write('\n'.join(main_txt_lines))

            if not cfg.inst_gt_format and not cfg.inst_preds_format:
                raise ValueError('Instance GT and preds format both are false. Set at least one to true')

            


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()

    main(args)