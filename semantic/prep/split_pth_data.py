'''
read each pth data
split all the vertex properties into smaller pth files
do this by splitting the scene in the XY axis into chunks
eg. 10m x 10m
pick all the points within these chunks and save to separate files
'''

from pathlib import Path
from common.utils.utils import read_txt_list
import torch
from tqdm import tqdm
import argparse
from common.file_io import load_yaml_munch


def main(args):
    cfg = load_yaml_munch(args.config_file)

    Path(cfg.out_list_path).parent.mkdir(parents=True, exist_ok=True)
    out_pth_dir = Path(cfg.out_pth_dir)
    out_pth_dir.mkdir(parents=True, exist_ok=True)

    chunk_dim_x, chunk_dim_y = cfg.chunk_dims_xy
    chunk_stride_x, chunk_stride_y = cfg.chunk_stride_xy

    # go through each pth file
    # iterate over all possible chunks with dimensions chunk_dims_xy in the XY dims
    # based on the coordinates in vtx_coords
    # split all the "vertex_*" properties into smaller pth files
    # add _1, _2 as suffix to split files and save them
    # add the new pth filename the new split list
    # create new split list file with new pth names
    # done
    split_list = read_txt_list(cfg.orig_list_path)
    # list of all chunk files
    out_split_list = []

    few_points, few_labels, few_instances = [], [], []

    # vertices or sampled points?
    prop_type = cfg.prop_type

    for scene_id in tqdm(split_list, desc='scene'):
        fname = f'{scene_id}.pth'
        pth_data = torch.load(Path(cfg.orig_pth_dir) / fname)

        # keep keys that start with prop type - vtx_ or sampled_
        keep_keys = [k for k in pth_data.keys() if k.startswith(prop_type)]
        coords = pth_data[f'{prop_type}coords']

        # get min and max of x and y
        # get number of chunks in x and y
        x_max = coords[:, 0].max()

        y_max = coords[:, 1].max()

        # chunk bounds
        chunks_bds = []

        # get the valid chunk boundaries
        x_start = 0
        while x_start < x_max:
            x_end = x_start + chunk_dim_x

            y_start = 0
            while y_start < y_max:
                y_end = y_start + chunk_dim_y
                chunks_bds.append([x_start, x_end, y_start, y_end])
                y_start += chunk_stride_y
            x_start += chunk_stride_x

        if len(chunks_bds) > 1:
            for chunk_ndx, chunk_bd in enumerate(tqdm(chunks_bds, desc='chunk', leave=False)):
                x_start, x_end, y_start, y_end = chunk_bd
                pth_data_chunk = {}
                # get all points within chunk
                keep_ndx = (coords[:, 0] >= x_start) & (coords[:, 0] < x_end) & (coords[:, 1] >= y_start) & (coords[:, 1] < y_end)

                out_scene_id = f'{scene_id}_{chunk_ndx}'

                # no points in chunk
                if keep_ndx.sum() == 0:
                    continue
                # keep only chunks with min #points
                if keep_ndx.sum() < cfg.n_points_threshold:
                    few_points.append(out_scene_id)
                    continue
                # keep only chunks with min #unique labels
                if len(torch.unique(torch.LongTensor(pth_data[f'{prop_type}labels'][keep_ndx]))) <= cfg.n_labels_threshold:
                    few_labels.append(out_scene_id)
                    continue

                # IDs of complete instances
                complete_instances = []
                # keep only almost-complete instances
                # go through each instance id in the chunk
                for instance_id in tqdm(torch.unique(torch.LongTensor(pth_data[f'{prop_type}instance_labels'][keep_ndx])).tolist(), desc='check_instance', leave=False):
                    # ignore invalid instance IDs
                    if instance_id == cfg.ignore_label or instance_id < 0: continue

                    # get the mask of instance inside the chunk
                    chunk_inst_mask = pth_data[f'{prop_type}instance_labels'][keep_ndx] == instance_id
                    # get the mask of instance in the full scene
                    scene_inst_mask = pth_data[f'{prop_type}instance_labels'] == instance_id

                    # if the fraction of points < cfg.instance_frac_threshold within the chunk
                    if (chunk_inst_mask.sum() / scene_inst_mask.sum()) >= cfg.instance_frac_threshold:
                        complete_instances.append(instance_id)

                # keep only chunks with min #complete instances
                if len(complete_instances) <= cfg.n_instances_threshold:
                    few_instances.append(out_scene_id)
                    continue

                # get the number of instances in this chunk
                # some of them might have been removed by the previous step
                # in case not checking for complete instances
                if len(torch.unique(torch.LongTensor(pth_data[f'{prop_type}instance_labels'][keep_ndx]))) <= cfg.n_instances_threshold:
                    few_instances.append(out_scene_id)
                    continue

                # keep all the vertex properties
                pth_data_chunk['scene_id'] = out_scene_id

                for k in keep_keys:
                    pth_data_chunk[k] = pth_data[k][keep_ndx]

                # keep instance IDs of only complete instances
                for instance_id in torch.unique(torch.LongTensor(pth_data_chunk[f'{prop_type}instance_labels'])).tolist():
                    if instance_id not in complete_instances:
                        inst_mask = pth_data_chunk[f'{prop_type}instance_labels'] == instance_id
                        pth_data_chunk[f'{prop_type}instance_labels'][inst_mask] = cfg.ignore_label
                
                out_fname = f'{out_scene_id}.pth'
                # save to new pth file
                torch.save(pth_data_chunk, out_pth_dir / out_fname)

                # add to new split list
                out_split_list.append(out_scene_id)
                        
        else:
            # no need to split
            # write pth data to new folder
            # add scene id as-is to new list
            out_split_list.append(scene_id)
            torch.save(pth_data, out_pth_dir / fname)

    print('Original scenes:', len(split_list))
    print('Output chunks:', len(out_split_list))

    # write scene list to new file
    with open(cfg.out_list_path, 'w') as f:
        for scene_id in out_split_list:
            f.write(scene_id + '\n')

    print('Total discarded chunks:', len(few_points) + len(few_labels) + len(few_instances))
    print('Few points:', len(few_points))
    print('Few labels:', len(few_labels))
    print('Few instances:', len(few_instances))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()

    main(args)