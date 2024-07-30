'''
usage: python -m semantic.prep.split_pth_data_parallel preprocess --config_file=semantic/configs/split_pth_data_train.yml

read each pth data
split all the vertex properties into smaller pth files
do this by splitting the scene in the XY axis into chunks
eg. 10m x 10m
pick all the points within these chunks and save to separate files
'''

from collections import defaultdict
from pathlib import Path
from common.utils.utils import read_txt_list
import torch
from common.file_io import load_yaml_munch

import multiprocessing
from fire import Fire
from joblib import Parallel, delayed
from loguru import logger


class BasePreprocessing:
    def __init__(self, config_file):
        self.cfg = load_yaml_munch(config_file)

        Path(self.cfg.out_list_path).parent.mkdir(parents=True, exist_ok=True)
        self.out_pth_dir = Path(self.cfg.out_pth_dir)
        self.out_pth_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_dim_x, self.chunk_dim_y = self.cfg.chunk_dims_xy
        self.chunk_stride_x, self.chunk_stride_y = self.cfg.chunk_stride_xy

        # done
        self.scene_ids = read_txt_list(self.cfg.orig_list_path)
        # list of all chunk files

        # vertices or sampled points?
        self.prop_type = self.cfg.prop_type
        self.n_jobs = -1

    @logger.catch
    def preprocess(self):
        self.n_jobs = (
            multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
        )

        logger.info(f"Tasks: {len(self.scene_ids)}")

        outputs = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self.process_file)(scene_id)
            for scene_id in self.scene_ids
        )

        output_dict = defaultdict(list)

        for output in outputs:
            # each one is a dict
            for key, val in output.items():
                output_dict[key] += val

        out_split_list = output_dict['output_chunks']

        # join all the chunk IDs together
        # write to file
        print('Original scenes:', len(self.scene_ids))
        print('Output chunks:', len(out_split_list))

        # write scene list to new file
        with open(self.cfg.out_list_path, 'w') as f:
            for scene_id in out_split_list:
                f.write(scene_id + '\n')

        for key in output_dict:
            print(f'{key}:', len(output_dict[key]))

    # process one scene id
    def process_file(self, scene_id):
        # write all the outputs here and return
        output_dict = defaultdict(list)

        fname = f'{scene_id}.pth'
        pth_data = torch.load(Path(self.cfg.orig_pth_dir) / fname)

        # keep keys that start with prop type - vtx_ or sampled_
        keep_keys = [k for k in pth_data.keys() if k.startswith(self.prop_type)]
        coords = pth_data[f'{self.prop_type}coords']

        # get min and max of x and y
        # get number of chunks in x and y
        x_max = coords[:, 0].max()

        y_max = coords[:, 1].max()

        # chunk bounds
        chunks_bds = []

        # get the valid chunk boundaries
        x_start = 0
        while x_start < x_max:
            x_end = x_start + self.chunk_dim_x

            y_start = 0
            while y_start < y_max:
                y_end = y_start + self.chunk_dim_y
                chunks_bds.append([x_start, x_end, y_start, y_end])
                y_start += self.chunk_stride_y
            x_start += self.chunk_stride_x

        if len(chunks_bds) > 1:
            for chunk_ndx, chunk_bd in enumerate(chunks_bds):
                x_start, x_end, y_start, y_end = chunk_bd
                pth_data_chunk = {}
                # get all points within chunk
                keep_ndx = (coords[:, 0] >= x_start) & (coords[:, 0] < x_end) & (coords[:, 1] >= y_start) & (coords[:, 1] < y_end)

                out_scene_id = f'{scene_id}_{chunk_ndx}'

                # no points in chunk
                if keep_ndx.sum() == 0:
                    continue
                # keep only chunks with min #points
                if keep_ndx.sum() < self.cfg.n_points_threshold:
                    output_dict['few_points'].append(out_scene_id)
                    continue
                # keep only chunks with min #unique labels
                if len(torch.unique(torch.LongTensor(pth_data[f'{self.prop_type}labels'][keep_ndx]))) <= self.cfg.n_labels_threshold:
                    output_dict['few_labels'].append(out_scene_id)
                    continue

                # IDs of complete instances
                complete_instances = []
                # keep only almost-complete instances
                # go through each instance id in the chunk
                for instance_id in torch.unique(torch.LongTensor(pth_data[f'{self.prop_type}instance_labels'][keep_ndx])).tolist():
                    # ignore invalid instance IDs
                    if instance_id == self.cfg.ignore_label or instance_id < 0: continue

                    # get the mask of instance inside the chunk
                    chunk_inst_mask = pth_data[f'{self.prop_type}instance_labels'][keep_ndx] == instance_id
                    # get the mask of instance in the full scene
                    scene_inst_mask = pth_data[f'{self.prop_type}instance_labels'] == instance_id

                    # if the fraction of points < cfg.instance_frac_threshold within the chunk
                    if (chunk_inst_mask.sum() / scene_inst_mask.sum()) >= self.cfg.instance_frac_threshold:
                        complete_instances.append(instance_id)

                # keep only chunks with min #complete instances
                if len(complete_instances) <= self.cfg.n_instances_threshold:
                    output_dict['few_instances'].append(out_scene_id)
                    continue

                # get the number of instances in this chunk
                # some of them might have been removed by the previous step
                # in case not checking for complete instances
                if len(torch.unique(torch.LongTensor(pth_data[f'{self.prop_type}instance_labels'][keep_ndx]))) <= self.cfg.n_instances_threshold:
                    output_dict['few_instances'].append(out_scene_id)
                    continue

                # keep all the vertex properties
                pth_data_chunk['scene_id'] = out_scene_id

                for k in keep_keys:
                    pth_data_chunk[k] = pth_data[k][keep_ndx]

                # keep instance IDs of only complete instances
                for instance_id in torch.unique(torch.LongTensor(pth_data_chunk[f'{self.prop_type}instance_labels'])).tolist():
                    if instance_id not in complete_instances:
                        inst_mask = pth_data_chunk[f'{self.prop_type}instance_labels'] == instance_id
                        pth_data_chunk[f'{self.prop_type}instance_labels'][inst_mask] = self.cfg.ignore_label
                
                out_fname = f'{out_scene_id}.pth'
                # save to new pth file
                torch.save(pth_data_chunk, self.out_pth_dir / out_fname)

                # add to new split list
                output_dict['output_chunks'].append(out_scene_id)
        else:
            # no need to split
            # write pth data to new folder
            # add scene id as-is to new list
            output_dict['output_chunks'].append(scene_id)
            torch.save(pth_data, self.out_pth_dir / fname)

        return output_dict

if __name__ == "__main__":
    Fire(BasePreprocessing)