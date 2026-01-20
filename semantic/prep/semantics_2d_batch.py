'''
queue a batch of gen_caption jobs
'''

from pathlib import Path
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from common.file_io import read_txt_list
from scannetpp_tools.utils.job_utils import slurm_hydra_job


@hydra.main(version_base=None, config_path="../configs", config_name="semantics_2d")
def main(cfg : DictConfig) -> None:
    print('Config:', cfg)

    n_jobs, n_scenes_queued = 0, 0

    # get the scene ids to be done
    if cfg.get('scene_list'):
        scene_ids_to_be_done = cfg.scene_list
    elif cfg.get('scene_list_file'):
        scene_ids_to_be_done = read_txt_list(cfg.scene_list_file)
    else:
        raise ValueError('Either scene_list_file or scene_list must be provided')

    print('Num scenes to be done:', len(scene_ids_to_be_done))

    # group scene ids into groups based on cfg.scenes_per_job
    scene_groups = [scene_ids_to_be_done[i:i+cfg.scenes_per_job] for i in range(0, len(scene_ids_to_be_done), cfg.scenes_per_job)]
    print('Num scene groups:', len(scene_groups))


    for scene_id_group in tqdm(scene_groups, desc='scene_group'):
        # create job file and queue the job
        hydra_options = {
            # specify the scenes to run on 
            'filter_scenes': scene_id_group,
            # overrides from shell script
            'rasterout_dir': cfg.rasterout_dir,
            'visiblity_cache_dir': cfg.visiblity_cache_dir,
            'subsample_factor': cfg.subsample_factor,
            'scene_list_file': cfg.scene_list_file,
            'visibility_topk': cfg.visibility_topk,
            'save_dir_root': cfg.save_dir_root,
            'process_each_object': cfg.process_each_object,
            'save_obj_crop': cfg.save_obj_crop,
            'save_obj_crop_mask': cfg.save_obj_crop_mask,
        }

        n_scenes_queued += len(scene_id_group)

        if cfg.short_filename:
            job_name = cfg.job_name
        else:
            job_name = cfg.job_name + '_' + '_'.join(scene_id_group)

        if cfg.dry_run:
            print(f'[DRY RUN] Would queue job for {scene_id_group} with job name {job_name}')
        else:
            print(f'Queuing job for {scene_id_group} with job name {job_name}')
            slurm_hydra_job(cfg.script_path, hydra_options, cfg.job_dir,
                            job_name=job_name,
                            mem_gb=cfg.mem_gb, time_h=cfg.time_h,
                            slurm_extra_lines=cfg.slurm_extra_lines,
                            extra_pre_commands=cfg.extra_pre_commands)
            
        n_jobs += 1

        if cfg.limit_jobs is not None and n_jobs == cfg.limit_jobs:
            print('Reached job limit:', n_jobs)
            break

    if cfg.dry_run:
        print(f'[DRY RUN] Would queue {n_jobs} jobs with {n_scenes_queued} scenes')
    else:
        print(f'Queued {n_jobs} jobs with {n_scenes_queued} scenes')
    
if __name__ == "__main__":
    main()