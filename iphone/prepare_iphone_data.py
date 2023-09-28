
'''
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
'''

import argparse
from pathlib import Path
import yaml
from munch import Munch
from tqdm import tqdm
import json
import sys
import subprocess
import zlib
import numpy as np
import imageio as iio
import lz4.block

from common.scene_release import ScannetppScene_Release

def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()

    return lines

def load_json(path):
    with open(path) as f:
        j = json.load(f)

    return j

def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)

def run_command(cmd: str, verbose=False):
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        if out.stderr is not None:
            print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out

def extract_rgb(scene):
    scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {scene.iphone_video_path} -start_number 0 -q:v 1 {scene.iphone_rgb_dir}/frame_%06d.jpg"
    run_command(cmd, verbose=True)

def extract_masks(scene):
    scene.iphone_video_mask_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {str(scene.iphone_video_mask_path)} -pix_fmt gray -start_number 0 {scene.iphone_video_mask_dir}/frame_%06d.png"
    run_command(cmd, verbose=True)

def extract_depth(scene):
    # global compression with zlib
    height, width = 192, 256
    sample_rate = 1
    scene.iphone_depth_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(scene.iphone_depth_path, 'rb') as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in tqdm(range(0, depth.shape[0], sample_rate), desc='decode_depth'):
            iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", (depth * 1000).astype(np.uint16))
    # per frame compression with lz4/zlib
    except:
        frame_id = 0
        with open(scene.iphone_depth_path, 'rb') as infile:
            while True:
                size = infile.read(4)   # 32-bit integer
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder='little')
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                # read the whole file
                data = infile.read(size)
                try:
                    # try using lz4
                    data = lz4.block.decompress(data, uncompressed_size=height * width * 2)  # UInt16 = 2bytes
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                except:
                    # try using zlib
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                    depth = (depth * 1000).astype(np.uint16)

                # 6 digit frame id = 277 minute video at 60 fps
                iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", depth)
                frame_id += 1

def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get the scenes to process
    if cfg.get('scene_ids'):
        scene_ids = cfg.scene_ids
    elif cfg.get('splits'):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / 'splits' / f'{split}.txt'
            scene_ids += read_txt_list(split_path)

    # get the options to process
    # go through each scene
    for scene_id in tqdm(scene_ids, desc='scene'):
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / 'data')

        if cfg.extract_rgb:
            extract_rgb(scene)

        if cfg.extract_masks:
            extract_masks(scene)

        if cfg.extract_depth:
            extract_depth(scene)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()

    main(args)