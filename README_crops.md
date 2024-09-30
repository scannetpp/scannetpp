# Object Crops

## Setup
Follow the main README for installation. Also install [renderpy](https://github.com/liu115/renderpy).   
Install [SAM2](https://github.com/facebookresearch/segment-anything-2/tree/main) to refine masks.

## Usage
Make sure to extract the RGB frames for your scenes. Update the config in `iphone/configs/prepare_iphone_data.yml`.

```
python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
```

Then extract object crops with the following command. Update the config in `common/configs/render.yml`.

```
python3 -m common.render_crops common/configs/render.yml
```

Example `common/configs/render.yml`.
```
# folder where the data is downloaded
data_root: /data/concept-graphs/scannetpp

# Set True to render depth for iphone frames
render_iphone: True
# Set True to render depth for dslr frames
render_dslr: False

splits: [nvs_sem_train, nvs_sem_val]

# Specify scene ids if you want to render depth for specific scenes
scene_ids: [c0f5742640]

# The near and far planes for the depth camera during rendering in meters.
near: 0.05
far: 20.0

# Output directory for the rendered depth images. If not given, the output will be saved to data folder in data_root
output_dir: /data/concept-graphs/scannetpp/data

# SAM2 checkpoints directory and config
sam2_checkpoint_dir: "/path/to/checkpoints/sam2.1_hiera_large.pt"
sam2_model_cfg: "configs/sam2.1/sam2.1_hiera_l.yaml"
```

Crops are saved at `output_dir/scene_id/iphone/render_crops`.   
Refined crops after running SAM2 are saved at `output_dir/scene_id/iphone/render_crops_sam2`.
