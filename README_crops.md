# Object Crops

## Setup
First install PyTorch corresponding to your local cuda-toolkit version from [here](https://pytorch.org/get-started/locally/).   
Follow the main README for installation. Also install [renderpy](https://github.com/liu115/renderpy).   
Install [SAM2](https://github.com/facebookresearch/segment-anything-2/tree/main) to refine masks.

## Usage
Make sure to extract the RGB frames for your scenes. Update the config in `iphone/configs/prepare_iphone_data.yml`.

```bash
python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
```

Then extract object crops with the following command. Update the config in `common/configs/render.yml`.

```bash
python3 -m common.render_crops common/configs/render.yml
```

Example `common/configs/render.yml`.
```yaml
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
sam2_checkpoint_dir: /path/to/checkpoints/sam2.1_hiera_large.pt
sam2_model_cfg: configs/sam2.1/sam2.1_hiera_l.yaml
```

Crops are saved at `output_dir/scene_id/iphone/render_crops`.   
Refined crops after running SAM2 are saved at `output_dir/scene_id/iphone/render_crops_sam2`.

## Interactive Mask Editors

There are two interactive editors to visualize and refine object masks:

1. **SAM2 Video Predictor Editor**: For refining one frame of each object and using SAM2's video predictor to propagate masks to other frames.
2. **SAM2 Image Predictor Editor**: For refining masks for individual frames of each object.

To run the interactive editor using the SAM2 video predictor, use:

```bash
python3 -m common.interactive_editor common/configs/render.yml
```

To run the interactive editor using the SAM2 image predictor, use :

```bash
python3 -m common.interactive_editor_image common/configs/render.yml
```

### Editor Modes and Controls

- **`a`**: Switch to addition mode. Use this mode to click on the images and add foreground points for refining the mask.
- **`x`**: Switch to subtraction mode. Use this mode to click on the images and remove background points from the mask.
- **`r`**: Reset mode. Clears all points and labels. 
  > Note: Reset mode does not clear the mask currently being displayed, but you can proceed with adding points from scratch.
- **`enter`**: Save the refined mask for the current object and exit the editor.
- **`n`**: Save the refined mask and proceed to the next object.

### Additional Information

- The initial mode for the editor is set to **reset** mode.
- Whenever you move to the next object using the `n` key, the mode is also set back to **reset**.
- Keep an eye out for user input prompts in the terminal when using the mask editor.
- Either of the images seen in the editor can be clicked on to add or remove points based on the current mode.
- Visualizations of the manually refined crops are saved in the `output_dir/scene_id/iphone/render_crops_manual` directory.
