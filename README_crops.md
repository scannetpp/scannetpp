# Object Crops

## Setup
Follow the main README for installation. Also install [renderpy](https://github.com/liu115/renderpy).

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

# This needs to be set
output_dir: /data/concept-graphs/scannetpp/data
```

Crops are saved at `output_dir/scene_id/iphone/render_crops`.