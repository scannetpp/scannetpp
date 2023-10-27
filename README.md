ScanNet++ Toolkit
-----------------

## Requirements
```
conda create -n scannetpp python=3.10
conda activate scannetpp
pip install -r requirements.txt
```

# DSLR

## Undistortion
Undistort DSLR images (and masks) based on COLMAP so that the output images are pinhole camera models.

You will need [COLMAP](https://colmap.github.io/) installed to run this script.

Insert `data_root` and `output_dir` in `dslr/configs/undistort_dslr.yml` and run:
```
python -m dslr.undistort_dslr dslr/configs/undistort_dslr.yml
```
The output will be saved in `output_dir` with the following structure:
```
output_dir/SCENE_ID
├── colmap
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt
├── images
├── masks
└── nerfstudio/transforms.json
```

# Render Depth for DSLR and iPhone

Install the python package from https://github.com/liu115/renderpy in addtion to the requirements.

```
python -m common.render common/configs/render.yml
```
The output will be saved in `output_dir` with the following structure:
```
output_dir/SCENE_ID/[dslr, iphone]
├── render_rgb
└── render_depth
```
The rendered depth maps are single-channel uint16 png, where the unit is mm and 0 means invalid depth.

## Render Semantics (coming soon)

# iPhone
## Extract RGB frames, masks and depth frames
```
python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
```


## Evaluation
The evaluation script here is the same that runs on the benchmark server. Therefore, it's highly encouraged to run the evaluation script before submitting the results (on the val set) to the benchmark server.

### Novel View Synthesis
The results should be saved in the following structure:
```
SCENE_ID0/
├── DSC00001.JPG
├── DSC00002.JPG
├── ...
SCENE_ID1/
├── ...
```

```

```