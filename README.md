ScanNet++ Toolkit
-----------------

## Requirements
```
conda create -n scannetpp python=3.10
conda activate scannetpp
pip install -r requirements.txt
```
# DSLR Undistortion

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

# iPhone
## Extract RGB frames, masks and depth frames
```
python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
```
