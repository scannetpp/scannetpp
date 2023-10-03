ScanNet++ Toolkit
-----------------

## Requirements
```
conda create -n scannetpp python=3.10
conda activate scannetpp
pip install -r requirements.txt
```

# iPhone
## Extract RGB frames, masks and depth frames
```
python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
```
