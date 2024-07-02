# Render RGB, Depth, Semantics for DSLR and iPhone
# python -m common.render common/configs/render.yml

# Extract RGB frames, masks and depth frames from iPhone videos
# python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml

# Undistortion: convert fisheye images to pinhole with OpenCV
# python -m dslr.undistort dslr/configs/undistort.yml

# Downscale the DSLR images
# python -m dslr.downscale dslr/configs/downscale.yml

# -----Semantics-----
# Prepare 3D Semantics Training Data
# python -m semantic.prep.prepare_training_data semantic/configs/prepare_training_data.yml
# Split PTH files into chunks for training
# python -m semantic.prep.split_pth_data semantic/configs/split_pth_data_train.yml
# Visualize training data
# python -m semantic.viz.viz_pth_data semantic/configs/viz_pth_data.yml
# Prepare Semantic/Instance Ground Truth Files for Evaluation
# python -m semantic.prep.prepare_semantic_gt semantic/configs/prepare_semantic_gt.yml
# Rasterize 3D Semantics onto 2D Images
# python -m semantic.prep.rasterize_semantics_2d semantic/configs/rasterize_semantics_2d.yml