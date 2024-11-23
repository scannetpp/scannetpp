import cv2
import numpy as np

def compute_undistort_intrinsic(K, height, width, distortion_params):
    assert len(distortion_params.shape) == 1
    assert distortion_params.shape[0] == 4  # OPENCV_FISHEYE has k1, k2, k3, k4

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        distortion_params,
        (width, height),
        np.eye(3),
        balance=0.0,
    )
    # Make the cx and cy to be the center of the image
    new_K[0, 2] = width / 2.0
    new_K[1, 2] = height / 2.0
    return new_K