import cv2
import numpy as np

def crop_undistorted_dslr_image(undistorted_image, crop_factor):
    '''
    crop the undist. dslr on left and right side by crop_factor
    image: HW3
    crop_factor: [lr_factor, tb_factor]

    return: cropped image
    '''
    lr_factor, tb_factor = crop_factor
    h, w = undistorted_image.shape[:2]
    lr_crop = int(w * lr_factor)
    tb_crop = int(h * tb_factor)
    # HWC -> hwC or HW-> hw
    if len(undistorted_image.shape) == 3:
        cropped_image = undistorted_image[tb_crop:h-tb_crop, lr_crop:w-lr_crop, :]
    else:
        cropped_image = undistorted_image[tb_crop:h-tb_crop, lr_crop:w-lr_crop]
    return cropped_image

def adjust_intrinsic_matrix(intrinsic, factor):
    # divide fx, fy, cx, cy by factor
    intrinsic /= factor
    intrinsic[2, 2] = 1
    return intrinsic

def get_undistort_maps(intrinsic, distort_params, undistort_intrinsic, img_height, img_width):
    undistort_map1, undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
            intrinsic, distort_params, np.eye(3), undistort_intrinsic, (img_width, img_height), cv2.CV_32FC1
        )
    return undistort_map1, undistort_map2

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