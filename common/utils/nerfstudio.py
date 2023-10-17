from typing import List, Dict, Any, Tuple
import json

from pathlib import Path
import numpy as np
from common.utils.colmap import Camera, Image, read_model


def convert_camera(camera: Camera) -> Dict[str, Any]:
    camera_params = camera.params
    out = {
       "fl_x": float(camera_params[0]),
        "fl_y": float(camera_params[1]),
        "cx": float(camera_params[2]),
        "cy": float(camera_params[3]),
        "w": camera.width,
        "h": camera.height,
    }

    if camera.model == "OPENCV_FISHEYE":
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "k3": float(camera_params[6]),
                "k4": float(camera_params[7]),
                "camera_model": "OPENCV_FISHEYE",
            }
        )
    elif camera.model == "OPENCV":
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "p1": float(camera_params[6]),
                "p2": float(camera_params[7]),
                "camera_model": "OPENCV",
            }
        )
    else:
        out.update(
            {
                "camera_model": "PINHOLE",
            }
        )
    return out


def convert_frames(images: Dict[int, Image]) -> List[Dict[str, Any]]:
    frames = []
    for image_id, image in images.items():
        w2c = image.world_to_camera
        c2w = np.linalg.inv(w2c)

        # Convert from COLMAP's camera coordinate system to nerfstudio/instant-ngp
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        image_name = image.name.split("/")[-1]
        frame = {
            "file_path": image_name,
            "transform_matrix": c2w.tolist(),
        }

        frames.append(frame)
    return frames



def prepare_transforms_json(
    model_path: Path,
    out_path: Path,
    train_list: List[str],
    test_list: List[str],
    has_mask: bool = False,
):
    cameras, images, points3D = read_model(model_path, ".txt")
    assert len(cameras) == 1, "Multiple cameras not supported"
    camera = next(iter(cameras.values()))
    data = convert_camera(camera)
    frame_data = convert_frames(images)
    train_frames = []
    test_frames = []

    for frame in frame_data:
        fn = frame["file_path"]
        if has_mask:
            frame["mask_path"] = fn.replace(".JPG", ".png")
        if fn in train_list:
            train_frames.append(frame)
        elif fn in test_list:
            test_frames.append(frame)

    data["frames"] = train_frames
    data["test_frames"] = test_frames

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(json.dumps(data, indent=4))
