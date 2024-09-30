import argparse
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2

import numpy as np

from common.utils.utils import load_yaml_munch, read_txt_list
from common.render_crops_utils import plot_grid_images
from common.sam2_models import SAM2ImageMaskModel

# Global variables for the modes and points
current_mode = "r"  # 'a' for addition, 'x' for subtraction, 'r' for reset/no mode
current_points = []  # List to store clicked points in the format [(x, y)]
current_labels = []  # Store the labels (1 or 0) for each point
current_obj_id = None  # Global variable to track the current object ID


def load_crops_data(data_dir):
    crops_data = dict()

    rgb_dir = data_dir / "rgbs"
    mask_dir = data_dir / "masks"
    metadata_dir = data_dir / "metadata"

    metadata_files = natsorted(metadata_dir.glob("*.json"))

    for metadata_file in metadata_files:
        obj_id = metadata_file.stem.split("_")[0]
        obj_id_int = int(obj_id)

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        crops_data[obj_id_int] = {
            "rgbs": [],
            "masks": [],
            "scores": metadata["scores"],
            "label": metadata["label"],
        }

        rgb_files = natsorted(rgb_dir.glob(f"{obj_id}_*_rgb.png"))
        mask_files = natsorted(mask_dir.glob(f"{obj_id}_*_mask.npy"))

        for rgb_file, mask_file in zip(rgb_files, mask_files):
            rgb_image = cv2.imread(str(rgb_file))
            rgb_image = cv2.cvtColor(
                rgb_image, cv2.COLOR_BGR2RGB
            )  # Convert to RGB format

            mask = np.load(mask_file)

            crops_data[obj_id_int]["rgbs"].append(rgb_image)
            crops_data[obj_id_int]["masks"].append(mask)

    return crops_data


def save_masks_data_only(crops_data, save_dir, obj_id=None):
    """
    Save mask data for the obj_id if given, otherwise save all masks.
    """

    mask_dir = save_dir / "masks"
    os.makedirs(mask_dir, exist_ok=True)

    if obj_id is not None:
        data = crops_data[obj_id]
        padded_obj_id = str(obj_id).zfill(5)

        for crop_id, mask in enumerate(data["masks"]):
            mask_filename = os.path.join(
                mask_dir, f"{padded_obj_id}_{crop_id}_mask.npy"
            )
            np.save(mask_filename, mask)

        return

    print("Saving all masks data...")
    for obj_id, data in crops_data.items():
        padded_obj_id = str(obj_id).zfill(5)

        for crop_id, (mask) in enumerate(data["masks"]):
            mask_filename = os.path.join(
                mask_dir, f"{padded_obj_id}_{crop_id}_mask.npy"
            )
            np.save(mask_filename, mask)

    return


def save_render_plot(crops_data, save_dir, obj_id):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rgbs = crops_data[obj_id]["rgbs"]
    masks = crops_data[obj_id]["masks"]
    label = crops_data[obj_id]["label"]

    plot_grid_images(rgbs, masks, grid_width=len(rgbs), title=label)
    plt.savefig(save_dir / f"{str(obj_id).zfill(5)}.jpg")
    plt.close()


def save_all(sam2_image_model, current_obj_id, data_save_dir, plot_save_dir):
    """Save all masks and plots for the current object."""
    global crops_data

    crops_data[current_obj_id]["masks"] = sam2_image_model.masks

    save_masks_data_only(crops_data, data_save_dir, obj_id=current_obj_id)
    save_render_plot(crops_data, plot_save_dir, current_obj_id)


def get_required_ids(obj_ids, frame_ids):
    print(f"Available object IDs: {', '.join(map(str, obj_ids))}")

    while True:
        required_obj_id = input("Enter the required object ID (number): ").strip()

        if not required_obj_id.isdigit():
            print("Invalid input. Please enter a valid number.")
            continue

        required_obj_id = int(required_obj_id)

        if required_obj_id in obj_ids:
            break
        else:
            print(
                f"Invalid object ID. Please select a valid ID from: {', '.join(map(str, obj_ids))}"
            )

    print(f"Available frame IDs: {', '.join(map(str, frame_ids))}")

    while True:
        required_frame_id = input("Enter the required frame ID (number): ").strip()

        if not required_frame_id.isdigit():
            print("Invalid input. Please enter a valid number.")
            continue

        required_frame_id = int(required_frame_id)

        if required_frame_id in frame_ids:
            return required_obj_id, required_frame_id
        else:
            print(
                f"Invalid frame ID. Please select a valid ID from: {', '.join(map(str, frame_ids))}"
            )


def reset_state():
    """Reset mode, points, and labels."""

    global current_mode, current_points, current_labels
    current_mode = "r"
    current_points = []
    current_labels = []


def load_next_object(
    obj_ids, next_idx, frame_idx, sam2_image_model, data_save_dir, plot_save_dir
):
    """Load the next object and start interactive editor."""

    interactive_mask_editor(
        sam2_image_model=sam2_image_model,
        obj_ids=obj_ids,
        current_idx=next_idx,
        data_save_dir=data_save_dir,
        plot_save_dir=plot_save_dir,
        frame_idx=frame_idx,
    )


def proceed(sam2_image_model, data_save_dir, plot_save_dir, obj_ids, current_idx):
    """Move to the next object."""
    reset_state()
    sam2_image_model.cleanup()

    frame_ids = list(range(0, 4))  # Assuming 4 frames per object
    required_obj_id, frame_idx = get_required_ids(obj_ids, frame_ids)

    # Load the next required object
    next_idx = obj_ids.index(required_obj_id)

    if next_idx < len(obj_ids):
        plt.close()
        load_next_object(
            obj_ids,
            next_idx,
            frame_idx,
            sam2_image_model,
            data_save_dir,
            plot_save_dir,
        )
    else:
        print("All objects have been processed.")
        plt.close()


def on_click(event, sam2_image_model, mask_overlay, frame_idx):
    global current_points, current_labels

    if (
        current_mode in ["a", "x"]
        and event.xdata is not None
        and event.ydata is not None
    ):
        # Register the clicked point and label based on mode
        x, y = int(event.xdata), int(event.ydata)
        current_points.append((x, y))
        current_labels.append(1 if current_mode == "a" else 0)

        # Update mask with the new points
        points, labels = np.array(current_points), np.array(current_labels)
        refined_mask, _ = sam2_image_model.refine_mask_w_points_prompt(
            points, labels, frame_idx
        )

        # Update the mask overlay with the refined mask
        mask_overlay.set_data(refined_mask)
        plt.draw()

        # # Reset current points and labels after updating the mask
        # current_points = []
        # current_labels = []

    elif current_mode == "r":
        print(
            "No mode selected. Switch to addition or subtraction mode first (press 'a' or 'x')."
        )


def on_key(event, sam2_image_model, data_save_dir, plot_save_dir, obj_ids, current_idx):
    global current_mode, current_points, current_labels, current_obj_id

    # Switch modes or handle events based on key presses
    if event.key == "a":
        current_mode = "a"
        print("Switched to addition mode")

    elif event.key == "x":
        current_mode = "x"
        print("Switched to subtraction mode")

    elif event.key == "r":
        reset_state()
        print("Reset to no mode and cleared points/labels")

    elif event.key == "enter":
        save_all(sam2_image_model, current_obj_id, data_save_dir, plot_save_dir)
        plt.close()  # Exit editor on pressing 'enter'

    elif event.key == "n":
        save_all(sam2_image_model, current_obj_id, data_save_dir, plot_save_dir)
        proceed(sam2_image_model, data_save_dir, plot_save_dir, obj_ids, current_idx)


def get_data_from_crops(crops_data, obj_id):
    rgbs = crops_data[obj_id]["rgbs"]
    masks = crops_data[obj_id]["masks"]
    scores = crops_data[obj_id]["scores"]
    label = crops_data[obj_id]["label"]

    return rgbs, masks, scores, label


def interactive_mask_editor(
    sam2_image_model,
    obj_ids,
    current_idx,
    data_save_dir,
    plot_save_dir,
    frame_idx=0,
):
    global current_obj_id
    current_obj_id = obj_ids[current_idx]

    rgbs, masks, scores, label = get_data_from_crops(crops_data, current_obj_id)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    img = rgbs[frame_idx]

    # Top subplot: RGB image
    ax1.imshow(img)
    ax1.axis("off")

    # Bottom subplot: RGB image with mask overlay
    mask = masks[frame_idx]
    img_overlay = ax2.imshow(img)
    mask_overlay = ax2.imshow(mask, cmap="jet", alpha=0.5)
    ax2.axis("off")

    # Initialize the SAM2 model with the provided RGBs and masks
    sam2_image_model.store_data(rgbs, masks, scores)

    # Connect events to handlers
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: on_click(event, sam2_image_model, mask_overlay, frame_idx),
    )
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key(
            event, sam2_image_model, data_save_dir, plot_save_dir, obj_ids, current_idx
        ),
    )

    plt.suptitle(
        f"Interactive Mask Editor - Label: {label} (Object ID: {current_obj_id})"
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def main(args):
    global crops_data

    cfg = load_yaml_munch(args.config_file)

    if cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)

    output_dir = cfg.get("output_dir")
    if output_dir is None:
        output_dir = Path(cfg.data_root) / "data"
    output_dir = Path(output_dir)

    render_devices = []
    if cfg.get("render_dslr", False):
        render_devices.append("dslr")
        raise Exception("This code has not been tested with the DSLR data.")
    if cfg.get("render_iphone", False):
        render_devices.append("iphone")

    # Assuming one scene ID and one device for simplicity
    scene_id = scene_ids[0]
    device = render_devices[0]

    # Set path for crops data
    crops_data_dir = output_dir / scene_id / device / "crops_data"
    crops_data = load_crops_data(crops_data_dir)
    obj_ids = list(crops_data.keys())
    frame_ids = list(range(0, 4))  # Assuming 4 frames per object

    required_obj_id, required_frame_id = get_required_ids(obj_ids, frame_ids)

    # Initialize the SAM2 video model with checkpoint and configuration
    sam2_checkpoint = cfg.get("sam2_checkpoint_dir", None)
    sam2_model_cfg = cfg.get("sam2_model_cfg", None)
    if sam2_checkpoint is None or sam2_model_cfg is None:
        raise Exception("Please provide sam2_checkpoint_dir and sam2_model_cfg")

    sam2_image_model = SAM2ImageMaskModel(
        sam2_checkpoint, sam2_model_cfg, num_points=3, ransac_iterations=5
    )

    # Directory to save rendered crops
    plot_save_dir = crops_data_dir.parent / "render_crops_manual"

    # Start the interactive mask editor with the first object
    starting_idx = obj_ids.index(required_obj_id)
    frame_idx = required_frame_id
    interactive_mask_editor(
        sam2_image_model=sam2_image_model,
        obj_ids=obj_ids,
        current_idx=starting_idx,
        frame_idx=frame_idx,
        data_save_dir=crops_data_dir,
        plot_save_dir=plot_save_dir,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "config_file", help="Path to config file", default="configs/render_crops.yaml"
    )
    args = p.parse_args()

    main(args)
