import argparse
import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from natsort import natsorted
import cv2

import imageio
import numpy as np
from tqdm import tqdm

from utils.utils import load_yaml_munch, read_txt_list
from render_crops_utils import plot_grid_images
from sam2_models import SAM2ImageMaskModel, SAM2VideoMaskModel

# Global variables for the modes and points
current_mode = "r"  # 'a' for addition, 's' for subtraction, 'r' for reset/no mode
current_points = []  # List to store clicked points in the format [(x, y)]
current_labels = []  # Store the labels (1 or 0) for each point
current_obj_id = None  # Global variable to track the current object ID


def load_refined_crops(data_dir):
    refined_crops = dict()

    rgb_dir = data_dir / "rgbs"
    mask_dir = data_dir / "masks"
    metadata_dir = data_dir / "metadata"

    metadata_files = natsorted(metadata_dir.glob("*.json"))

    for metadata_file in metadata_files:
        obj_id = metadata_file.stem.split("_")[0]
        obj_id_int = int(obj_id)

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        refined_crops[obj_id_int] = {
            "rgbs": [],
            "refined_masks": [],
            "scores": metadata["scores"],
            "label": metadata["label"],
        }

        rgb_files = natsorted(rgb_dir.glob(f"{obj_id}_*_rgb.png"))
        mask_files = natsorted(mask_dir.glob(f"{obj_id}_*_mask.npy"))

        for rgb_file, mask_file in zip(rgb_files, mask_files):
            rgb_image = cv2.imread(str(rgb_file))
            rgb_image = cv2.cvtColor(
                rgb_image, cv2.COLOR_BGR2RGB
            )  # Convert back to RGB format

            mask = np.load(mask_file)

            refined_crops[obj_id_int]["rgbs"].append(rgb_image)
            refined_crops[obj_id_int]["refined_masks"].append(mask)

    return refined_crops


def save_masks_data_only(refined_crops, save_dir, pad_length=3, obj_id=None):
    # Create separate directories for RGBs, masks, and metadata
    mask_dir = save_dir / "masks"

    os.makedirs(mask_dir, exist_ok=True)

    if obj_id is not None:
        data = refined_crops[obj_id]
        padded_obj_id = str(obj_id).zfill(pad_length)

        for crop_id, mask in enumerate(data["refined_masks"]):
            mask_filename = os.path.join(
                mask_dir, f"{padded_obj_id}_{crop_id}_mask.npy"
            )
            np.save(mask_filename, mask)

        return

    # Iterate through each object in the refined crops dictionary
    for obj_id, data in refined_crops.items():
        # Pad the obj_id with zeros up to the specified length
        padded_obj_id = str(obj_id).zfill(pad_length)

        for crop_id, (mask) in enumerate(data["refined_masks"]):
            mask_filename = os.path.join(
                mask_dir, f"{padded_obj_id}_{crop_id}_mask.npy"
            )
            np.save(mask_filename, mask)

    return


def save_render_plot(rgbs, manual_refined_masks, save_dir, obj_id, label):
    # Save the refined masks and plot the images
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save the images
    plot_grid_images(rgbs, manual_refined_masks, grid_width=len(rgbs), title=label)
    plt.savefig(save_dir / f"{str(obj_id).zfill(5)}.jpg")
    plt.close()


def save_all(sam2_video_model, current_obj_id, data_save_dir, plot_save_dir):
    """Save all masks and plots for the current object."""
    global refined_crops

    refined_crops[current_obj_id]["refined_masks"] = sam2_video_model.masks

    save_masks_data_only(refined_crops, data_save_dir, obj_id=current_obj_id)
    save_render_plot(
        refined_crops[current_obj_id]["rgbs"],
        refined_crops[current_obj_id]["refined_masks"],
        plot_save_dir,
        current_obj_id,
        refined_crops[current_obj_id]["label"],
    )


def reset_state():
    """Reset mode, points, and labels."""
    global current_mode, current_points, current_labels
    current_mode = "r"
    current_points = []
    current_labels = []


def load_next_object(obj_ids, next_idx, sam2_video_model, data_save_dir, plot_save_dir):
    """Load the next object and start interactive editor."""
    interactive_mask_editor(
        rgbs=refined_crops[obj_ids[next_idx]]["rgbs"],
        refined_masks=refined_crops[obj_ids[next_idx]]["refined_masks"],
        scores=refined_crops[obj_ids[next_idx]]["scores"],
        sam2_video_model=sam2_video_model,
        label=refined_crops[obj_ids[next_idx]]["label"],
        obj_ids=obj_ids,
        current_idx=next_idx,
        data_save_dir=data_save_dir,
        plot_save_dir=plot_save_dir,
    )


def proceed(sam2_video_model, data_save_dir, plot_save_dir, obj_ids, current_idx):
    """Move to the next object."""
    reset_state()

    # Load the next object, if available
    next_idx = current_idx + 1
    if next_idx < len(obj_ids):
        plt.close()
        load_next_object(
            obj_ids, next_idx, sam2_video_model, data_save_dir, plot_save_dir
        )
    else:
        print("All objects have been processed.")
        plt.close()


def on_click(event, mask_overlay, sam2_video_model, frame_idx):
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
        sam2_video_model.set_state_and_refine_masks_w_manual_prompt(
            points, labels, frame_idx
        )
        sam2_video_model.unpad_masks_to_original_size()

        rgbs = sam2_video_model.rgbs
        manual_refined_masks = sam2_video_model.masks_refined
        scores = sam2_video_model.scores

        # Update the mask overlay with the refined mask
        mask_overlay.set_data(manual_refined_masks[frame_idx])
        plt.draw()

        # Reset the model and reinitialize with the updated masks
        sam2_video_model.cleanup()
        sam2_video_model.pad_and_store(rgbs, manual_refined_masks, scores)
    elif current_mode == "r":
        print(
            "No mode selected. Switch to addition or subtraction mode first (press 'a' or 'x')."
        )


def on_key(event, sam2_video_model, data_save_dir, plot_save_dir, obj_ids, current_idx):
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
        save_all(sam2_video_model, current_obj_id, data_save_dir, plot_save_dir)
        plt.close()  # Exit editor on pressing 'enter'
    elif event.key == "n":
        save_all(sam2_video_model, current_obj_id, data_save_dir, plot_save_dir)
        proceed(sam2_video_model, data_save_dir, plot_save_dir, obj_ids, current_idx)


def interactive_mask_editor(
    rgbs,
    refined_masks,
    scores,
    sam2_video_model,
    label,
    obj_ids,
    current_idx,
    data_save_dir,
    plot_save_dir,
    frame_idx=0,
):
    global current_obj_id
    current_obj_id = obj_ids[current_idx]  # Set current object ID

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    img = rgbs[frame_idx]

    # Top subplot: RGB image
    ax1.imshow(img)
    ax1.axis("off")

    # Bottom subplot: RGB image with mask overlay
    mask = refined_masks[frame_idx]
    img_overlay = ax2.imshow(img)
    mask_overlay = ax2.imshow(mask, cmap="jet", alpha=0.5)
    ax2.axis("off")

    # Initialize the SAM2 model with the provided RGBs and masks
    sam2_video_model.cleanup()
    sam2_video_model.pad_and_store(rgbs, refined_masks, scores)

    # Connect events to handlers
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: on_click(event, mask_overlay, sam2_video_model, frame_idx),
    )
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key(
            event, sam2_video_model, data_save_dir, plot_save_dir, obj_ids, current_idx
        ),
    )

    plt.suptitle(
        f"Interactive Mask Editor - Label: {label} (Object ID: {current_obj_id})"
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def main(args):
    global refined_crops

    # Load configurations from the provided YAML file
    cfg = load_yaml_munch(args.config_file)

    # Initialize paths and configurations
    if cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)

    output_dir = cfg.get("output_dir")
    if output_dir is None:
        # Default to data folder in data_root
        output_dir = Path(cfg.data_root) / "data"
    output_dir = Path(output_dir)

    # Determine the render device(s) to use
    render_devices = []
    if cfg.get("render_dslr", False):
        render_devices.append("dslr")
        raise Exception("This code has not been tested with the DSLR data.")
    if cfg.get("render_iphone", False):
        render_devices.append("iphone")

    # Assuming one scene ID and one device for simplicity
    scene_id = scene_ids[0]
    device = render_devices[0]

    # Set paths for refined crops data
    refined_crops_data_dir = output_dir / scene_id / device / "refined_crops_data"
    refined_crops = load_refined_crops(refined_crops_data_dir)
    obj_ids = list(refined_crops.keys())  # Get object IDs to process

    # Initialize the SAM2 video model with checkpoint and configuration
    sam2_checkpoint = "/home/kumaraditya/checkpoints/sam2_hiera_large.pt"
    sam2_model_cfg = "sam2_hiera_l.yaml"
    sam2_video_model = SAM2VideoMaskModel(sam2_checkpoint, sam2_model_cfg)

    # Directory to save edited masks
    plot_save_dir = refined_crops_data_dir.parent / "render_crops_manual_refined"

    # Print instructions for using the interactive editor
    print("Instructions:")
    print("1. Click on either image to add or subtract the mask.")
    print("2. Press 'a' to switch to addition mode.")
    print("3. Press 'x' to switch to subtraction mode.")
    print("4. Press 'r' to reset.")
    print("5. Press 'n' to save and move to the next object.")
    print("6. Press 'enter' to save and exit the editor.")

    # Start the interactive mask editor with the first object
    starting_idx = 0
    interactive_mask_editor(
        rgbs=refined_crops[obj_ids[starting_idx]]["rgbs"],
        refined_masks=refined_crops[obj_ids[starting_idx]]["refined_masks"],
        scores=refined_crops[obj_ids[starting_idx]]["scores"],
        sam2_video_model=sam2_video_model,
        label=refined_crops[obj_ids[starting_idx]]["label"],
        obj_ids=obj_ids,
        current_idx=starting_idx,
        data_save_dir=refined_crops_data_dir,
        plot_save_dir=plot_save_dir,
    )

    # Cleanup model resources after all objects are processed
    sam2_video_model.cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "config_file",
        help="Path to config file",
        default="/home/kumaraditya/scannetpp/common/configs/render.yml",
        nargs="?",
    )
    args = p.parse_args([])

    print(f"Config file: {args.config_file}")

    main(args)
