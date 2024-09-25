from typing import List, Union
import matplotlib.pyplot as plt
import heapq
import numpy as np
import cv2


def vert_to_obj_lookup(segments_anno):
    # Find max
    vert_max = 0
    for obj in segments_anno["segGroups"]:
        vert_max = max(vert_max, max(obj["segments"]))

    # Objects start at 1 so use 0 for background
    lookup = np.zeros(shape=vert_max + 1, dtype=np.uint32)
    for obj in segments_anno["segGroups"]:
        for vert in obj["segments"]:
            lookup[vert] = obj["id"]

    return lookup


class Crop:
    def __init__(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        score: float,
    ):
        self.rgb = rgb
        self.mask = mask
        self.score = score

    def __lt__(self, other):
        return self.score < other.score


class CropHeap:
    def __init__(self, max_size: int = 9):
        self.max_size = max_size
        self.heap = []

    def __getitem__(self, item):
        return self.heap[item]

    def __iter__(self):
        return iter(self.heap)

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        return f"CropHeap of size {len(self)} with max size {self.max_size} and segment scores {[s.score for s in self]}"

    def push(self, segment: Crop) -> bool:
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, segment)
            return True
        else:
            current_min = self.heap[0]
            heapq.heappushpop(self.heap, segment)
            new_min = self.heap[0]
            return new_min != current_min

    def get_sorted(self) -> List[Crop]:
        return sorted(self.heap, key=lambda x: x.score, reverse=True)


def mask_to_bbox(mask, inflate_px=0):
    # Ensure the input is a numpy array
    mask = np.asarray(mask)

    # Find indices where mask is non-zero
    non_zero_indices = np.nonzero(mask)

    # If the mask is empty (all zeros), return None or a default bbox
    if len(non_zero_indices[0]) == 0:
        return None

    # Extract coordinates
    rows = non_zero_indices[0]
    cols = non_zero_indices[1]

    # Compute the bounding box
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    # Detect if original bbox touches border (before inflate)
    h, w = mask.shape[:2]
    touches_top = min_row <= 0
    touches_bottom = max_row >= h - 1
    touches_left = min_col <= 0
    touches_right = max_col >= w - 1
    touches_border = touches_left | touches_right | touches_top | touches_bottom

    # Inflate the bounding box
    min_row -= inflate_px
    max_row += inflate_px
    min_col -= inflate_px
    max_col += inflate_px

    # Ensure the bounding box is within the image boundaries
    min_row = max(min_row, 0)
    max_row = min(max_row, mask.shape[0] - 1)
    min_col = max(min_col, 0)
    max_col = min(max_col, mask.shape[1] - 1)

    return (min_row, max_row, min_col, max_col), touches_border


def compute_optical_flow_score(rgb, rgb_rendered) -> float:
    """
    Computes a score based on optical flow magnitude between two images.

    Parameters:
    - rgb: Original RGB image (numpy array).
    - rgb_rendered: Rendered RGB image (numpy array).

    Returns:
    - score: A float score where lower optical flow magnitudes yield higher scores.
    """

    # Convert the images to grayscale
    gray_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_rgb_rendered = cv2.cvtColor(rgb_rendered, cv2.COLOR_RGB2GRAY)

    # scale down by 3 for faster computation
    gray_rgb = cv2.resize(gray_rgb, None, fx=1 / 4, fy=1 / 4)
    gray_rgb_rendered = cv2.resize(gray_rgb_rendered, None, fx=1 / 4, fy=1 / 4)

    # Compute the optical flow between the two grayscale images
    flow = cv2.calcOpticalFlowFarneback(
        gray_rgb, gray_rgb_rendered, None, 0.5, 2, 15, 3, 5, 1.2, 0
    )

    # Compute the magnitude of the optical flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_mismatch = np.mean(magnitude)
    score = 1 / (avg_mismatch + 1e-6)  # Add small value to avoid division by zero

    return score


def crop_rgb_mask(rgb, rgb_rendered, mask, inflate_px=0, border_penalty_factor=0.1):
    # optical_flow_score = compute_optical_flow_score(rgb, rgb_rendered)
    # optical_flow_score = 1.0

    (x_min, x_max, y_min, y_max), touches_border = mask_to_bbox(
        mask, inflate_px=inflate_px
    )
    rgb, mask = rgb[x_min:x_max, y_min:y_max], mask[x_min:x_max, y_min:y_max]

    score = mask.sum()
    # score *= optical_flow_score

    if touches_border:
        score *= border_penalty_factor

    return Crop(rgb, mask, score)


def plot_grid_images(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    grid_width: int = 4,
    title: str = "",
) -> None:
    n_images = len(images)
    grid_height = 3
    fig, axs = plt.subplots(
        grid_height,
        grid_width,
        figsize=(
            grid_width * 4,
            grid_height * 4,
        ),
    )

    for i in range(grid_width):
        if i >= n_images:
            break  # If fewer images than grid_width, break

        img = images[i]
        mask = masks[i]

        axs[0, i].imshow(img)
        axs[0, i].axis("off")

        axs[1, i].imshow(mask, cmap="viridis")  # Display binary mask
        axs[1, i].axis("off")

        masked_img = img.copy()
        masked_img[mask == 0] = [
            255,
            255,
            255,
        ]  # Set the background to white where mask is 0
        axs[2, i].imshow(masked_img)
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.suptitle(title, fontsize=30)
    plt.subplots_adjust(top=0.90)
