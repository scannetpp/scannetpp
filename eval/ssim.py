from typing import Optional

import torch
from math import exp
import torch.nn as nn
import torch.nn.functional as F


def gaussian(kernel_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor(
        [
            exp(-((x - kernel_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(kernel_size)
        ],
    ).float()
    return gauss / gauss.sum()


def create_gaussian_kernel(kernel_size: int, channel: int) -> torch.Tensor:
    gaussian_1d = gaussian(kernel_size, 1.5).unsqueeze(1)
    gaussian_2d = torch.matmul(gaussian_1d, gaussian_1d.t()).unsqueeze(0).unsqueeze(0)

    kernel = gaussian_2d.expand(channel, 1, kernel_size, kernel_size).contiguous()
    return kernel


def get_mask_with_kernel(mask: torch.Tensor, kernel_size: int = 11) -> torch.Tensor:
    """Compute the mask of valid pixels after applying a kernel.
    Args:
        mask: (Tensor) mask in format (batch, height, width). 1 denotes valid pixels while 0 denotes invalid pixels.
        kernel_size: (int) size of the gaussian kernel. Default: 11
    Returns:
        Tensor: new mask in format (batch, height, width). 1 denotes valid pixels while 0 denotes invalid pixels.
    """
    mask = mask.float()
    assert mask.min() >= 0 and mask.max() <= 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask.device)
    # Pad zeros to the mask boundary
    pad_h = (kernel_size - 1) // 2
    pad_w = (kernel_size - 1) // 2

    # The border area will be filled with 0 (invalid pixels) because the original image is padded
    # Those regions are not considered in the ssim calculation
    mask = torch.nn.functional.pad(
        mask, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0
    )
    mask = torch.nn.functional.conv2d(mask, kernel)
    full_value = kernel_size * kernel_size
    new_mask = mask >= full_value
    return new_mask


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kernel_size: int = 11
) -> torch.Tensor:
    """Calculate SSIM (structural similarity) for a batch of images
    Args:
        img1: (Tensor) images in format (batch, channel, height, width)
        img2: (Tensor) images in format (batch, channel, height, width)
        mask: (Tensor) mask in format (batch, height, width). If None, no mask will be applied. Default: None
            1 denotes valid pixels while 0 denotes invalid pixels.
        kernel_size: (int) size of the gaussian kernel. Default: 11
    Returns:
        Tensor: ssim results.
    """
    assert img1.size() == img2.size()
    assert img1.min() >= 0 and img1.max() <= 1
    assert img2.min() >= 0 and img2.max() <= 1
    assert (
        len(img1.shape) == 4
    ), "image input should have shape (batch, channel, height, width)"
    if mask is not None:
        assert len(mask.shape) == 3, f"mask should have shape (batch, height, width) instead of {mask.shape}"

    channel = img1.size(-3)
    kernel = create_gaussian_kernel(kernel_size, channel)
    kernel = kernel.to(img1.device)

    pad_size = (kernel_size - 1) // 2
    img1 = F.pad(img1, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
    img2 = F.pad(img2, (pad_size, pad_size, pad_size, pad_size), mode="reflect")

    ssim_map = _ssim(img1, img2, kernel, channel)

    if mask is None:
        # The border area will not be considered since they are affected by the padding
        valid_ssim = ssim_map[..., pad_size:-pad_size, pad_size:-pad_size]
    else:
        # Every pixels that are affected by the masked region will not be considered
        mask = get_mask_with_kernel(mask, kernel_size)
        valid_ssim = torch.masked_select(ssim_map, mask)
    return valid_ssim.mean()


def _ssim(img1, img2, kernel, channel):
    mu1 = F.conv2d(img1, kernel, groups=channel)
    mu2 = F.conv2d(img2, kernel, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map
