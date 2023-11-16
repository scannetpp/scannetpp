from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    """LPIPS score that supports inputing a invalid mask, and consider only the patches that are not affected by the invalid regions.
    """

    def __init__(
        self,
        net_type: str = "vgg",
        version: str = "0.1",
        normalize: bool = True,
        eval_mode: bool = True,
    ) -> None:
        assert net_type in ["vgg"], "only vgg is supported now"
        assert version in ["0.1"], "v0.1 is only supported now"

        super(LPIPS, self).__init__()

        self.normalize = normalize
        # pretrained network
        self.net = get_network(net_type)
        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

        if eval_mode:
            self.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.normalize:
            x = x * 2 - 1
            y = y * 2 - 1
        feat_x, feat_y = self.net(x), self.net(y)
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)

        mask_each_layer = []
        vgg_num_convs_each_layer = [2, 2, 3, 3, 3]
        if mask is not None:
            mask = mask.float()
            for num_convs in vgg_num_convs_each_layer:
                # The equivalent kernel size is 1+2*num_convs
                kernel_size = num_convs * 2 + 1
                pad_size = (kernel_size - 1) // 2
                conv = torch.ones(1, 1, kernel_size, kernel_size).to(mask.device)
                mask = F.pad(
                    mask,
                    (pad_size, pad_size, pad_size, pad_size),
                    mode="constant",
                    value=1.0,
                )
                mask = F.conv2d(mask, conv)
                mask = (mask >= kernel_size * kernel_size).float()
                mask_each_layer.append(mask > 0)
                mask = F.avg_pool2d(mask, (2, 2))
                mask = (mask >= 1.0).float()

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d) for d, l in zip(diff, self.lin)]

        if mask is not None:
            res = [torch.masked_select(r, m).mean() for r, m in zip(res, mask_each_layer)]
        else:
            res = [r.mean() for r in res]
        return sum(res)
