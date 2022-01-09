import torch
from torch.nn import functional as F
import numpy as np


def ms_pp(input_view, pan_model, flip_grid, disp, min_disp, max_pix):
    _, _, H, W = input_view.shape

    up_fac = 2 / 3
    upscaled = F.interpolate(
        F.grid_sample(input_view, flip_grid, align_corners=False),
        scale_factor=up_fac,
        mode="bilinear",
        align_corners=True,
        recompute_scale_factor=True,
    )
    dwn_flip_disp = pan_model(
        input_left=upscaled,
        min_disp=min_disp,
        max_disp=max_pix,
        ret_disp=True,
        ret_pan=False,
        ret_subocc=False,
    )
    dwn_flip_disp = (1 / up_fac) * F.interpolate(
        dwn_flip_disp, size=(H, W), mode="nearest"
    )  # , align_corners=True)
    dwn_flip_disp = F.grid_sample(dwn_flip_disp, flip_grid, align_corners=False)

    norm = disp / (np.percentile(disp.detach().cpu().numpy(), 95) + 1e-6)
    norm[norm > 1] = 1

    return (1 - norm) * disp + norm * dwn_flip_disp


def local_normalization(img, win=3):
    B, C, _, _ = img.shape
    mean = [0.411, 0.432, 0.45]
    m_rgb = torch.ones((B, C, 1, 1)).type(img.type())
    m_rgb[:, 0, :, :] = mean[0] * m_rgb[:, 0, :, :]
    m_rgb[:, 1, :, :] = mean[1] * m_rgb[:, 1, :, :]
    m_rgb[:, 2, :, :] = mean[2] * m_rgb[:, 2, :, :]

    img = img + m_rgb
    img = img.cpu()

    # Get mean and normalize
    win_mean_T = F.avg_pool2d(
        img, kernel_size=win, stride=1, padding=(win - 1) // 2
    )  # B,C,H,W
    win_std = F.avg_pool2d(
        (img - win_mean_T) ** 2, kernel_size=win, stride=1, padding=(win - 1) // 2
    ) ** (1 / 2)
    win_norm_img = (img - win_mean_T) / (win_std + 0.0000001)

    return win_norm_img
