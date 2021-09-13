import os

import numpy as np

import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
from misc.listdataset_run import ListDataset as RunListDataset
import matplotlib.pyplot as plt

from models.FAL_netB import FAL_netB
from misc import utils, data_transforms


def predict(args, device="cpu"):
    print("-------Predicting on " + str(device) + "-------")

    if args.model.isdigit():
        save_path = os.path.join("predict", args.model.zfill(10))
    else:
        model_number = args.model.split("/")[-2].zfill(10)
        save_path = os.path.join("predict", model_number)
    if args.f_post_process:
        save_path = save_path + "fpp"
    if args.ms_post_process:
        save_path = save_path + "mspp"
    print("=> Saving to {}".format(save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    utils.display_config(args, save_path)

    input_transform = transforms.Compose(
        [
            data_transforms.ArrayToTensor(),
            transforms.Normalize(
                mean=[0, 0, 0], std=[255, 255, 255]
            ),  # (input - mean) / std
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
        ]
    )

    # Torch Data Set List
    predict_dataset = RunListDataset(
        path_list=[args.input],
        transform=input_transform,
    )

    print("len(predict_dataset)", len(predict_dataset))
    # Torch Data Loader
    val_loader = torch.utils.data.DataLoader(
        predict_dataset,
        batch_size=1,  # kitty mixes image sizes!
        num_workers=args.workers,
        pin_memory=False,
        shuffle=False,
    )

    # create pan model
    if args.model.isdigit():
        model_path = os.path.join(
            args.dataset + "_stage2", args.model.zfill(10), "model_best.pth.tar"
        )
        if not os.path.exists(model_path):
            model_path = os.path.join(
                args.dataset + "_stage2", args.model.zfill(10), "checkpoint.pth.tar"
            )
    else:
        model_path = args.model

    print(model_path)

    pan_model = FAL_netB(no_levels=args.no_levels, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    pan_model.load_state_dict(checkpoint["model_state_dict"])
    if device.type == "cuda":
        pan_model = torch.nn.DataParallel(pan_model).to(device)

    model_parameters = utils.get_n_params(pan_model)
    print("=> Number of parameters '{}'".format(model_parameters))
    cudnn.benchmark = True

    l_disp_path = os.path.join(save_path, "l_disp")
    if not os.path.exists(l_disp_path):
        os.makedirs(l_disp_path)

    # Set the max disp
    right_shift = args.max_disp * args.rel_baset

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            input_left = input.to(device)
            B, C, H, W = input_left.shape

            # Prepare flip grid for post-processing
            i_tetha = torch.zeros(B, 2, 3).to(device)
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            flip_grid = F.affine_grid(i_tetha, [B, C, H, W], align_corners=False)
            flip_grid[:, :, :, 0] = -flip_grid[:, :, :, 0]

            # Convert min and max disp to bx1x1 tensors
            max_disp = (
                torch.Tensor([right_shift])
                .unsqueeze(1)
                .unsqueeze(1)
                .type(input_left.type())
            )
            min_disp = max_disp * args.min_disp / args.max_disp

            disp = pan_model(
                input_left=input_left,
                min_disp=min_disp,
                max_disp=max_disp,
                ret_disp=True,
                ret_subocc=False,
                ret_pan=False,
            )

            if args.ms_post_process:
                disp = ms_pp(input_left, pan_model, flip_grid, disp, min_disp, max_disp)

            disparity = disp.squeeze().cpu().numpy()
            disparity = 256 * np.clip(
                disparity / (np.percentile(disparity, 99) + 1e-9), 0, 1
            )
            plt.imsave(
                os.path.join(l_disp_path, "{:010d}.png".format(i)),
                np.rint(disparity).astype(np.int32),
                cmap="inferno",
                vmin=0,
                vmax=256,
            )


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
