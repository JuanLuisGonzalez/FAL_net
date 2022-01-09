# Test_KITTI.py Test trained model on diverse KITTI splits
# Copyright (C) 2021  Juan Luis Gonzalez Bello (juanluisgb@kaist.ac.kr)
# This software is not for commercial use
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import os

import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.FAL_netB import FAL_netB
from misc.dataloader import load_data

import torch
import torch.utils.data
import torch.nn.parallel
from torch.backends import cudnn
from torchvision import transforms
from torch.nn import functional as F

from misc import utils, data_transforms


def main(args, device="cpu"):
    print("-------Testing on " + str(device) + "-------")

    input_transform = data_transforms.ApplyToMultiple(
        transforms.Compose(
            [
                transforms.Resize(size=(320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
            ]
        )
    )
    output_transforms = data_transforms.ApplyToMultiple(
        transforms.Compose(
            [
                data_transforms.NormalizeInverse(
                    mean=[0.411, 0.432, 0.45], std=[1, 1, 1]
                ),
                transforms.ToPILImage(),
            ]
        )
    )

    # Torch Data Set List
    test_dataset = load_data(
        dataset=args.dataset,
        root=args.data_directory,
        shuffle_test=False,
        transform=input_transform,
    )

    print("len(test_dataset)", len(test_dataset))
    # Torch Data Loader
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=False,
    )

    print("len(val_loader)", len(val_loader))

    # create pan model
    model_path = args.model

    print(model_path)

    pan_model = FAL_netB(no_levels=args.no_levels, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    pan_model.load_state_dict(checkpoint["model_state_dict"])
    if device.type == "cuda":
        pan_model = torch.nn.DataParallel(pan_model).to(device)

    pan_model.eval()
    model_parameters = utils.get_n_params(pan_model)
    print("=> Number of parameters '{}'".format(model_parameters))
    cudnn.benchmark = True

    print("len(val_loader)", len(val_loader))
    # evaluate on validation set
    validate(
        args,
        val_loader,
        pan_model,
        args.save_path,
        model_parameters,
        device,
        output_transforms,
    )


def validate(
    args, val_loader, pan_model, save_path, model_param, device, output_transforms
):
    batch_time = utils.AverageMeter()
    EPEs = utils.AverageMeter()
    asm_erros = utils.multiAverageMeter(utils.image_similarity_measures)

    l_disp_path = os.path.join(save_path, "l_disp")
    if not os.path.exists(l_disp_path):
        os.makedirs(l_disp_path)

    input_path = os.path.join(save_path, "Input im")
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # Set the max disp
    right_shift = args.max_disp * args.relative_baseline

    with torch.no_grad():
        print("with torch.no_grad():")
        for i, ([input_left, input_right], _max_pix) in enumerate(val_loader):

            input_left = input_left.to(device)
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

            # Synthesis
            end = time.time()

            # Get disp
            if args.save_pan:
                pan_im, disp, maskL, maskRL, dispr = pan_model(
                    input_left=input_left,
                    min_disp=min_disp,
                    max_disp=max_disp,
                    ret_disp=True,
                    ret_subocc=True,
                    ret_pan=True,
                )
            else:
                pan_im, disp = pan_model(
                    input_left=input_left,
                    min_disp=min_disp,
                    max_disp=max_disp,
                    ret_disp=True,
                    ret_subocc=False,
                    ret_pan=True,
                )

            if args.ms_post_process:
                disp = ms_pp(input_left, pan_model, flip_grid, disp, min_disp, max_disp)

            # measure elapsed time
            batch_time.update(time.time() - end, 1)

            # Save outputs to disk
            if args.save:
                # Save monocular lr
                disparity = disp.squeeze().cpu().numpy()
                disparity = 256 * np.clip(
                    disparity / (np.percentile(disparity, 95) + 1e-6), 0, 1
                )
                plt.imsave(
                    os.path.join(l_disp_path, "{:010d}.png".format(i)),
                    np.rint(disparity).astype(np.int32),
                    cmap="plasma",
                    vmin=0,
                    vmax=256,
                )

            if args.save_input:
                print("save the input image in path", input_path)
                denormalize = np.array([0.411, 0.432, 0.45])
                denormalize = denormalize[:, np.newaxis, np.newaxis]
                p_im = input_left.squeeze().cpu().numpy() + denormalize
                im = Image.fromarray(
                    np.rint(255 * p_im.transpose(1, 2, 0)).astype(np.uint8)
                )
                im.save(os.path.join(input_path, "{:010d}.png".format(i)))

            for target_im, pred_im in zip(input_right, pan_im):
                [target_im, pred_im] = output_transforms([target_im, pred_im])
                errors = utils.compute_asm_errors(target_im, pred_im)
                asm_erros.update(errors)

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t Time {2}\t SSIM {3:.4f}".format(
                        i, len(val_loader), batch_time, asm_erros.avg[5]
                    )
                )

    # Save erros and number of parameters in txt file
    with open(os.path.join(save_path, "errors.txt"), "w+") as f:
        f.write("\nNumber of parameters {}\n".format(model_param))
        f.write("\nEPE {}\n".format(EPEs.avg))
        f.write("\nKitti metrics: \n{}\n".format(asm_erros))

    print("* EPE: {0}".format(EPEs.avg))
    print(asm_erros)


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
