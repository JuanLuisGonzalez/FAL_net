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
from imageio import imsave
import matplotlib.pyplot as plt
from PIL import Image

import models
from misc.dataloader import load_data

import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F

from misc import utils, data_transforms
from misc.loss_functions import realEPE


model_names = sorted(name for name in models.__all__)


def main(args, device="cpu"):
    print("-------Testing on " + str(device) + "-------")

    save_path = os.path.join("Test_Results", args.dataset, args.model, args.time_stamp)
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

    target_transform = transforms.Compose(
        [
            data_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0], std=[1]),
        ]
    )

    # Torch Data Set List
    input_path = os.path.join(args.data_directory, args.dataset)
    test_dataset = load_data(
        split=args.test_split,
        dataset=args.dataset,
        root=input_path,
        disp=True,
        shuffle_test=False,
        transform=input_transform,
        target_transform=target_transform,
    )

    print("len(test_dataset)", len(test_dataset))
    # Torch Data Loader
    args.batch_size = 1  # kitty mixes image sizes!
    args.sparse = True  # disparities are sparse (from lidar)
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=False,
    )

    # create pan model
    model_dir = os.path.join("KITTI_stage2", args.time_stamp, args.model + args.details)
    print(model_dir)
    pan_network_data = torch.load(model_dir, map_location=torch.device(device))

    pan_model = pan_network_data[
        next(item for item in pan_network_data.keys() if "model" in str(item))
    ]

    print("=> using pre-trained model for pan '{}'".format(pan_model))
    pan_model = models.__dict__[pan_model](
        pan_network_data, no_levels=args.no_levels, device=device
    ).to(device)
    pan_model = torch.nn.DataParallel(pan_model).to(device)
    if device.type == "cpu":
        pan_model = pan_model.module.to(device)
    pan_model.eval()
    model_parameters = utils.get_n_params(pan_model)
    print("=> Number of parameters '{}'".format(model_parameters))
    cudnn.benchmark = True

    print("len(val_loader)", len(val_loader))
    # evaluate on validation set
    validate(args, val_loader, pan_model, save_path, model_parameters, device)


def validate(args, val_loader, pan_model, save_path, model_param, device):
    batch_time = utils.AverageMeter()
    EPEs = utils.AverageMeter()
    kitti_erros = utils.multiAverageMeter(utils.kitti_error_names)

    l_disp_path = os.path.join(save_path, "l_disp")
    if not os.path.exists(l_disp_path):
        os.makedirs(l_disp_path)

    input_path = os.path.join(save_path, "Input im")
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    pan_path = os.path.join(save_path, "Pan")
    if not os.path.exists(pan_path):
        os.makedirs(pan_path)

    pc_path = os.path.join(save_path, "Point_cloud")
    if not os.path.exists(pc_path):
        os.makedirs(pc_path)

    feats_path = os.path.join(save_path, "feats")
    if not os.path.exists(feats_path):
        os.makedirs(feats_path)

    # Set the max disp
    right_shift = args.max_disp * args.rel_baset

    with torch.no_grad():
        print("with torch.no_grad():")
        for i, (input, target, _) in enumerate(val_loader):
            # print("for i, (input, target, f_name) in enumerate(val_loader):", i)
            target = target[0].to(device)
            input_left = input[0].to(device)
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
                # You can append any feature map to feats, and they will be printed as 1 channel grayscale images
                feats = [maskL, maskRL]
                feats = [local_normalization(input_left), dispr / 100, maskL, maskRL]
            else:
                disp = pan_model(
                    input_left=input_left,
                    min_disp=min_disp,
                    max_disp=max_disp,
                    ret_disp=True,
                    ret_subocc=False,
                    ret_pan=False,
                )
                feats = None

            if args.f_post_process:
                flip_disp = pan_model(
                    input_left=F.grid_sample(
                        input_left, flip_grid, align_corners=False
                    ),
                    min_disp=min_disp,
                    max_disp=max_disp,
                    ret_disp=True,
                    ret_pan=False,
                    ret_subocc=False,
                )
                flip_disp = F.grid_sample(flip_disp, flip_grid, align_corners=False)
                disp = (disp + flip_disp) / 2
            elif args.ms_post_process:
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

                if args.save_pc:
                    # equalize tone
                    m_rgb = torch.ones((B, C, 1, 1)).to(device)
                    m_rgb[:, 0, :, :] = 0.411 * m_rgb[:, 0, :, :]
                    m_rgb[:, 1, :, :] = 0.432 * m_rgb[:, 1, :, :]
                    m_rgb[:, 2, :, :] = 0.45 * m_rgb[:, 2, :, :]
                    point_cloud = utils.get_point_cloud(
                        (input_left + m_rgb) * 255, disp
                    )
                    utils.save_point_cloud(
                        point_cloud.squeeze(0).cpu().numpy(),
                        os.path.join(pc_path, "{:010d}.ply".format(i)),
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

                if args.save_pan:
                    # save synthetic image
                    denormalize = np.array([0.411, 0.432, 0.45])
                    denormalize = denormalize[:, np.newaxis, np.newaxis]
                    im = pan_im.squeeze().cpu().numpy() + denormalize
                    im = Image.fromarray(
                        np.rint(255 * im.transpose(1, 2, 0)).astype(np.uint8)
                    )
                    im.save(os.path.join(pan_path, "{:010d}.png".format(i)))

                # save features per channel as grayscale images
                if feats is not None:
                    for layer in range(len(feats)):
                        _, nc, _, _ = feats[layer].shape
                        for inc in range(nc):
                            feature = (
                                255
                                * torch.abs(feats[layer][:, inc, :, :])
                                .squeeze()
                                .cpu()
                                .numpy()
                            )
                            feature[feature < 0] = 0
                            feature[feature > 255] = 255
                            imsave(
                                os.path.join(
                                    feats_path,
                                    "{:010d}_l{}_c{}.png".format(i, layer, inc),
                                ),
                                np.rint(feature).astype(np.uint8),
                            )

            if args.evaluate:
                # Record kitti metrics
                target_disp = target.squeeze(1).cpu().numpy()
                pred_disp = disp.squeeze(1).cpu().numpy()
                if (
                    args.dataset == "Kitti_eigen_test_improved"
                    or args.dataset == "Kitti_eigen_test_original"
                    or args.dataset == "KITTI"
                ):
                    target_depth, pred_depth = utils.disps_to_depths_kitti(
                        target_disp, pred_disp
                    )
                    kitti_erros.update(
                        utils.compute_kitti_errors(
                            target_depth[0], pred_depth[0], use_median=args.median
                        ),
                        target.size(0),
                    )
                if args.dataset == "Kitti2015":
                    EPE = realEPE(disp, target, sparse=True)
                    EPEs.update(EPE.detach(), target.size(0))
                    target_depth, pred_depth = utils.disps_to_depths_kitti2015(
                        target_disp, pred_disp
                    )
                    kitti_erros.update(
                        utils.compute_kitti_errors(
                            target_depth[0], pred_depth[0], use_median=args.median
                        ),
                        target.size(0),
                    )

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t Time {2}\t a1 {3:.4f}".format(
                        i, len(val_loader), batch_time, kitti_erros.avg[4]
                    )
                )

    # Save erros and number of parameters in txt file
    with open(os.path.join(save_path, "errors.txt"), "w+") as f:
        f.write("\nNumber of parameters {}\n".format(model_param))
        f.write("\nEPE {}\n".format(EPEs.avg))
        f.write("\nKitti metrics: \n{}\n".format(kitti_erros))

    if args.evaluate:
        print("* EPE: {0}".format(EPEs.avg))
        print(kitti_erros)


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
