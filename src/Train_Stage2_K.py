# Train_Stage2_K.py Fine tune model with MOM on KITTI only
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

from misc.dataloader import load_data
import models
from models.FAL_netB import FAL_netB

import torch
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


# Usefull tensorboard call
# tensorboard --logdir=C:ProjectDir/NeurIPS2020_FAL_net/Kitti --port=6012

from misc import utils
from misc import data_transforms
from misc.loss_functions import realEPE, smoothness, VGGLoss


def main(args, device="cpu"):
    print("-------Training Stage 2 on " + str(device) + "-------")
    best_rmse = -1

    save_path = os.path.join(args.dataset + "_stage2")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    _, sub_directories, _ = next(os.walk(save_path))
    filtered = filter(lambda x: x.isdigit(), sorted(sub_directories))
    idx = len(list(filtered))
    save_path = os.path.join(save_path, str(idx).zfill(10))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    utils.display_config(args, save_path)
    print("=> will save everything to {}".format(save_path))

    # Set output writters for showing up progress on tensorboardX
    train_writer = SummaryWriter(os.path.join(save_path, "train"))
    test_writer = SummaryWriter(os.path.join(save_path, "test"))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, "test", str(i))))

    # Set up data augmentations
    co_transform = data_transforms.Compose(
        [
            data_transforms.RandomResizeCrop(
                (args.crop_height, args.crop_width), down=0.75, up=1.5
            ),
            data_transforms.RandomHorizontalFlip(),
            data_transforms.RandomGamma(min=0.8, max=1.2),
            data_transforms.RandomBrightness(min=0.5, max=2.0),
            data_transforms.RandomCBrightness(min=0.8, max=1.2),
        ]
    )

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
    train_dataset0 = load_data(
        split=args.train_split,
        dataset=args.dataset,
        root=input_path,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_transform,
        max_pix=args.max_disp,
        fix=True,
    )
    input_path = os.path.join(args.data_directory, args.validation_dataset)
    test_dataset = load_data(
        split=args.validation_split,
        dataset=args.validation_dataset,
        root=input_path,
        disp=True,
        of=False,
        shuffle_test=False,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=co_transform,
    )

    # Torch Data Loaders
    train_loader0 = torch.utils.data.DataLoader(
        train_dataset0,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.tbatch_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=False,
    )

    print("len(train_loader0)", len(train_loader0))
    print("len(val_loader)", len(val_loader))

    # create model
    if args.pretrained:
        if len(args.pretrained) > 11:
            network_data = torch.load(args.pretrained)
        else:
            pretrained_model_path = os.path.join(
                args.dataset + "_stage2",
                args.pretrained,
            )
            if not os.path.exists(pretrained_model_path):
                raise Exception(
                    f"No pretrained model with timestamp {args.pretrained} was found."
                )
            pretrained_model_path = os.path.join(
                pretrained_model_path,
                next(
                    d
                    for d in (next(os.walk(pretrained_model_path))[1])
                    if not d[0] == "."
                ),
                "checkpoint.pth.tar",
            )
            network_data = torch.load(pretrained_model_path)

        args.model = network_data[
            next(item for item in network_data.keys() if "model" in str(item))
        ]
        print("=> using pre-trained model '{}'".format(args.model))
    else:
        network_data = None
        print("=> creating m model '{}'".format(args.model))

    model = models.__dict__[args.model](
        network_data, no_levels=args.no_levels, device=device
    ).to(device)
    model = torch.nn.DataParallel(model).to(device)
    print("=> Number of parameters m-model '{}'".format(utils.get_n_params(model)))

    # create fix model

    if args.fix_model.isdigit():
        fix_model_path = os.path.join(
            args.dataset + "_stage1", args.fix_model.zfill(10), "model_best.pth.tar"
        )
        if not os.path.exists(fix_model_path):
            fix_model_path = os.path.join(
                args.dataset + "_stage1", args.fix_model.zfill(10), "checkpoint.pth.tar"
            )
    else:
        fix_model_path = args.fix_model

    fix_model = FAL_netB(no_levels=args.no_levels, device=device)
    checkpoint = torch.load(fix_model_path, map_location=device)
    fix_model.load_state_dict(checkpoint["model_state_dict"])
    if device.type == "cuda":
        fix_model = torch.nn.DataParallel(fix_model).to(device)

    print("=> Number of parameters m-model '{}'".format(utils.get_n_params(fix_model)))

    fix_model.eval()

    # Optimizer Settings
    print("Setting {} Optimizer".format(args.optimizer))
    param_groups = [
        {"params": model.module.bias_parameters(), "weight_decay": args.bias_decay},
        {
            "params": model.module.weight_parameters(),
            "weight_decay": args.weight_decay,
        },
    ]
    if args.optimizer == "adam":
        g_optimizer = torch.optim.Adam(
            params=param_groups, lr=args.lr2, betas=(args.momentum, args.beta)
        )
    g_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        g_optimizer, milestones=args.milestones2, gamma=0.5
    )

    for epoch in range(args.start_epoch):
        g_scheduler.step()

    vgg_loss = VGGLoss(device=device)
    scaler = GradScaler()

    for epoch in range(args.start_epoch, args.epochs2):
        # train for one epoch
        loss, train_loss = train(
            args,
            train_loader0,
            model,
            fix_model,
            g_optimizer,
            epoch,
            device,
            vgg_loss,
            scaler,
        )
        train_writer.add_scalar("train_loss", train_loss, epoch)

        # evaluate on validation set, RMSE is from stereoscopic view synthesis task
        rmse = validate(args, val_loader, model, epoch, output_writers, device)
        test_writer.add_scalar("mean RMSE", rmse, epoch)

        # Apply LR schedule (after optimizer.step() has been called for recent pyTorch versions)
        g_scheduler.step()

        if best_rmse < 0:
            best_rmse = rmse
        is_best = rmse < best_rmse
        best_rmse = min(rmse, best_rmse)
        utils.save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict()
                if isinstance(model, torch.nn.DataParallel)
                else model.state_dict(),
                "optimizer_state_dict": g_optimizer.state_dict(),
                "loss": loss,
            },
            is_best,
            save_path,
        )


def train(
    args, train_loader, model, fix_model, g_optimizer, epoch, device, vgg_loss, scaler
):
    epoch_size = (
        len(train_loader)
        if args.epoch_size == 0
        else min(len(train_loader), args.epoch_size)
    )

    batch_time = utils.RunningAverageMeter()
    data_time = utils.AverageMeter()
    rec_losses = utils.AverageMeter()
    losses = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, input_data0 in enumerate(train_loader):
        # Read training data
        left_view = input_data0[0][0].to(device)
        right_view = input_data0[0][1].to(device)
        max_disp = input_data0[1].unsqueeze(1).unsqueeze(1).type(left_view.type())
        min_disp = max_disp * args.min_disp / args.max_disp
        B, C, H, W = left_view.shape

        # measure data loading time
        data_time.update(time.time() - end)

        # Reset gradients
        g_optimizer.zero_grad()

        # Flip Grid (differentiable)
        i_tetha = torch.autograd.Variable(torch.zeros(B, 2, 3)).to(device)
        i_tetha[:, 0, 0] = 1
        i_tetha[:, 1, 1] = 1
        i_grid = F.affine_grid(i_tetha, [B, C, H, W], align_corners=True)
        flip_grid = i_grid.clone()
        flip_grid[:, :, :, 0] = -flip_grid[:, :, :, 0]

        # Get mirrored disparity from fixed falnet model
        if args.a_mr > 0:
            with torch.no_grad():
                disp = fix_model(
                    torch.cat(
                        (
                            F.grid_sample(left_view, flip_grid, align_corners=True),
                            right_view,
                        ),
                        0,
                    ),
                    torch.cat((min_disp, min_disp), 0),
                    torch.cat((max_disp, max_disp), 0),
                    ret_disp=True,
                    ret_pan=False,
                    ret_subocc=False,
                )
                mldisp = F.grid_sample(
                    disp[0:B, :, :, :], flip_grid, align_corners=True
                ).detach()
                mrdisp = disp[B::, :, :, :].detach()

        with autocast():
            ###### LEFT disp
            pan, disp, mask0, mask1 = model(
                torch.cat(
                    (
                        left_view,
                        F.grid_sample(right_view, flip_grid, align_corners=True),
                    ),
                    0,
                ),
                torch.cat((min_disp, min_disp), 0),
                torch.cat((max_disp, max_disp), 0),
                ret_disp=True,
                ret_pan=True,
                ret_subocc=True,
            )
            rpan = pan[0:B, :, :, :]
            lpan = pan[B::, :, :, :]
            ldisp = disp[0:B, :, :, :]
            rdisp = disp[B::, :, :, :]

            lmask = mask0[0:B, :, :, :]
            rmask = mask0[B::, :, :, :]
            rlmask = mask1[0:B, :, :, :]
            lrmask = mask1[B::, :, :, :]

            # Unflip right view stuff
            lpan = F.grid_sample(lpan, flip_grid, align_corners=True)
            rdisp = F.grid_sample(rdisp, flip_grid, align_corners=True)
            rmask = F.grid_sample(rmask, flip_grid, align_corners=True)
            lrmask = F.grid_sample(lrmask, flip_grid, align_corners=True)

            # Compute rec loss
            if args.a_p > 0:
                vgg_right = vgg_loss.vgg(right_view)
                vgg_left = vgg_loss.vgg(left_view)
            else:
                vgg_right = None
                vgg_left = None
            # Obtain final occlusion masks
            O_L = lmask * lrmask
            O_L[:, :, :, 0 : int(0.20 * W)] = 1
            O_R = rmask * rlmask
            O_R[:, :, :, int(0.80 * W) : :] = 1
            if args.a_mr == 0:  # no mirror loss, then it is just more training
                O_L = 1
                O_R = 1
            # Over 2 as measured twice for left and right
            rec_loss = (
                vgg_loss.rec_loss_fnc(O_R, rpan, right_view, vgg_right, args.a_p)
                + vgg_loss.rec_loss_fnc(O_L, lpan, left_view, vgg_left, args.a_p)
            ) / 2
            rec_losses.update(rec_loss.detach().cpu(), args.batch_size)

            # Compute smooth loss
            sm_loss = 0
            if args.smooth2 > 0:
                # Here we ignore the 20% left dis-occluded region, as there is no suppervision for it due to parralax
                sm_loss = (
                    smoothness(
                        left_view[:, :, :, int(0.20 * W) : :],
                        ldisp[:, :, :, int(0.20 * W) : :],
                        gamma=2,
                        device=device,
                    )
                    + smoothness(
                        right_view[:, :, :, 0 : int(0.80 * W)],
                        rdisp[:, :, :, 0 : int(0.80 * W)],
                        gamma=2,
                        device=device,
                    )
                ) / 2

            # Compute mirror loss
            mirror_loss = 0
            if args.a_mr > 0:
                # Normalize error ~ between 0-1, by diving over the max disparity value
                nmaxl = 1 / F.max_pool2d(mldisp, kernel_size=(H, W))
                nmaxr = 1 / F.max_pool2d(mrdisp, kernel_size=(H, W))
                mirror_loss = (
                    torch.mean(
                        nmaxl
                        * (1 - O_L)[:, :, :, int(0.20 * W) : :]
                        * torch.abs(ldisp - mldisp)[:, :, :, int(0.20 * W) : :]
                    )
                    + torch.mean(
                        nmaxr
                        * (1 - O_R)[:, :, :, 0 : int(0.80 * W)]
                        * torch.abs(rdisp - mrdisp)[:, :, :, 0 : int(0.80 * W)]
                    )
                ) / 2

            # compute gradient and do optimization step
            loss = rec_loss + args.smooth2 * sm_loss + args.a_mr * mirror_loss
            losses.update(loss.detach().cpu(), args.batch_size)

        scaler.scale(loss).backward()
        scaler.step(g_optimizer)
        scaler.update()
        g_optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            eta = utils.eta_calculator(
                batch_time.get_avg(), epoch_size, args.epochs2 - epoch, i
            )
            print(
                f"Epoch: [{epoch}][{i}/{epoch_size}] ETA {eta} Batch Time {batch_time}  Loss {losses} RecLoss {rec_losses}"
            )

        # End training epoch earlier if args.epoch_size != 0
        if i >= epoch_size:
            break

    return loss, losses.avg


def validate(args, val_loader, model, epoch, output_writers, device):

    test_time = utils.AverageMeter()
    RMSES = utils.AverageMeter()
    EPEs = utils.AverageMeter()
    kitti_erros = utils.multiAverageMeter(utils.kitti_error_names)

    # switch to evaluate mode
    model.eval()

    # Disable gradients to save memory
    with torch.no_grad():
        for i, input_data in enumerate(val_loader):
            input_left = input_data[0][0].to(device)
            input_right = input_data[0][1].to(device)
            target = input_data[1][0].to(device)
            max_disp = (
                torch.Tensor([args.max_disp * args.rel_baset])
                .unsqueeze(1)
                .unsqueeze(1)
                .type(input_left.type())
            )

            # Prepare input data
            end = time.time()
            min_disp = max_disp * args.min_disp / args.max_disp
            p_im, disp, maskL, maskRL = model(
                input_left=input_left,
                min_disp=min_disp,
                max_disp=max_disp,
                ret_disp=True,
                ret_pan=True,
                ret_subocc=True,
            )
            test_time.update(time.time() - end)

            # Measure RMSE
            rmse = utils.get_rmse(p_im, input_right, device=device)
            RMSES.update(rmse)

            # record EPE
            flow2_EPE = realEPE(disp, target, sparse=args.sparse)
            EPEs.update(flow2_EPE.detach(), target.size(0))

            # Record kitti metrics
            target_depth, pred_depth = utils.disps_to_depths_kitti2015(
                target.detach().squeeze(1).cpu().numpy(),
                disp.detach().squeeze(1).cpu().numpy(),
            )
            kitti_erros.update(
                utils.compute_kitti_errors(target_depth[0], pred_depth[0]),
                target.size(0),
            )

            denormalize = np.array([0.411, 0.432, 0.45])
            denormalize = denormalize[:, np.newaxis, np.newaxis]

            if i < len(output_writers):  # log first output of first batches
                if epoch == 0:
                    output_writers[i].add_image(
                        "Input left", input_left[0].cpu().numpy() + denormalize, 0
                    )

                # Plot disp
                output_writers[i].add_image(
                    "Left disparity", utils.disp2rgb(disp[0].cpu().numpy(), None), epoch
                )

                # Plot left subocclsion mask
                output_writers[i].add_image(
                    "Left sub-occ", utils.disp2rgb(maskL[0].cpu().numpy(), None), epoch
                )

                # Plot right-from-left subocclsion mask
                output_writers[i].add_image(
                    "RightL sub-occ",
                    utils.disp2rgb(maskRL[0].cpu().numpy(), None),
                    epoch,
                )

                # Plot synthetic right (or panned) view output
                p_im = p_im[0].detach().cpu().numpy() + denormalize
                p_im[p_im > 1] = 1
                p_im[p_im < 0] = 0
                output_writers[i].add_image("Output Pan", p_im, epoch)

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t Time {2}\t RMSE {3}".format(
                        i, len(val_loader), test_time, RMSES
                    )
                )

    print("* RMSE {0}".format(RMSES.avg))
    print(" * EPE {:.3f}".format(EPEs.avg))
    print(kitti_erros)
    return RMSES.avg
