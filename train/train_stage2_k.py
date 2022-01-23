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

from misc.dataloader import load_data
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
from misc.loss_functions import smoothness, VGGLoss
from train.kitti_validation import validate


def main(args, device="cpu"):
    print("-------Training Stage 2 on " + str(device) + "-------")
    best_rmse = -1

    # Set output writters for showing up progress on tensorboardX
    train_writer = SummaryWriter(os.path.join(args.save_path, "train"))
    test_writer = SummaryWriter(os.path.join(args.save_path, "test"))
    output_writers = []
    for i in range(3):
        output_writers.append(
            SummaryWriter(os.path.join(args.save_path, "test", str(i)))
        )

    transform = data_transforms.ApplyToMultiple(
        transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=(args.crop_height, args.crop_width),
                    scale=(0.10, 1.0),
                    ratio=(1, 1),
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3606, 0.3789, 0.3652], std=[0.3123, 0.3173, 0.3216]
                ),  # (input - mean) / std
            ]
        ),
        RandomHorizontalFlipChance=0.5,
    )

    # Set up data augmentations

    # Torch Data Set List
    input_path = os.path.join(args.data_directory, args.dataset)
    train_dataset = load_data(
        split=args.train_split,
        dataset=args.dataset,
        root=input_path,
        transform=transform,
    )

    input_transform = data_transforms.ApplyToMultiple(
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
            ]
        )
    )

    input_path = os.path.join(args.data_directory, args.validation_dataset)
    test_dataset = load_data(
        split=args.validation_split,
        dataset=args.validation_dataset,
        root=input_path,
        transform=input_transform,
    )

    # Torch Data Loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=False,
    )

    print("len(train_loader)", len(train_loader))
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

        model_description = network_data[
            next(item for item in network_data.keys() if "model" in str(item))
        ]
        print("=> using pre-trained model '{}'".format(model_description))
    else:
        network_data = None
        print("=> creating new FAL_netB stage 2 model.")

    model = FAL_netB(network_data, no_levels=args.no_levels, device=device).to(device)
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
            params=param_groups, lr=args.lr, betas=(args.momentum, args.beta)
        )
    g_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        g_optimizer, milestones=args.milestones, gamma=0.5
    )

    vgg_loss = VGGLoss(device=device)
    scaler = GradScaler()

    for epoch in range(args.epochs):
        # train for one epoch
        loss, train_loss = train(
            args,
            train_loader,
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
            args.save_path,
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
    for i, (input_data0, _, _) in enumerate(train_loader):
        # Read training data
        left_view = input_data0[0].to(device)
        right_view = input_data0[1].to(device)
        max_disp = (
            torch.Tensor([args.max_disp * args.relative_baseline])
            .repeat(args.batch_size)
            .unsqueeze(1)
            .unsqueeze(1)
            .type(left_view.type())
        )
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
            if args.smooth > 0:
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
            loss = rec_loss + args.smooth * sm_loss + args.a_mr * mirror_loss
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
                batch_time.get_avg(), epoch_size, args.epochs - epoch, i
            )
            print(
                f"Epoch: [{epoch}][{i}/{epoch_size}] ETA {eta} Batch Time {batch_time}  Loss {losses} RecLoss {rec_losses}"
            )

        # End training epoch earlier if args.epoch_size != 0
        if i >= epoch_size:
            break

    return loss, losses.avg
