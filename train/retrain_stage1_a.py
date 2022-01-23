import os
import time
import numpy as np

from misc.dataloader import load_data
from models.FAL_netB import FAL_netB

import torch
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from misc import utils, data_transforms
from misc.loss_functions import smoothness, VGGLoss


def main(args, device="cpu"):
    print("-------Training Stage 1 on " + str(device) + "-------")

    best_rec_loss = -1

    # Set output writters for showing up progress on tensorboardX
    train_writer = SummaryWriter(os.path.join(args.save_path, "train"))
    output_writers = []
    for i in range(3):
        output_writers.append(
            SummaryWriter(os.path.join(args.save_path, "test", str(i)))
        )

    # Set up data augmentations
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

    # Torch Data Set List
    input_path = os.path.join(args.data_directory, args.dataset)
    train_dataset0, val_dataset0 = load_data(
        dataset=args.dataset,
        root=input_path,
        transform=transform,
        create_val=0.1,
    )
    print("len(train_dataset0)", len(train_dataset0))
    print("len(val_dataset0)", len(val_dataset0))

    # Torch Data Loaders
    train_loader0 = torch.utils.data.DataLoader(
        train_dataset0,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=True,
    )
    print("len(train_loader0)", len(train_loader0))

    # Torch Data Loaders
    val_loader0 = torch.utils.data.DataLoader(
        val_dataset0,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=True,
    )
    print("len(val_loader0)", len(val_loader0))

    # create model
    model = FAL_netB(no_levels=args.no_levels, device=device)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if device.type == "cuda":
        print("torch.nn.DataParallel(model).to(device)")
        model = torch.nn.DataParallel(model).to(device)
    print("=> Number of parameters model '{}'".format(utils.get_n_params(model)))

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

    for epoch in range(args.start_epoch):
        g_scheduler.step()

    vgg_loss = VGGLoss(device=device)
    scaler = GradScaler()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        loss, train_loss, rec_loss = train(
            args, train_loader0, model, g_optimizer, epoch, device, vgg_loss, scaler
        )
        train_writer.add_scalar("train_loss", train_loss, epoch)

        # validate()

        # Apply LR schedule (after optimizer.step() has been called for recent pyTorch versions)
        g_scheduler.step()

        if best_rec_loss < 0:
            best_rec_loss = rec_loss
        is_best = rec_loss < best_rec_loss
        best_rec_loss = min(rec_loss, best_rec_loss)
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


def train(args, train_loader, model, g_optimizer, epoch, device, vgg_loss, scaler):
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
        max_disp = (
            torch.Tensor([args.max_disp * args.relative_baseline])
            .repeat(args.batch_size)
            .unsqueeze(1)
            .unsqueeze(1)
            .type(left_view.type())
        )
        _, _, _, W = left_view.shape

        # measure data loading time
        data_time.update(time.time() - end)

        # Reset gradients
        g_optimizer.zero_grad()

        with autocast():
            ###### LEFT disp
            min_disp = max_disp * args.min_disp / args.max_disp
            rpan, ldisp = model(
                input_left=left_view,
                min_disp=min_disp,
                max_disp=max_disp,
                ret_disp=True,
                ret_pan=True,
                ret_subocc=False,
            )
            # Compute rec loss

            if args.a_p > 0:
                vgg_right = vgg_loss.vgg(right_view)
            else:
                vgg_right = None

            # Over 2 as measured twice for left and right
            mask = 1
            rec_loss = vgg_loss.rec_loss_fnc(
                mask, rpan, right_view, vgg_right, args.a_p
            )
            rec_losses.update(rec_loss.detach().cpu(), args.batch_size)

            #  Compute smooth loss
            sm_loss = 0
            if args.smooth > 0:
                # Here we ignore the 20% left dis-occluded region, as there is no suppervision for it due to parralax
                sm_loss = smoothness(
                    left_view[:, :, :, int(0.20 * W) : :],
                    ldisp[:, :, :, int(0.20 * W) : :],
                    gamma=2,
                    device=device,
                )

            # compute gradient and do optimization step
            loss = rec_loss + args.smooth * sm_loss
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

    return loss, losses.avg, rec_loss
