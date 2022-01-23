import os, pickle

import numpy as np

import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
from misc.listdataset_test import ListDataset as RunListDataset
import matplotlib.pyplot as plt

from models.FAL_netB import FAL_netB
from misc import utils, data_transforms
from misc.postprocessing import ms_pp


def predict(args, device="cpu"):
    print("-------Predicting on " + str(device) + "-------")

    input_transform = data_transforms.ApplyToMultiple(
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
            ]
        )
    )

    output_transform = transforms.Compose(
        [
            data_transforms.NormalizeInverse(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
            transforms.ToPILImage(),
        ]
    )

    # Torch Data Set List
    predict_dataset = RunListDataset(
        path_list=[[args.input], []]
        if os.path.isfile(args.input)
        else [
            [[os.path.join(args.input, x)], []]
            for x in sorted(next(os.walk(args.input))[2])
        ],
        transform=input_transform,
    )

    print("len(predict_dataset)", len(predict_dataset))
    # Torch Data Loader
    val_loader = torch.utils.data.DataLoader(
        predict_dataset,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=False,
    )

    print(args.model)

    pan_model = FAL_netB(no_levels=args.no_levels, device=device)
    checkpoint = torch.load(args.model, map_location=device)
    pan_model.load_state_dict(checkpoint["model_state_dict"])
    if device.type == "cuda":
        pan_model = torch.nn.DataParallel(pan_model).to(device)

    model_parameters = utils.get_n_params(pan_model)
    print("=> Number of parameters '{}'".format(model_parameters))
    cudnn.benchmark = True

    l_disp_path = os.path.join(args.save_path, "l_disp")
    if not os.path.exists(l_disp_path):
        os.makedirs(l_disp_path)

    input_path = os.path.join(args.save_path, "input")
    if args.save_input and not os.path.exists(input_path):
        os.makedirs(input_path, exist_ok=True)

    pickle_path = os.path.join(args.save_path, "pickle")
    if args.pickle_predictions and not os.path.exists(pickle_path):
        os.makedirs(pickle_path, exist_ok=True)
    if args.pickle_predictions:
        predicte_disparities = []

    # Set the max disp
    right_shift = args.max_disp * args.relative_baseline

    with torch.no_grad():
        for i, ([input], _, _) in enumerate(val_loader):
            input_left = input.to(device)
            B, C, H, W = input_left.shape

            # Prepare flip grid for post-processing
            i_tetha = torch.zeros(B, 2, 3).to(device)
            i_tetha[:, 0, 0] = 1
            i_tetha[:, 1, 1] = 1
            flip_grid = F.affine_grid(i_tetha, [B, C, H, W], align_corners=False)
            flip_grid[:, :, :, 0] = -flip_grid[:, :, :, 0]

            # Convert min and max disp to bx1x1 where b=1 tensors
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

            if args.save_input:
                print("save the input image in path", input_path)
                input_image = output_transform(input[0])
                input_image.save(os.path.join(input_path, "{:010d}.png".format(i)))

            if args.pickle_predictions:
                predicte_disparities.append(disp.squeeze().cpu().numpy())

        if args.pickle_predictions:
            pickle.dump(
                predicte_disparities,
                open(os.path.join(pickle_path, "predictions.pickle"), "wb"),
            )
