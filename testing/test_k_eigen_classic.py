import os

import numpy as np

import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
from misc.dataloader import load_data

from models.FAL_netB import FAL_netB
from misc import utils, data_transforms
from misc.postprocessing import ms_pp
from testing.test_k_eigen_classic_utils import main as eval_kitti


def main(args, device="cpu"):
    print("-------Predicting on " + str(device) + "-------")

    input_transform = data_transforms.ApplyToMultiple(
        transforms.Compose(
            [
                # transforms.Resize(
                #     size=(256, 512), interpolation=transforms.InterpolationMode.BILINEAR
                # ),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.3606, 0.3789, 0.3652], std=[0.3123, 0.3173, 0.3216]
                # ),
                transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1]),
                transforms.Resize(
                    size=(256, 512), interpolation=transforms.InterpolationMode.NEAREST
                ),
            ]
        )
    )

    input_path = os.path.join(args.data_directory, args.dataset)
    test_dataset = load_data(
        split=args.test_split,
        dataset=args.dataset,
        root=input_path,
        transform=input_transform,
    )

    print("len(predict_dataset)", len(test_dataset))
    # Torch Data Loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
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

    if args.save_path:
        pickle_path = os.path.join(args.save_path, "pickle")
        if args.pickle_predictions and not os.path.exists(pickle_path):
            os.makedirs(pickle_path, exist_ok=True)

    # Set the max disp
    right_shift = args.max_disp * args.relative_baseline

    disparities = np.zeros((697, 256, 512), dtype=np.float32)

    with torch.no_grad():
        for i, ([input], _, _) in enumerate(test_loader):
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

            disparities[i] = disp.squeeze().cpu().numpy()

        # Hotfix to allow the original eigen test to be used.
        # This is needed as monodepth disparities have a scale from 9.135863e-06 to 0.15535712,
        # while falnet disparities have a scale from 2.0244858 to 172.61395.
        # I don't know the reason for this discrepancy, but for now I am rescaling the disparities.
        # ToDo: calculate the depth as defined in falnet and compare on depth directly,
        # as both monodepth and falnet should have the same scale for depth values.

        def rescale(value, rmin, rmax, tmin, tmax):
            return ((value - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin

        rescaled_disparities = rescale(
            disparities,
            np.min(disparities),
            np.max(disparities),
            9.135863e-06,
            0.15535712,
        )
        result = eval_kitti(rescaled_disparities, input_path)
        print(result)
        with open(os.path.join(args.save_path, "results.txt"), "w") as f:
            f.write(result)

        if args.pickle_predictions:
            np.save(os.path.join(pickle_path, "predictions.npy"), disparities)
