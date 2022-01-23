import os, sys

import torch
import torchvision.transforms as transforms

from misc.dataloader import load_data
from misc import data_transforms


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for idx, (inputs, _, _) in enumerate(dataloader):
        print(f"Processed batch {idx} of {len(dataloader)}.")
        for inp in inputs:
            # Mean over batch, height and width, but not over the channels
            channels_sum += torch.mean(inp, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(inp ** 2, dim=[0, 2, 3])

        num_batches += 2

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def main(args, _):
    print(" ".join(sys.argv[:]))

    # Set up data augmentations
    input_transform = data_transforms.ApplyToMultiple(
        transforms.Compose(
            [
                transforms.Resize(
                    size=(375, 1241), interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.ToTensor(),
            ]
        )
    )

    # Torch Data Set List
    input_path = os.path.join(args.data_directory, "KITTI")
    mean_dataset = load_data(
        dataset="KITTI",
        split="eigen_train",
        root=input_path,
        transform=input_transform,
    )

    mean_loader = torch.utils.data.DataLoader(
        mean_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=False,
    )

    print("len(mean_loader): ", len(mean_loader))

    print("len(mean_dataset): ", len(mean_dataset))

    print("mean & std: ", get_mean_and_std(mean_loader))
