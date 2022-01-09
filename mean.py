import os, sys, argparse
import torch
import torchvision.transforms as transforms
from misc.dataloader import load_data
from misc import data_transforms


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for ([input_left, input_right], max_pix) in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(input_left, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(input_left ** 2, dim=[0, 2, 3])
        channels_sum += torch.mean(input_right, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(input_right ** 2, dim=[0, 2, 3])
        num_batches += 2

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def main():
    print(" ".join(sys.argv[:]))

    parser = argparse.ArgumentParser(
        description="Stereo Mean Std Calculation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-dd",
        "--data_directory",
        metavar="DIR",
        default="./data",
        help="Directory containing the datasets",
    )

    args = parser.parse_args()

    # Set up data augmentations
    input_transform = data_transforms.ApplyToMultiple(
        transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    )

    # Torch Data Set List
    input_path = os.path.join(args.data_directory, "ASM_stereo")
    train_dataset0 = load_data(
        dataset=args.dataset,
        root=input_path,
        transform=input_transform,
        max_pix=args.max_disp,
    )

    train_loader0 = torch.utils.data.DataLoader(
        train_dataset0,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=True,
    )

    print("len(train_dataset0)", len(train_dataset0))

    print("mean & std", get_mean_and_std(train_loader0))


if __name__ == "__main__":

    main()
