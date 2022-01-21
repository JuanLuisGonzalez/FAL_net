import pickle, sys, os

import torch
from torchvision import transforms


def main(args, _):
    print(" ".join(sys.argv[:]))

    with open(
        os.path.join(args.data_directory, "ASM_stereo/ASM_stereo_test"), "rb"
    ) as fp:
        ASM_stereo_test = pickle.load(fp)

    with open(
        os.path.join(args.data_directory, "ASM_stereo/ASM_stereo_train"), "rb"
    ) as fp:
        ASM_stereo_train = pickle.load(fp)

    transform = transforms.ToTensor()

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for idx, (input_left, input_right) in enumerate(ASM_stereo_test + ASM_stereo_train):
        input_left = transform(input_left)
        input_right = transform(input_right)
        print(f"Processed batch {idx} of {len(ASM_stereo_test)+len(ASM_stereo_train)}.")
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(input_left, dim=[1, 2])
        channels_squared_sum += torch.mean(input_left ** 2, dim=[1, 2])
        channels_sum += torch.mean(input_right, dim=[1, 2])
        channels_squared_sum += torch.mean(input_right ** 2, dim=[1, 2])
        num_batches += 2

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    print("mean, std", mean, std)
