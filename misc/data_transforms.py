# data_transforms.py Contains various useful data transformations
# in a double sized batch input
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

import random, numbers
import torch
from torchvision import transforms
from torchvision.transforms.functional import hflip
import numpy as np
from PIL import Image


class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target=None):
        for t in self.co_transforms:
            input, target = t(input, target)
        return input, target


class ArrayToTensor(object):
    def __call__(self, array):
        assert isinstance(array, np.ndarray)
        if len(array.shape) == 3 and (array.shape[-1] == 3 or array.shape[-1] == 1):
            array = np.moveaxis(array, -1, 0)
        tensor = torch.from_numpy(array.copy())
        return tensor.float()


class RandomResizeCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, down, up):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.s_factor = (down, up)

    def __call__(self, inputs, targets=None):
        h, w, _ = inputs[0].shape
        th, tw = self.size

        min_factor = max(
            max((th + 1) / h, (tw + 1) / w), self.s_factor[0]
        )  # plus one to ensure
        max_factor = self.s_factor[1]
        factor = np.random.uniform(low=min_factor, high=max_factor)

        for i in range(len(inputs)):
            input = Image.fromarray(inputs[i]).resize(
                (int(w * factor), int(h * factor)), resample=Image.BICUBIC
            )
            inputs[i] = np.array(input)
        if targets is not None:
            for i in range(len(targets)):
                target = Image.fromarray(targets[i]).resize(
                    (int(w * factor), int(h * factor)), resample=Image.BICUBIC
                )
                targets[i] = np.array(target)

        h, w, _ = inputs[0].shape
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for i in range(len(inputs)):
            inputs[i] = inputs[i][y1 : y1 + th, x1 : x1 + tw]
        if targets is not None:
            for i in range(len(targets)):
                targets[i] = targets[i][y1 : y1 + th, x1 : x1 + tw]
        return inputs, targets


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    if doing this on disparity estimation you need both disparities left and right need stereo targets
    """

    def __init__(self):
        return

    def __call__(self, inputs, targets=None):
        o_inputs = []

        if random.random() < 0.5:
            o_inputs.append(np.copy(np.fliplr(inputs[1])))
            o_inputs.append(np.copy(np.fliplr(inputs[0])))
            if targets is not None:
                o_target = []
                o_target.append(np.copy(np.fliplr(targets[1])))
                o_target.append(np.copy(np.fliplr(targets[0])))
                return o_inputs, o_target
            else:
                return o_inputs, targets
        else:
            return inputs, targets


class RandomGamma(object):
    def __init__(self, min=1, max=1):
        self.min = min
        self.max = max
        self.A = 255

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            factor = random.uniform(self.min, self.max)
            for i in range(len(inputs)):
                inputs[i] = self.A * ((inputs[i] / 255) ** factor)
            return inputs, targets
        else:
            return inputs, targets


class RandomBrightness(object):
    def __init__(self, min=0, max=0):
        self.min = min
        self.max = max

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            factor = random.uniform(self.min, self.max)
            for i in range(len(inputs)):
                inputs[i] = inputs[i] * factor
                inputs[i][inputs[i] > 255] = 255
            return inputs, targets
        else:
            return inputs, targets


class RandomCBrightness(object):
    def __init__(self, min=0, max=0):
        self.min = min
        self.max = max

    def __call__(self, inputs, targets=None):
        if random.random() < 0.5:
            for i in range(len(inputs)):
                for c in range(3):
                    factor = random.uniform(self.min, self.max)
                    inputs[i][:, :, c] = inputs[i][:, :, c] * factor
                inputs[i][inputs[i] > 255] = 255
            return inputs, targets
        else:
            return inputs, targets


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std = std.eq(0).mul(1e-7).add(std)
        std_inv = 1 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)


class ApplyToMultiple:
    def __init__(
        self,
        transform,
        RandomHorizontalFlipChance=0,
        same_rand_state=True,
    ):
        self.transform = transform
        self.same_rand_state = same_rand_state
        self.RandomHorizontalFlipChance = RandomHorizontalFlipChance

    def _apply_to_features(self, transform, input, same_rand_state):
        if same_rand_state:

            # move randomness
            np.random.rand()
            random.random()
            torch.rand(1)

            # save state
            np_state = np.random.get_state()
            rd_state = random.getstate()
            tr_state = torch.random.get_rng_state()

        intermediate = input
        if self.RandomHorizontalFlipChance:
            if torch.rand(1) < self.RandomHorizontalFlipChance:
                intermediate = [hflip(x) for x in input]
                intermediate.reverse()
            torch.set_rng_state(tr_state)

        output = []
        for item in intermediate:
            output.append(transform(item))

            if same_rand_state:
                np.random.set_state(np_state)
                random.setstate(rd_state)
                torch.set_rng_state(tr_state)

        return output

    def __call__(self, input_list):
        return self._apply_to_features(self.transform, input_list, self.same_rand_state)
