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

from __future__ import division
import torch
import random
import numpy as np
import numbers
from PIL import Image


class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input, target)
        return input, target


class ArrayToTensor(object):
    def __call__(self, array):
        assert (isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
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

        min_factor = max(max((th + 1) / h, (tw + 1) / w), self.s_factor[0])  # plus one to ensure
        max_factor = self.s_factor[1]
        factor = np.random.uniform(low=min_factor, high=max_factor)

        for i in range(len(inputs)):
            input = Image.fromarray(inputs[i]).resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
            inputs[i] = np.array(input)
        if targets is not None:
            for i in range(len(targets)):
                target = Image.fromarray(targets[i]).resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
                targets[i] = np.array(target)

        h, w, _ = inputs[0].shape
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for i in range(len(inputs)):
            inputs[i] = inputs[i][y1: y1 + th, x1: x1 + tw]
        if targets is not None:
            for i in range(len(targets)):
                targets[i] = targets[i][y1: y1 + th, x1: x1 + tw]
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
