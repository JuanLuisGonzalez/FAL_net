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

import random
import torch
from torchvision import transforms
from torchvision.transforms.functional import hflip
import numpy as np


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
