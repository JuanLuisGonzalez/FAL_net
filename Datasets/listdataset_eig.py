# listdataset_eig.py Load training images for eigen test split
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

import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np


def img_loader(input_root, path_imgs, index):
    imgs = [os.path.join(input_root, path) for path in path_imgs]
    return imread(imgs[index])


def depth_loader(input_root, path_depth):
    depth = np.load(os.path.join(input_root, path_depth))
    return depth[:, :, np.newaxis]


class ListDataset(data.Dataset):
    def __init__(self, input_root, target_root, path_list, transform=None, target_transform=None, co_transform=None):
        self.input_root = input_root
        self.target_root = target_root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

        self.input_loader = img_loader
        self.target_loader = depth_loader

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]
        file_name = os.path.basename(inputs[0])[:-4]

        inputs = [self.input_loader(self.input_root, inputs, 0),
                  self.input_loader(self.input_root, inputs, 1)]

        targets = [self.target_loader(self.input_root, targets[0])]

        if self.co_transform is not None:
            inputs, targets = self.co_transform(inputs, targets)
        if self.transform is not None:
            for i in range(len(inputs)):
                inputs[i] = self.transform(inputs[i])
        if self.target_transform is not None:
            for i in range(len(targets)):
                targets[i] = self.target_transform(targets[i])

        return inputs, targets, file_name
