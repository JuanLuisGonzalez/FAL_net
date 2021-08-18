# listdataset_test.py Load training images during testing
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
import scipy.io as sio
from PIL import Image

LR_DATASETS = ["Kitti_eigen_test_improved"]


def Make3Ddisp_loader(input_root, path_img):
    disp = os.path.join(input_root, path_img)
    disp = sio.loadmat(disp, verify_compressed_data_integrity=False)
    disp = disp["Position3DGrid"][:, :, 3]
    disp = Image.fromarray(disp).resize((1704, 2272), resample=Image.NEAREST)
    disp = np.array(disp)
    return disp[:, :, np.newaxis]


def img_loader(input_root, path_img):
    imgs = os.path.join(input_root, path_img)
    return imread(imgs)


def kittidisp_loader(input_root, path_img):
    disp = os.path.join(input_root, path_img)
    disp = imread(disp) / 256
    return disp[:, :, np.newaxis]


def kittidepth_loader(input_root, path_depth):
    depth = np.load(os.path.join(input_root, path_depth))
    return depth[:, :, np.newaxis]


class ListDataset(data.Dataset):
    def __init__(
        self,
        input_root,
        target_root,
        path_list,
        disp=False,
        of=False,
        data_name="Kitti2015",
        transform=None,
        target_transform=None,
        co_transform=None,
    ):
        self.input_root = input_root
        self.target_root = target_root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.disp = disp
        self.of = of
        self.data_name = data_name

        if data_name == "Kitti2015" or data_name == "Kitti_eigen_test_improved":
            self.input_loader = img_loader
            if self.disp:
                self.target_loader = kittidisp_loader
        elif data_name == "Kitti_eigen_test_original":
            self.input_loader = img_loader
            if self.disp:
                self.target_loader = kittidepth_loader
        elif data_name == "Make3D":
            self.input_loader = img_loader
            if self.disp:
                self.target_loader = Make3Ddisp_loader

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]

        if self.data_name in LR_DATASETS:
            if self.disp:
                targets = [
                    self.target_loader(self.target_root, targets[0]),
                    self.target_loader(self.target_root, targets[1]),
                ]
        else:
            if self.disp:
                targets = [self.target_loader(self.target_root, targets[0])]

        file_name = os.path.basename(inputs[0])[:-4]
        inputs = [
            self.input_loader(self.input_root, inputs[0]),
            self.input_loader(self.input_root, inputs[1]),
        ]

        if self.co_transform is not None:
            inputs, targets = self.co_transform(inputs, targets)
        if self.transform is not None:
            for i in range(len(inputs)):
                inputs[i] = self.transform(inputs[i])
        if targets is None:
            return inputs, 0, file_name

        if self.target_transform is not None:
            for i in range(len(targets)):
                targets[i] = self.target_transform(targets[i])

        return inputs, targets, file_name
