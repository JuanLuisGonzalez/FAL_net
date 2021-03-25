# listdataset_train.py Load training images during training
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
import random

# Indexes for lr models
Lt_indexlr = 0
Rt_indexlr = 1
Lt1_indexlr = 2
Rt1_indexlr = 3
Dl_indexlr = 0
Dr_indexlr = 1
Ofl_indexlr = 2
Ofr_indexlr = 3

# indexes for l models
Lt_index = 0
Rt_index = 1
Lt1_index = 2
Rt1_index = 3
Dl_index = 0
Ofl_index = 1


def img_loader(input_root, path_imgs, index):
    imgs = [os.path.join(input_root, path) for path in path_imgs]
    return imread(imgs[index])


class ListDataset(data.Dataset):
    def __init__(self, input_root, target_root, path_list, disp=False, of=False, data_name='Monkaa', transform=None,
                 target_transform=None, co_transform=None, max_pix=100, reference_transform=None, fix=False):
        self.input_root = input_root
        self.target_root = target_root
        self.path_list = path_list
        self.transform = transform
        self.reference_transform = reference_transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.disp = disp
        self.of = of
        self.data_name = data_name
        self.input_loader = img_loader
        self.max = max_pix
        self.fix_order = fix

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]
        file_name = os.path.basename(inputs[Lt_index])[:-4]

        if random.random() < 0.5 or self.fix_order:
            x_pix = self.max
            inputs = [self.input_loader(self.input_root, inputs, Lt_index),
                      self.input_loader(self.input_root, inputs, Rt_indexlr)]
        else:
            x_pix = -self.max
            inputs = [self.input_loader(self.input_root, inputs, Rt_indexlr),
                      self.input_loader(self.input_root, inputs, Lt_index)]

        # x_pix = np.random.uniform(low=0, high=self.max)
        y_pix = np.random.uniform(low=-self.max, high=self.max)

        # file_name = os.path.basename(inputs[Lt_index])[:-4]
        # inputs = [self.input_loader(self.input_root, inputs, Lt_index),
        #           self.input_loader(self.input_root, inputs, Rt_indexlr)]

        if self.reference_transform is not None:
            inputs[0] = self.reference_transform(inputs[0])
        if self.co_transform is not None:
            inputs, _ = self.co_transform(inputs, targets)
        if self.transform is not None:
            for i in range(len(inputs)):
                inputs[i] = self.transform(inputs[i])

        return inputs, x_pix, y_pix, file_name
