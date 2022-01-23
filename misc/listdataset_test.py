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
from PIL import Image
import torch


def kittidisp_loader(path_img):
    disp = imread(path_img) / 256
    disp = disp[np.newaxis, :, :]
    disp = torch.from_numpy(disp).float()
    return disp


class ListDataset(data.Dataset):
    def __init__(
        self,
        data_list,
        transform=lambda x: x,
    ):
        self.data_list = data_list
        self.transform = transform
        self.is_image_list = isinstance(data_list[0][0], Image.Image)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.is_image_list:
            inputs = self.data_list[index]
            file_name = str(index).zfill(10)
            targets = []
        else:
            inputs, targets = self.data_list[index]
            file_name = os.path.basename(inputs[0])[:-4]
            inputs = [Image.open(path_img) for path_img in inputs]
            targets = [kittidisp_loader(target) for target in targets]

        inputs = self.transform(inputs)

        return inputs, targets, file_name
