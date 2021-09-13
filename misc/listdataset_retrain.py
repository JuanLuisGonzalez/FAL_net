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
import numpy as np
from skimage.transform import resize
from skimage.util import img_as_ubyte


class ListDataset(data.Dataset):
    def __init__(
        self,
        img_array,
        transform=None,
        co_transform=None,
        max_pix=100,
    ):
        self.img_array = img_array
        self.transform = transform
        self.co_transform = co_transform
        self.max = max_pix

    def __len__(self):
        return len(self.img_array)

    def __getitem__(self, index):
        inputs = self.img_array[index]

        x_pix = self.max
        inputs = np.moveaxis(inputs, 1, -1)
        inputs = [inputs[0], inputs[1]]
        inputs[0] = img_as_ubyte(
            resize(
                inputs[0],
                (inputs[0].shape[0] * 2, inputs[0].shape[1] * 2),
                anti_aliasing=True,
            )
        )
        inputs[1] = img_as_ubyte(
            resize(
                inputs[1],
                (inputs[1].shape[0] * 2, inputs[1].shape[1] * 2),
                anti_aliasing=True,
            )
        )

        # print("type(inputs[0])", type(inputs[0]))
        if self.co_transform is not None:
            inputs, _ = self.co_transform(inputs)
        if self.transform is not None:
            for i in range(len(inputs)):
                inputs[i] = self.transform(inputs[i])

        # print("inputs.shape", np.asarray(inputs).shape)
        return inputs, x_pix
