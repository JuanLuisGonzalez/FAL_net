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
from PIL import Image


class ListDataset(data.Dataset):
    def __init__(
        self,
        img_array,
        transform=None,
        max_pix=100,
    ):
        self.img_array = img_array
        self.transform = transform
        self.max_pix = max_pix

    def __len__(self):
        return len(self.img_array)

    def __getitem__(self, index):
        inputs = self.img_array[index]

        inputs = np.moveaxis(inputs, 1, -1)
        inputs = [inputs[0], inputs[1]]
        inputs[0] = Image.fromarray(inputs[0])
        inputs[1] = Image.fromarray(inputs[1])

        if self.transform is not None:
            inputs = self.transform(inputs)

        # print("inputs.shape", np.asarray(inputs).shape)
        return inputs, self.max_pix
