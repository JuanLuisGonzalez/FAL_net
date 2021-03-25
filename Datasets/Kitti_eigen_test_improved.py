# Kitti_eigen_test_improved.py Create dataset for loading the improved kitti eigen test split
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


import os.path
from .util import split2list
from .listdataset_test import ListDataset
from random import shuffle


def Kitti_eigen_test_improved(split, **kwargs):
    input_root = kwargs.pop('root')
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    shuffle_test = kwargs.pop('shuffle_test', False)

    with open("Datasets/kitti_eigen_test_improved.txt", 'r') as f:
        train_list = list(f.read().splitlines())
        train_list = [[line.split(" "),
                       [os.path.join(line.split(" ")[0][0:-29], 'proj_depth', 'groundtruth', 'image_02',
                                     line.split(" ")[0][-14:]),
                        os.path.join(line.split(" ")[0][0:-29], 'proj_depth', 'velodyne_raw', 'image_02',
                                     line.split(" ")[0][-14:])]]
                      for line in train_list if (os.path.isfile(os.path.join(input_root,
                                                                             line.split(" ")[0][0:-29],
                                                                             'proj_depth', 'groundtruth', 'image_02',
                                                                             line.split(" ")[0][-14:]))
                                                 and os.path.isfile(os.path.join(input_root, line.split(" ")[0])))]

    train_list, test_list = split2list(train_list, split)

    train_dataset = ListDataset(input_root, input_root, train_list, data_name='Kitti_eigen_test_improved',
                                disp=True, of=False,
                                transform=transform, target_transform=target_transform, co_transform=co_transform)
    if shuffle_test:
        shuffle(test_list)

    test_dataset = ListDataset(input_root, input_root, test_list, data_name='Kitti_eigen_test_improved',
                               disp=True, of=False,
                               transform=transform, target_transform=target_transform)

    return train_dataset, test_dataset
