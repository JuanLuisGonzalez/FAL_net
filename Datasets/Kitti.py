# Kitti.py Create dataset for loading the KITTI dataset
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
from .listdataset_train import ListDataset as panListDataset
from random import shuffle


def Kitti(split, **kwargs):
    input_root = kwargs.pop('root')
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    reference_transform = kwargs.pop('reference_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    shuffle_test_data = kwargs.pop('shuffle_test_data', False)
    max_pix = kwargs.pop('max_pix', 100)
    fix = kwargs.pop('fix', False)
    train_split = kwargs.pop('train_split', 'eigen_train_split')

    # From Eigen et. al (NeurIPS 2014)
    if train_split == 'eigen_train_split':
        with open("Datasets/kitti_eigen_train.txt", 'r') as f:
            train_list = list(f.read().splitlines())
            train_list = [[line.split(" "), None] for line in train_list if
                          os.path.isfile(os.path.join(input_root, line.split(" ")[0]))]
    # From Godard et. al (CVPR 2017)
    elif train_split == 'kitti_train_split':
        with open("Datasets/kitti_train_files.txt", 'r') as f:
            train_list = list(f.read().splitlines())
            train_list = [[line.split(" "), None] for line in train_list if
                          os.path.isfile(os.path.join(input_root, line.split(" ")[0]))]

    train_list, test_list = split2list(train_list, split)

    train_dataset = panListDataset(input_root, input_root, train_list, data_name='Kitti2015', disp=False, of=False,
                                   transform=transform, target_transform=target_transform,
                                   co_transform=co_transform,
                                   max_pix=max_pix, reference_transform=reference_transform, fix=fix)
    if shuffle_test_data:
        shuffle(test_list)
    test_dataset = panListDataset(input_root, input_root, test_list, data_name='Kitti2015', disp=False, of=False,
                                  transform=transform, target_transform=target_transform, fix=fix)
    return train_dataset, test_dataset


def Kitti_list(split, **kwargs):
    input_root = kwargs.pop('root')
    with open('kitti_train_files.txt', 'r') as f:
        train_list = list(f.read().splitlines())
    train_list, test_list = split2list(train_list, split)
    return train_list, test_list
