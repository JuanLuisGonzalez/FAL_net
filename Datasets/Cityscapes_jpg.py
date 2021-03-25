# Cityscapes_jpg.py Create dataset for loading the CityScape dataset
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
import glob
from random import shuffle
from .listdataset_train import ListDataset as panListDataset


def make_dataset(main_dir):
    train_directories = []
    val_directories = []
    directories = (train_directories, val_directories)
    left_img_folder = os.path.join(main_dir, 'leftImg8bit')
    for ttv_dirs in os.listdir(left_img_folder):
        if os.path.isfile(os.path.join(left_img_folder, ttv_dirs)):
            continue
        if ttv_dirs == 'val':
            selector = 1 # put that in the validation folder
        else:
            selector = 0
        for city_dirs in os.listdir(os.path.join(left_img_folder, ttv_dirs)):
            if os.path.isfile(os.path.join(left_img_folder, ttv_dirs, city_dirs)):
                continue
            left_dir = os.path.join(left_img_folder, ttv_dirs, city_dirs)
            for target in glob.iglob(os.path.join(left_dir, '*.jpg')):
                target = os.path.basename(target)
                root_filename = target[:-15]  # remove leftImg8bit.png
                imgl_t = os.path.join('leftImg8bit', ttv_dirs, city_dirs, root_filename + 'leftImg8bit.jpg')  # rgb input left
                imgr_t = os.path.join('rightImg8bit', ttv_dirs, city_dirs, root_filename + 'rightImg8bit.jpg')  # rgb input right

                # Check valid files
                if not (os.path.isfile(os.path.join(main_dir, imgl_t))
                        and os.path.isfile(os.path.join(main_dir, imgr_t))):
                    continue
                directories[selector].append([[imgl_t, imgr_t], None])

    return directories[0], directories[1]


def Cityscapes_jpg(split, **kwargs):
    input_root = kwargs.pop('root')
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    reference_transform = kwargs.pop('reference_transform', None)
    co_transform = kwargs.pop('co_transform', None)
    shuffle_test_data = kwargs.pop('shuffle_test_data', False)
    max_pix = kwargs.pop('max_pix', 100)
    fix = kwargs.pop('fix', False)

    train_list, test_list = make_dataset(input_root, split)

    train_dataset = panListDataset(input_root, input_root, train_list, transform=transform,
                                   target_transform=target_transform, co_transform=co_transform,
                                   max_pix=max_pix, reference_transform=reference_transform, fix=fix)
    if shuffle_test_data:
        shuffle(test_list)
    test_dataset = panListDataset(input_root, input_root, test_list, transform=transform,
                                  target_transform=target_transform, fix=fix)
    return train_dataset, test_dataset


def Cityscapes_list_jpg(split, **kwargs):
    input_root = kwargs.pop('root')
    train_list, test_list = make_dataset(input_root, split)
    return train_list, test_list


