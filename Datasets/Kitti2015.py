# Kitti2015.py Create dataset for loading the the KITTI2015 dataset
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


def make_dataset(main_dir, split, scene_flow=True, training=True):
    directories = []
    if training:
        mode = "training"
    else:
        mode = "testing"
    left_dir = os.path.join(mode, "image_2")
    right_dir = os.path.join(mode, "image_3")
    if scene_flow:
        disp_dir = os.path.join(mode, "disp_occ_0")
        of_dir = os.path.join(mode, "flow_occ")
        for i in range(200):  # 200 lr pairs
            imgl_t = os.path.join(left_dir, "%06d_10.png" % i)  # rgb input left
            imgr_t = os.path.join(right_dir, "%06d_10.png" % i)  # rgb input right
            imgl_t1 = os.path.join(left_dir, "%06d_11.png" % i)  # rgb input left
            imgr_t1 = os.path.join(right_dir, "%06d_11.png" % i)  # rgb input right
            disp = os.path.join(disp_dir, "%06d_10.png" % i)  # rgb input right
            of = os.path.join(of_dir, "%06d_10.png" % i)  # rgb input right

            # Check valid files
            if not (
                os.path.isfile(os.path.join(main_dir, imgl_t))
                and os.path.isfile(os.path.join(main_dir, imgr_t))
                and os.path.isfile(os.path.join(main_dir, imgl_t1))
                and os.path.isfile(os.path.join(main_dir, imgr_t1))
            ):
                continue
            directories.append([[imgl_t, imgr_t, imgl_t1, imgr_t1], [disp, of]])
    else:
        for i in range(200):  # 400 lr pairs
            imgl_t = os.path.join(left_dir, "%06d_10.png" % i)  # rgb input left
            imgr_t = os.path.join(right_dir, "%06d_10.png" % i)  # rgb input right
            imgl_t1 = os.path.join(left_dir, "%06d_11.png" % i)  # rgb input left
            imgr_t1 = os.path.join(right_dir, "%06d_11.png" % i)  # rgb input right

            # Check valid files
            if not (
                os.path.isfile(os.path.join(main_dir, imgl_t))
                and os.path.isfile(os.path.join(main_dir, imgr_t))
                and os.path.isfile(os.path.join(main_dir, imgl_t1))
                and os.path.isfile(os.path.join(main_dir, imgr_t1))
            ):
                continue
            directories.append([[imgl_t, imgr_t], None])
            directories.append([[imgl_t1, imgr_t1], None])

    return split2list(directories, split)


def Kitti2015(split, **kwargs):
    input_root = kwargs.pop("root")
    disp = kwargs.pop("disp", False)
    of = kwargs.pop("of", False)
    shuffle_test_data = kwargs.pop("shuffle_test", False)
    transform = kwargs.pop("transform", None)
    target_transform = kwargs.pop("target_transform", None)
    co_transform = kwargs.pop("co_transform", None)

    if disp or of:
        [train_list, test_list] = make_dataset(
            input_root, split, scene_flow=True, training=True
        )
    else:
        train_list = make_dataset(input_root, split, scene_flow=False, training=True)[0]
        test_list = make_dataset(input_root, split, scene_flow=False, training=False)[0]

    train_dataset = ListDataset(
        input_root,
        input_root,
        train_list,
        data_name="Kitti2015",
        disp=disp,
        of=of,
        transform=transform,
        target_transform=target_transform,
        co_transform=co_transform,
    )
    if shuffle_test_data:
        shuffle(test_list)
    test_dataset = ListDataset(
        input_root,
        input_root,
        test_list,
        data_name="Kitti2015",
        disp=disp,
        of=of,
        transform=transform,
        target_transform=target_transform,
    )
    return train_dataset, test_dataset


def Kitti2015_list(split, **kwargs):
    input_root = kwargs.pop("root")
    disp = kwargs.pop("disp", False)
    of = kwargs.pop("of", False)

    if disp or of:
        [train_list, test_list] = make_dataset(
            input_root, split, scene_flow=True, training=True
        )
    else:
        train_list = make_dataset(input_root, split, scene_flow=False, training=True)[0]
        test_list = make_dataset(input_root, split, scene_flow=False, training=False)[0]
    shuffle(test_list)
    return train_list, test_list
