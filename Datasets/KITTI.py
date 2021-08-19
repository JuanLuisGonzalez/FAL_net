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
import csv


def Kitti(split, **kwargs):
    input_root = kwargs.pop("root")
    transform = kwargs.pop("transform", None)
    target_transform = kwargs.pop("target_transform", None)
    reference_transform = kwargs.pop("reference_transform", None)
    co_transform = kwargs.pop("co_transform", None)
    max_pix = kwargs.pop("max_pix", 100)
    fix = kwargs.pop("fix", False)
    train_split = kwargs.pop("train_split", "eigen_train_split")

    # From Eigen et. al (NeurIPS 2014)
    if train_split == "eigen_train_split":
        with open("./Datasets/split/eigen_train.txt", "r") as eigen_train_file:
            eigen_train_reader = csv.reader(eigen_train_file, delimiter=" ")
            eigen_train_list = []
            for row in eigen_train_reader:
                inputleft = (
                    input_root
                    + "/raw/"
                    + row[0].split("/")[1]
                    + "/image_02/data/"
                    + row[1].zfill(10)
                    + ".png"
                )
                inputright = (
                    input_root
                    + "/raw/"
                    + row[0].split("/")[1]
                    + "/image_03/data/"
                    + row[1].zfill(10)
                    + ".png"
                )
                if os.path.isfile(inputleft) and os.path.isfile(inputright):
                    eigen_train_list.append([[inputleft, inputright], None])

            train_list = eigen_train_list
            if len(train_list) != 45200:
                raise Exception(
                    f'Could only load {len(train_list)} images from "KITTI eigen test improved split" of size 45200.'
                )

    train_dataset = panListDataset(
        input_root,
        input_root,
        train_list,
        data_name="Kitti2015",
        disp=False,
        of=False,
        transform=transform,
        target_transform=target_transform,
        co_transform=co_transform,
        max_pix=max_pix,
        reference_transform=reference_transform,
        fix=fix,
    )
    return train_dataset, None
