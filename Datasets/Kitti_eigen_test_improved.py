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
from .listdataset_test import ListDataset
from random import shuffle
import csv


def Kitti_eigen_test_improved(split, **kwargs):
    input_root = kwargs.pop('root')
    print("input_root", input_root)
    transform = kwargs.pop('transform', None)
    target_transform = kwargs.pop('target_transform', None)
    shuffle_test = kwargs.pop('shuffle_test', False)    

    with open('./Datasets/split/eigen_test_improved.txt') as eigen_test_improved_file:
        eigen_test_improved_reader = csv.reader(eigen_test_improved_file, delimiter=' ')
        eigen_test_improved_list = []
        for row in eigen_test_improved_reader:
            inputleft = input_root + "/raw/" + row[0].split("/")[1] +'/image_02/data/'+ row[1].zfill(10)+'.png'
            inputright = input_root + "/raw/" + row[0].split("/")[1] +'/image_03/data/'+ row[1].zfill(10)+'.png'
            groundtruthleft = input_root + "/raw/" + row[0].split("/")[1] +'/proj_depth/groundtruth/image_02/'+ row[1].zfill(10)+'.png'
            velodyneleft = input_root + "/raw/" + row[0].split("/")[1] +'/proj_depth/velodyne_raw/image_02/'+ row[1].zfill(10)+'.png'
            
            if os.path.isfile(inputleft) and os.path.isfile(inputright) and os.path.isfile(groundtruthleft) and os.path.isfile(velodyneleft):
                eigen_test_improved_list.append([[inputleft, inputright],[groundtruthleft, velodyneleft]])
        
        test_list = eigen_test_improved_list
        if len(test_list) != 652: 
            raise Exception(f"Could only load {len(test_list)} images from \"KITTI eigen test improved split\" of size 652.") 

        # todo
        test_list = test_list[:2]

    
    print("len(test_list)", len(test_list))
    
    if shuffle_test:
        shuffle(test_list)
    
    test_dataset = ListDataset(input_root, input_root, test_list, data_name='Kitti_eigen_test_improved',
                               disp=True, of=False,
                               transform=transform, target_transform=target_transform)

    return test_dataset, None
