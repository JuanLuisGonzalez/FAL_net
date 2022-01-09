import csv, pickle
import os.path as path
from misc.listdataset_test import ListDataset as TestListDataset
from misc.listdataset_train import ListDataset as TrainListDataset
from misc.listdataset_retrain import ListDataset as RetrainListDataset
from misc.listdataset_run import ListDataset as EigentestListDataset


from random import shuffle
from misc.utils import flatten
import numpy as np


def load_data(split=None, **kwargs):
    input_root = kwargs.pop("root")
    dataset = kwargs.pop("dataset")
    transform = kwargs.pop("transform", None)
    target_transform = kwargs.pop("target_transform", None)
    shuffle_test = kwargs.pop("shuffle_test", False)
    reference_transform = kwargs.pop("reference_transform", None)
    co_transform = kwargs.pop("co_transform", None)
    max_pix = kwargs.pop("max_pix", 100)
    fix = kwargs.pop("fix", False)
    of_arg = kwargs.pop("of", False)
    disp_arg = kwargs.pop("disp", False)
    create_val = kwargs.pop("create_val", False)

    if split == "eigen_test_improved" and dataset == "KITTI":
        splitfilelocation = "./splits/KITTI/eigen_test_improved.txt"
    elif split == "eigen_test_classic" and dataset == "KITTI":
        splitfilelocation = "./splits/KITTI/eigen_test_classic.txt"
    elif split == "eigen_train" and dataset == "KITTI":
        splitfilelocation = "./splits/KITTI/eigen_train.txt"
    elif split == "bello_val" and dataset == "KITTI2015":
        splitfilelocation = "./splits/KITTI2015/bello_val.txt"

    if dataset == "ASM_stereo_small_train":
        with open(path.join(input_root, dataset), "rb") as fp:
            datasetlist = pickle.load(fp)
    elif dataset == "ASM_stereo_small_test":
        with open(path.join(input_root, dataset), "rb") as fp:
            datasetlist = pickle.load(fp)
    elif dataset == "ASM_stereo_train":
        with open(path.join(input_root, dataset), "rb") as fp:
            datasetlist = pickle.load(fp)
    elif dataset == "ASM_stereo_test":
        with open(path.join(input_root, dataset), "rb") as fp:
            datasetlist = pickle.load(fp)
    elif split is not None:
        try:
            datasetfile = open(splitfilelocation)
        except:
            raise Exception(f"Could not open file at {splitfilelocation}.")

        datasetreader = csv.reader(datasetfile, delimiter=",")
        datasetlist = []
        for i, row in enumerate(datasetreader):
            if split == "eigen_test_improved" and dataset == "KITTI":
                inputleft = f"{input_root}{row[0]}"
                inputright = f"{input_root}{row[1]}"
                groundtruthleft = f"{input_root}{row[2]}"
                velodyneleft = f"{input_root}{row[3]}"
                files = [[inputleft, inputright], [groundtruthleft, velodyneleft]]

            if split == "eigen_test_classic" and dataset == "KITTI":
                info = row[0].split(" ")
                inputleft = f"/{info[0].split('/')[1]}/image_02/data/{info[1]}.png"
                inputleft = f"{input_root}{inputleft}"
                files = [inputleft]

            elif split == "eigen_train" and dataset == "KITTI":
                inputleft = f"{input_root}{row[0]}"
                inputright = f"{input_root}{row[1]}"
                files = [[inputleft, inputright], None]

            elif split == "bello_val" and dataset == "KITTI2015":
                inputleft_t0 = f"{input_root}/{row[0]}"
                inputright_t0 = f"{input_root}/{row[1]}"
                inputleft_t1 = f"{input_root}/{row[2]}"
                inputright_t1 = f"{input_root}/{row[3]}"
                disp = f"{input_root}/{row[4]}"
                of = f"{input_root}/{row[5]}"
                files = [
                    [inputleft_t0, inputright_t0, inputleft_t1, inputright_t1],
                    [disp, of],
                ]

            if all(
                map(lambda x: True if x == None else path.isfile(x), flatten(files))
            ):
                datasetlist.append(files)
            else:
                for item in flatten(files):
                    if item != None and not path.isfile(item):
                        raise Exception(f"Could not load file in location {item}.")

    if shuffle_test and isinstance(datasetlist, list):
        shuffle(datasetlist)
    elif shuffle_test and isinstance(datasetlist, np.ndarray):
        np.random.default_rng().shuffle(datasetlist, axis=0)

    if split == "eigen_test_improved" and dataset == "KITTI":
        dataset = TestListDataset(
            datasetlist,
            data_name="Kitti_eigen_test_improved",
            disp=True,
            of=False,
            transform=transform,
            target_transform=target_transform,
        )
    if split == "eigen_test_classic" and dataset == "KITTI":
        dataset = EigentestListDataset(
            path_list=flatten(datasetlist),
            transform=transform,
        )

    elif split == "eigen_train" and dataset == "KITTI":
        dataset = TrainListDataset(
            input_root,
            input_root,
            datasetlist,
            disp=False,
            of=False,
            transform=transform,
            co_transform=co_transform,
            max_pix=max_pix,
            reference_transform=reference_transform,
            fix=fix,
        )
    elif split == "bello_val" and dataset == "KITTI2015":
        dataset = TestListDataset(
            datasetlist,
            data_name="Kitti2015",
            disp=disp_arg,
            of=of_arg,
            transform=transform,
            target_transform=target_transform,
        )
    elif dataset == "ASM_stereo_small_train" or dataset == "ASM_stereo_train":
        if create_val:
            np.random.default_rng().shuffle(datasetlist, axis=0)
            val_size = int(len(datasetlist) * create_val)
            val_list = datasetlist[:val_size]
            datasetlist = datasetlist[val_size:]

            val_set = RetrainListDataset(
                val_list,
                transform=transform,
                max_pix=max_pix,
            )

            dataset = RetrainListDataset(
                datasetlist,
                transform=transform,
                max_pix=max_pix,
            )
            return dataset, val_set
        else:
            dataset = RetrainListDataset(
                datasetlist,
                transform=transform,
                max_pix=max_pix,
            )
    elif dataset == "ASM_stereo_small_test" or dataset == "ASM_stereo_test":
        dataset = RetrainListDataset(
            datasetlist,
            transform=transform,
        )

    return dataset
