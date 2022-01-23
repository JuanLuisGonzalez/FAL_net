import csv, pickle
import os.path as path
from misc.listdataset_test import ListDataset as TestListDataset

from misc.functional import flatten, apply
import numpy as np


def f(x):
    return {
        "eigen_test_improved": [[0, 1], [2, 3]],
        "eigen_test_classic": [[0], []],
        "eigen_train": [[0, 1], []],
        "bello_val": [[0, 1, 2, 3], [4, 5]],
    }[x]


def load_data(split=None, **kwargs):
    input_root = kwargs.pop("root")
    dataset = kwargs.pop("dataset")
    transform = kwargs.pop("transform", lambda x: x)
    create_val = kwargs.pop("create_val", False)

    if "ASM" in dataset:
        with open(path.join(input_root, dataset), "rb") as fp:
            datasetlist = pickle.load(fp)
    elif split is not None:
        splitfilelocation = f"./splits/{dataset}/{split}.txt"
        try:
            datasetfile = open(splitfilelocation)
        except:
            raise Exception(f"Could not open file at {splitfilelocation}.")

        datasetreader = csv.reader(datasetfile, delimiter=",")
        datasetlist = []
        for row in datasetreader:
            files = apply(f(split), lambda x: path.join(input_root, row[x]))
            for item in flatten(files):
                if item != None and not path.isfile(item):
                    raise Exception(f"Could not load file in location {item}.")
            datasetlist.append(files)

    # datasetlist = datasetlist[:100]

    dataset = TestListDataset(datasetlist, transform)
    if create_val:
        np.random.default_rng().shuffle(datasetlist, axis=0)
        val_size = int(len(datasetlist) * create_val)

        val_set = TestListDataset(datasetlist[:val_size], transform)
        dataset = TestListDataset(datasetlist[val_size:], transform)

        return dataset, val_set

    return dataset
