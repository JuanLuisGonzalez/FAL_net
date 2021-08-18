import numpy as np


def split2list(images, split=0.9):
    if split == 0:
        return [], images
    elif split == 1:
        return images, []
    elif isinstance(split, float):
        split_values = np.random.uniform(0, 1, len(images)) < split
        train_samples = [sample for sample, split in zip(images, split_values) if split]
        test_samples = [
            sample for sample, split in zip(images, split_values) if not split
        ]
        return train_samples, test_samples
