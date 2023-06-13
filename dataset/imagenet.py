import os
import numpy as np

import torchvision

from copy import deepcopy

imagenet_root_dir = '/home/gui/Downloads/Datasets/imagenet'

imagenet_train_root_dir = ''

class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform):
        super(ImageNet, self).__init__(root, transform)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx
    
def subsample_dataset(dataset, idxs):
    dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset

def subsample_classes(dataset, include_classes=range(200)):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_split(train_dataset, val_split=0.2):
    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)
    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset(train_dataset, train_idxs)
    val_dataset = subsample_dataset(val_dataset, val_idxs)

    return train_dataset, val_dataset

def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """
    if len(dataset1) > len(dataset2):
        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)
    elif len(dataset2) > len(dataset1):
        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2

def get_imagenet_datasets(train_transform, test_transform, train_classes=range(200), open_set_classes=range(200, 1000), 
                           balance_open_set_eval=False, split_train_val=True, seed=0):
    np.random.seed(seed)
    # Init train dataset and subsample training classes
    train_dataset_whole = ImageNet(root=imagenet_root_dir, transform=train_transform)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    test_dataset_known = ImageNet(root=image_val_root_dir, transform=test_transform)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = ImageNet(root=image_val_root_dir, transform=test_transform)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets


if __name__ == '__main__':
    x = get_imagenet_datasets(None, None, balance_open_set_eval=False, split_train_val=False)
    print([len(v) for k, v in x.items()])
    debug = 0