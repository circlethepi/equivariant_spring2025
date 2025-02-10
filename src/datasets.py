import os
import pickle

import torch.utils.data.dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.utils import *
from globals import *

# load in the dataset based on arg parameters

def get_datasets(dataset_name: str, greyscale: bool, image_size=None):
    # TODO: add in aumentations / group actions (or maybe those go in make transforms or something)
    """get train and val datasets from params"""

    train_transforms = []
    test_transforms = []
    both_transforms = []

    # if dataset_name == "mnist":
    #     if image_size is None:
    #         image_size = 28
    
    # elif dataset_name == "cifar":
    #     if image_size is None:
    #         image_size = 32
    # TODO: move input image size calculation in module builder

    # Normalization 
    if dataset_name == 'mnist':
        mean = [0.1307]
        std = [0.3081]
    elif greyscale:
        mean = [0.481]
        std = [0.239]
        both_transforms.append(transforms.Grayscale())
    else:
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    both_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    standard_datasets = dict(
        cifar10=datasets.CIFAR10,
        cifar100=datasets.CIFAR100,
        mnist=datasets.MNIST,
    ) 
    standard_dataset = standard_datasets[dataset_name]

    def get_dataset(train : bool):
        transform_list = train_transforms if train else test_transforms
        dataparams=dict(
            root= global_data_dir,
            transform=transforms.Compose(transform_list+both_transforms),
            train=train,
            download=True,
        )

        dataset = standard_dataset(**dataparams)
    
        return dataset
    
    train_set = get_dataset(train=True)
    test_set = get_dataset(train=False)

    n_classes = 100 if dataset_name == "cifar100" else 10

    return train_set, test_set, n_classes



# TODO: additional custom transformations / data augmentations / group actions

def additional_transforms(train_set, test_set, transforms):
    """additional transforms if we want them"""

    return train_set, test_set


def get_dataloader(dataset, batch_size, shuffle):
    """get dataloader from dataset"""
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)


def get_dataloaders(args, logfile=None, summaryfile=None, log=True):
    """get train and val dataloaders from args
    
    adds dataset info to log and training summary files if included
    """
    dataset_name = args.dataset

    if log:
        assert logfile is not None
        assert summaryfile is not None
        dataset_message = f'using dataset {dataset_name}'
        print_and_write(dataset_message, logfile, summaryfile)

    train_set, test_set, n_classes = get_datasets(dataset_name=dataset_name, greyscale=args.greyscale)
    train_set, test_set = additional_transforms(train_set, test_set)

    train_loader = get_dataloader(train_set, args.batch_size, shuffle=True)
    test_loader = get_dataloader(test_set, args.batch_size, shuffle=False)

    return train_loader, test_loader


# getting dataloaders for notebook environment / testing
def notebook_dataloaders(dataset_name="mnist", batch_size=256, greyscale=False):
    train_set, test_set, _ = get_datasets(dataset_name=dataset_name, greyscale=greyscale)
    train_load = get_dataloader(train_set, batch_size=batch_size, shuffle=True)
    test_load = get_dataloader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_load, test_load

