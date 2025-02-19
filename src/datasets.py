import os
import pickle

import torch.utils.data.dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2 as v2
import torchvision.datasets as datasets

# I am adding a validation set here
from sklearn.model_selection import train_test_split

from src.utils import *
from globals import *

# load in the dataset based on arg parameters

def get_datasets(dataset_name: str, greyscale: bool, image_size=None):
    # TODO: add in aumentations / group actions (or maybe those go in make transforms or something)
    """get train and val datasets from params"""

    train_transforms = []
    test_transforms = []
    both_transforms = []

    # Normalization 
    if dataset_name == 'mnist':
        mean = [0.1307]
        std = [0.3081]
    elif greyscale:
        mean = [0.481]
        std = [0.239]
        #Reduce channels to 1
        both_transforms.append(transforms.Grayscale())
    else:
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    both_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            #added a grayscale or RGB transform
            transforms.Grayscale() if (greyscale or dataset_name == 'mnist') else transforms.v2.RGB()
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

    return train_set, test_set



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
        # assert summaryfile is not None
        dataset_message = f'using dataset {dataset_name}'
        print_and_write(dataset_message, logfile)

    train_set, test_set = get_datasets(dataset_name=dataset_name, greyscale=args.greyscale)
    
    #train_set, test_set = additional_transforms(train_set, test_set, transforms= None)
    #Adding a validation set
    train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=args.seed)
    train_loader = get_dataloader(train_set, args.batch_size, shuffle=True)

    #Added a val loader
    val_loader = get_dataloader(val_set, args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_set, args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



# getting dataloaders for notebook environment / testing
def notebook_dataloaders(dataset_name="mnist", batch_size=256, greyscale=False):
    train_set, test_set, _ = get_datasets(dataset_name=dataset_name, greyscale=greyscale)
    
    #Adding a validation set

    train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42)

    train_load = get_dataloader(train_set, batch_size=batch_size, shuffle=True)

    #Added a val loader
    val_load = get_dataloader(val_set, batch_size=batch_size, shuffle=False)

    test_load = get_dataloader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_load, val_load, test_load

