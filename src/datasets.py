import os
import pickle

import torch.utils.data.dataset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2 as v2
import torchvision.datasets as datasets

from PIL import Image

# I am adding a validation set here
from sklearn.model_selection import train_test_split

from src.utils import *
from globals import *
import zipfile
import urllib.request

# load in the dataset based on arg parameters

class MnistRotDataset(Dataset):
            
            def __init__(self, mode, transform=None, extract_path=global_data_dir):
                assert mode in ['train', 'test']
                    
                if mode == "train":
                    file = os.path.join(extract_path, "mnist_all_rotation_normalized_float_train_valid.amat")
                else:
                    file = os.path.join(extract_path, "mnist_all_rotation_normalized_float_test.amat")
                
                self.transform = transform
    
                data = np.loadtxt(file, delimiter=' ')
                    
                self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
                self.labels = data[:, -1].astype(np.int64)
                self.num_samples = len(self.labels)
            
            def __getitem__(self, index):
                image, label = self.images[index], self.labels[index]
                image = Image.fromarray(image)
                if self.transform is not None:
                    image = self.transform(image)
                return image, label
            
            def __len__(self):
                return len(self.labels)

# 2025-02-27 MO: bringing this here to add as a training dataset option
# copied directly from averaging.py
# TODO: move dataset stuff from there to this file?
class RotatedDataset(Dataset):
    def __init__(self, original_dataset, angles):
        self.original_dataset = original_dataset
        self.angles = angles

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        angle = random.choice(self.angles)

        rotated_img = rotate_tensor(img, angle)
        return rotated_img, label

def rotate_tensor(tensor, angle):
    return transforms.functional.rotate(tensor, angle)


def random_rotate_dataset(dataloader, angles=[0, 90, 180, 270]):
    original_dataset = dataloader.dataset
    rotated_dataset = RotatedDataset(original_dataset, angles)
    return torch.utils.data.DataLoader(rotated_dataset, batch_size=dataloader.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            

def get_datasets(dataset_name: str, greyscale: bool=False, image_size=None):
    # TODO: add in augmentations / group actions (or maybe those go in make transforms or something)
    """get train and val datasets from params"""

    train_transforms = []
    test_transforms = []
    both_transforms = []

    # Normalization 
    if dataset_name == '90deg_mnist':
        dataset_name = 'mnist'

    if dataset_name in ('mnist', 'rotated_mnist'):
        mean = [0.1307]
        std = [0.3081]
        # pad = transforms.Pad((0,0,1,1), fill = 0)
        #train_transforms = [pad]
        #test_transforms = [pad]
    elif dataset_name == 'rotated_mnist':
        mean = [0.1307]
        std = [0.3081]
        # 2025-02-27 MO: removing random rotation here - can add back if we want later
        # pad = transforms.Pad((0,0,1,1), fill = 0)
        # resize1 = transforms.Resize(87)
        # # resize2 = transforms.Resize(29)
        # resize2 = transforms.Resize(28) # back to original size? 
        # rotate = transforms.RandomRotation(180, interpolation=Image.BILINEAR, expand=False)
        # train_transforms = [pad, resize1, rotate, resize2]
        # test_transforms = [pad]
    elif greyscale:
        mean = [0.481]
        std = [0.239]
        # both_transforms.append(transforms.Grayscale()) # add greyscale
    else:
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    
    both_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    

    if greyscale:
        both_transforms.append(transforms.Grayscale())

    standard_datasets = dict(
        cifar10=datasets.CIFAR10,
        cifar100=datasets.CIFAR100,
        mnist=datasets.MNIST,
    ) 

    if dataset_name in standard_datasets:
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
        
    elif dataset_name == "rotated_mnist":
        # download the dataset
        """Dataset of rotated MNIST digits from http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"""
        """Augmentations taken from https://github.com/QUVA-Lab/e2cnn/blob/master/examples/model.ipynb"""

        url = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"
        
        zip_path = os.path.join(global_data_dir, "mnist_rotation_new.zip")
        extract_path = os.path.join(global_data_dir, "mnist_rotation_new")

        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(url, zip_path)

        if not os.path.exists(extract_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

        train_set = MnistRotDataset(mode = "train", transform=transforms.Compose(train_transforms+both_transforms),extract_path=extract_path)
        test_set = MnistRotDataset(mode = "test", transform=transforms.Compose(test_transforms+both_transforms),extract_path=extract_path)

    
    else:
        raise ValueError(f"dataset {dataset_name} not supported")

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

    if args.dataset == '90deg_mnist':
        train_loader = random_rotate_dataset(train_loader)
        val_loader = random_rotate_dataset(val_loader)
        test_loader = random_rotate_dataset(test_loader)

    return train_loader, val_loader, test_loader



# getting dataloaders for notebook environment / testing
def notebook_dataloaders(dataset_name="mnist", batch_size=256, greyscale=False):
    train_set, test_set = get_datasets(dataset_name=dataset_name, 
                                          greyscale=greyscale)
    
    #Adding a validation set

    train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42)

    train_load = get_dataloader(train_set, batch_size=batch_size, shuffle=True)

    #Added a val loader
    val_load = get_dataloader(val_set, batch_size=batch_size, shuffle=False)

    test_load = get_dataloader(test_set, batch_size=batch_size, shuffle=False)

    if dataset_name == '90deg_mnist':
        train_load = random_rotate_dataset(train_load)
        val_load = random_rotate_dataset(val_load)
        test_load = random_rotate_dataset(test_load)
    
    return train_load, val_load, test_load

