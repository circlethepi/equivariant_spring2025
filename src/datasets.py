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
# 2025-03-27 MO: adjusted version to include upsampling
# TODO: move dataset stuff from there to this file?
class RotatedDataset(Dataset):
    """
    Custom Dataset class for rotating/filling images
    """
    def __init__(self, original_dataset, angles, 
                 upsample=None, downsample=None, fill=None):
        self.original_dataset = original_dataset
        self.angles = angles
        self.fill = fill

        if upsample is not None:
            assert upsample > 0, "upsample must be greater than 0"
        self.original_size = original_dataset[0][0].shape[-1]
        if downsample is not None:
            assert downsample > 0, "downsample must be greater than 0"

        self.downsample = downsample if downsample is not None else self.original_size
        self.upsample = upsample if upsample is not None else self.original_size

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        angle = random.choice(self.angles)

        # upsample
        up_img = transforms.functional.resize(img, (self.upsample, self.upsample)) \
              if self.upsample > img.shape[-1] else img
        # rotate       
        rotated_img = rotate_tensor(up_img, angle, self.fill) 

        # downsample
        down_img = transforms.functional.resize(rotated_img, (self.downsample, self.downsample))
        
        return down_img, label


def rotate_tensor(tensor, angle, fill):
    return transforms.functional.rotate(tensor, angle, fill=fill)


def calculate_possible_angles(increment, comma_loc=0):
    """
    Helper function to calculate angle list from a given increment
    Angles calculated from 0 to 360 degrees
    """
    # TODO: comma location other than 0
    assert increment != 0
    angles = [int(k) for k in np.arange(0, 360, increment)]
    return angles 


def random_rotate_dataset(dataloader, increment=None, angles=None,
                          upsample_size=None, fill=None):
    """
    Generates a rotated version of a dataloader
    """
   # get angles
    angles = angles if angles is not None else calculate_possible_angles(increment)
    original_dataset = dataloader.dataset
    original_size = original_dataset[0][0].shape[-1] 

    if fill == True:
        fill = torch.min(dataloader.dataset[0][0]).item() 

    # new dataset
    rotated_dataset = RotatedDataset(original_dataset, angles, upsample=upsample_size,
                                     downsample=original_size, fill=fill)

    return torch.utils.data.DataLoader(rotated_dataset, batch_size=dataloader.batch_size,
                                       shuffle=False, num_workers=0, pin_memory=True)
            


def get_datasets(dataset_name: str, greyscale: bool=False, image_size=None):
    # TODO: add in augmentations / group actions (or maybe those go in make transforms or something)
    """get train and val datasets from params"""

    train_transforms = []
    test_transforms = []
    both_transforms = []

    # Normalization 
    if dataset_name == '90deg_mnist' or dataset_name == '45deg_mnist':
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
        # 2025-02-27 MO: removing additional random rotation here - can add back if we want later

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

    # TODO: model after notebook, get better handling 
    if args.dataset == '90deg_mnist':
        angles = list(range(0, 360, 90))
    elif args.dataset == '45deg_mnist':
        angles = list(range(0, 360, 45))
    elif args.data_rt_inc is not None:
        angles = calculate_possible_angles(args.data_rt_inc)
    else:
        angles = None
    
    if args.data_rt_fill is not None and args.data_rt_fill == True:
        fill = torch.min(train_loader.dataset[0][0]).item() # get the minimum value in the dataset to fill with
    else:
        fill = None


    if angles is not None:
        train_loader = random_rotate_dataset(train_loader, angles=angles, 
                            fill=fill)
        val_loader = random_rotate_dataset(val_loader, angles=angles, 
                            fill=fill)
        test_loader = random_rotate_dataset(test_loader, angles=angles,
                            fill=fill)
        

    return train_loader, val_loader, test_loader



# getting dataloaders for notebook environment / testing
def notebook_dataloaders(dataset_name="mnist", batch_size=256, greyscale=False,
                         angles=None, increment=None, 
                         upsample_size=None, fill=None):

    train_set, test_set = get_datasets(dataset_name=dataset_name, 
                                          greyscale=greyscale)
    
    #Adding a validation set
    train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42)
    train_load = get_dataloader(train_set, batch_size=batch_size, shuffle=True)

    #Added a val loader
    val_load = get_dataloader(val_set, batch_size=batch_size, shuffle=False)
    test_load = get_dataloader(test_set, batch_size=batch_size, shuffle=False)

    
    if dataset_name == '90deg_mnist': # predetermined
        angles = list(range(0, 360, 90))

    if increment is not None: # if given increment
        angles = angles if angles is not None else \
            calculate_possible_angles(increment)
    
    # random rotate parameters
    if fill == True:
        fill = torch.min(train_load.dataset[0][0]).item()

    if angles is not None: # rotate
        train_load = random_rotate_dataset(train_load, angles=angles, 
                            fill=fill, upsample_size=upsample_size)
        val_load = random_rotate_dataset(val_load, angles=angles, 
                            fill=fill, upsample_size=upsample_size)
        test_load = random_rotate_dataset(test_load, angles=angles,
                            fill=fill, upsample_size=upsample_size)
    
    return train_load, val_load, test_load

