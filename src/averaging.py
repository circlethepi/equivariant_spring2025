import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
import wandb
from src import train
from src.utils import *

class RotatedDataset(Dataset):
    def __init__(self, original_dataset, angles):
        self.original_dataset = original_dataset
        self.angles = angles
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        # ])

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        angle = random.choice(self.angles)
        # rotated_img = self.transform(img).rotate(angle)
        # rotated_img = transforms.ToTensor()(rotated_img)
        rotated_img = rotate_tensor(img, angle)
        return rotated_img, label


def rotate_tensor(tensor, angle):
    # pil_image = transforms.ToPILImage()(tensor)
    # rotated_image = pil_image.rotate(angle)
    # return transforms.ToTensor()(rotated_image)
    return transforms.functional.rotate(tensor, angle)


def random_rotate_dataset(dataloader, angles=[0, 90, 180, 270]):
    original_dataset = dataloader.dataset
    rotated_dataset = RotatedDataset(original_dataset, angles)
    return DataLoader(rotated_dataset, batch_size=dataloader.batch_size, shuffle=False, num_workers=4, pin_memory=True)


def average_over90degrees_and_evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = []
            # Rotate inputs by 0, 90, 180, 270 degrees and run through the model
            for angle in [0,90,180,270]:
                rotated_inputs = torch.stack([rotate_tensor(img, angle) for img in inputs])
                rotated_inputs = rotated_inputs.to(device)
                output = model(rotated_inputs)
                outputs.append(output)

            # Average the outputs and divide by 4
            averaged_output = sum(outputs) /4
            _, predicted = averaged_output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_accuracy = 100. * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    wandb.log({"test_accuracy_after_averaging": test_accuracy})

    return test_accuracy


def average_and_evaluate(model, dataloader, device, topk=(1,), step=None, 
                         desc=None, actions:list=None, wandb_log=False,
                         loader_name='average', print_acc=False):
    """ version of train.evaluate_model that also averages over an input set of 
    actions

    :param dataloader: should be the unmodified dataset (as of now)
    :param actions: list of functions that take in an instance x and output the
    group action on x. ideally, all actions in a group (or a basis)
    """
    if actions is None:
        values = train.evaluate_model(model, dataloader, device, topk=topk,
                        step=step, desc=desc, print_acc=print_acc, 
                        wandb_log=wandb_log, loader_name=loader_name)
    
    else:
        meters = [AverageMeter(name=f'acc{k}') for k in topk]
        model.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=desc):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = []

                for action in actions:
                    actioned_inputs = torch.stack([action(x) for x in inputs]).to(device)
                    outputs.append(model(actioned_inputs))
                
                batch_averaged_output = torch.tensor(sum(outputs)/len(actions))
                accs = train.accuracy(batch_averaged_output, labels, topk)

                for k in range(len(meters)):
                    meters[k].update(accs[k], batch_averaged_output.size(0))
        
        outstrings = [f'Top {topk[k]}: {meters[k].avg:.2f}%' for k in range(len(meters))]

        if print_acc:
            print('\n'.join(outstrings))
        
        vals = [met.avg for met in meters]
        if wandb_log:
            keys = [f'{loader_name}_accuracy_{k}' for k in topk]
            logdict = dict(zip(keys, vals))
            wandb.log(logdict, step=step)
                
    return vals

# TODO: /wishlist item: Group class that will calculate the actions we want 
# TODO: conversion for layer-wise: will need to get out activations and apply 
# to each one?

class Group:

    def __init__(self, actions):
        self.actions = actions
    