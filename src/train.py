import wandb
from src import build_models, datasets, utils
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from tqdm import tqdm   # for progress bars
import os
import math
import json
import sys

from src import build_models as build, datasets
from src.utils import *


# training functionality for pre- and post- projection


def train_model(model, train_loader, val_loader, #args, 
                optimizer, criterion, device, epochs,
                logfile,
                checkpoint_type=None, checkpoint_path=None, save:bool=False, # can convert args to this maybe
                ):
        """
        Training loop that also logs information

        :param model:
        :param train_loader:
        :param val_loader:
        
        :param logfile:
        :param checkpoint_type:
        :param_checkpoint_path:
        :param_save: 
        """
        if save:
            assert checkpoint_type in ("batch", "epoch")
            assert checkpoint_path is not None

        train_losses = []
        train_accuracies_1 = []
        train_accuracies_5 = []

        val_losses = []
        val_accuracies_1 = []
        val_accuracies_5 = []
        global_batch_counter = 0 # counter for batches processed
        examples_seen = 0

        wandb.watch(model, criterion, log='all', log_freq=10) # TODO: add arg to change

    
        for epoch in range(1, epochs+1):

            # Validation 
            val_loss, val1, val5 = evaluate_model_loss(model, val_loader, criterion, 
                                              device, topk=(1,5,), 
                                              desc=f'Val epoch {epoch-1}/{epochs}')
            val_losses.append(val_loss)
            val_accuracies_1.append(val1)
            val_accuracies_5.append(val5)

            message = f'Val Loss: {val_loss:.4f}, acc@1: {val1:.2f}%, acc@5: {val5:.2f}%'
            print_and_write(message, logfile)
            
            # Train
    

        # Determine log-scaled batch intervals
        total_batches = len(train_loader) * epochs
        log_intervals = [int(math.pow(2, i)) for i in range(int(math.log2(total_batches)) + 1)]
        wandb.watch(model, criterion, log='all', log_freq=10)

        for epoch in range(epochs):
            model.train()
            train1 = AverageMeter('acc1')
            train5 = AverageMeter('acc5')
            train_loss = AverageMeter('loss')

            for inputs, labels in tqdm(train_loader, desc=f'Train epoch {epoch}/{epochs}'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                examples_seen += labels.size(0)
                global_batch_counter += 1

                # running_loss += loss.item()
                train_loss.update(loss.item(), outputs.size(0))
                train_acc1, train_acc5 = accuracy(outputs, labels, topk=(1,5))
                train1.update(train_acc1, outputs.size(0))
                train5.update(train_acc5, outputs.size(0))


                # Save checkpoint if required
                if save and checkpoint_type == "batch" and nice_interval(global_batch_counter):
                    save_model({
                        'epoch': epoch,
                        'batch': global_batch_counter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss.avg,
                        'train_accuracy_1': train1.avg,
                        'train_accuracy_5': train5.avg,
                        'val_loss': val_loss,
                        'val_accuracy_1': val1,
                        'val_accuracy_5': val5, # val accs from epoch (epoch-1)
                    }, global_batch_counter, checkpoint_type, checkpoint_path)
                
                    # print_and_write(f'Saved checkpoint at batch {global_batch_counter}')

            # train_loss = running_loss / len(train_loader)
            # train_accuracy = 100. * correct / total
            train_losses.append(train_loss.avg)
            train_accuracies_1.append(train1.avg)
            train_accuracies_5.append(train5.avg)

            # epoch save
            if save and checkpoint_type=='epoch' and nice_interval(epoch-1):
                save_model({
                    'epoch': epoch-1,
                    'batch': global_batch_counter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': running_loss / (global_batch_counter % len(train_loader)),
                    'loss' : train_loss.avg,
                    'train_accuracy_1': train1.avg,
                    'train_accuracy_5': train5.avg,
                    'val_loss': val_loss,
                    'val_accuracy_1': val1,
                    'val_accuracy_5': val5,
                }, epoch-1, checkpoint_type, checkpoint_path)
            
            message = f'Train Loss: {train_loss.avg:.4f}, acc@1: {train1.avg:.2f}%, acc@5: {train5.avg:.2f}%'
            print_and_write(message, logfile)

        # end of training val and save if applicable
        val_loss, val1, val5 = evaluate_model_loss(model, val_loader, criterion, device, topk=(1,5,), desc=f'Val epoch {epoch-1}')
        val_losses.append(val_loss)
        val_accuracies_1.append(val1)
        val_accuracies_5.append(val5)

        if save:
            save_model({
                    'epoch': epoch-1,
                    'batch': global_batch_counter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': running_loss / (global_batch_counter % len(train_loader)),
                    'loss' : train_loss.avg,
                    'train_accuracy_1': train1.avg,
                    'train_accuracy_5': train5.avg,
                    'val_loss': val_loss,
                    'val_accuracy_1': val1,
                    'val_accuracy_5': val5,
                }, epoch-1, checkpoint_type, checkpoint_path, force=True)
        
        message = f'Final Val Loss: {val_loss:.4f}, acc@1: {val1:.2f}%, acc@5: {val5:.2f}%'
        print_and_write(message, logfile)
        wandb.log({
                    # 'epoch': epoch-1,
                    # 'batch': global_batch_counter,
                    # 'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': running_loss / (global_batch_counter % len(train_loader)),
                    'loss' : train_loss.avg,
                    'train_accuracy_1': train1.avg,
                    'train_accuracy_5': train5.avg,
                    'val_loss': val_loss,
                    'val_accuracy_1': val1,
                    'val_accuracy_5': val5,
                }, step=global_batch_counter)
        
        return train_losses, train_accuracies_1, train_accuracies_5, val_losses, val_accuracies_1, val_accuracies_5


def save_model(save_state, interval, checkpoint_type, checkpoint_path, wandb_log=False):
    if checkpoint_type == "batch":
        torch.save(save_state, checkpoint_path.replace('.pth.tar', f'_batch{interval}.pth.tar'))
    elif checkpoint_type == "epoch":
        torch.save(save_state, checkpoint_path.replace('.pth.tar', f'_epoch{interval}.pth.tar'))
    
    if wandb_log:
        
        pass



def evaluate_model_loss(model, test_loader, criterion, device, topk=(1,), desc=None, 
                    print_acc=False, wandb_log=False, loader_name='test', 
                    step=None):

     # Test accuracy
    meters = [AverageMeter(name=f'acc{k}') for k in topk]
    loss_meter = AverageMeter(name='loss')
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=desc):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            accs = accuracy(outputs, labels, topk)
            # print(accs)
            for k in range(len(meters)):
                meters[k].update(accs[k], outputs.size(0))
            
            loss_meter.update(criterion(outputs, labels).item(), outputs.size(0))

    outstrings = [f'Top {topk[k]}: {meters[k].avg:.2f}%' for k in range(len(meters))]
    if print_acc:
        print('\n'.join(outstrings))

    vals = [loss_meter.avg] + [met.avg for met in meters]
    if wandb_log:
        keys = [f'{loader_name}_loss']+[f'{loader_name}_accuracy_{k}' for k in topk]
        logdict = dict(zip(keys, vals))
        wandb.log(logdict, step=step)

    return vals

    
def evaluate_model(model, test_loader, device, topk=(1,), step=None, desc=None, 
               print_acc=False, wandb_log=False, loader_name='test'):

    meters = [AverageMeter(name=f'acc{k}') for k in topk]
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=desc):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            accs = accuracy(outputs, labels, topk)
            # print(accs)
            for k in range(len(meters)):
                meters[k].update(accs[k], outputs.size(0))
    
    outstrings = [f'Top {topk[k]}: {meters[k].avg:.2f}%' for k in range(len(meters))]

    if print_acc:
        print('\n'.join(outstrings))
    
    vals = [met.avg for met in meters]
    if wandb_log:
        keys = [f'{loader_name}_accuracy_{k}' for k in topk]
        
        logdict = dict(zip(keys, vals))
        wandb.log(logdict, step=step)

    return vals


def accuracy(output, target, topk=(1,)):
    """top k precision for specifed k values"""
    maxk = max(topk)
    batch_size=target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul(100.0/batch_size).item())

    return res



# train: take in args
def train(args, model, train_loader, val_loader, test_loader, 
          model_savefilename, checkpoint_type, logfile):

    # get optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr) if \
        args.optimizer == 'Adam' else optim.SGD(model.parameters(), 
                                                lr=args.lr, 
                                                momentum=args.momentum)
    # get loss function
    criterion = nn.CrossEntropyLoss() if args.criterion == 'CrossEntropyLoss' \
        else nn.MSELoss()
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if \
                          torch.backends.mps.is_available() else "cpu")

    print_and_write(f'Training epochs: {args.epochs}, Device: {device}', logfile)
    
    # send model to device
    model = model.to(device)

    train_losses, train_1, train_5, val_losses, val_1, val_5 = train_model(model=model, 
                train_loader=train_loader, val_loader=val_loader, 
                optimizer=optimizer, criterion=criterion, device=device, 
                epochs=args.epochs, logfile=logfile, checkpoint_type=checkpoint_type,
                checkpoint_path=model_savefilename, save=args.save_model)
    
    # test model
    test_loss, test_1, test_5 = evaluate_model_loss(model, test_loader, 
                            criterion, device, topk=(1, 5), desc='Final Test', 
                            wandb_log=True, loader_name='test')
    
    message = f'Test after {args.epochs} epochs\nLoss: {test_loss:.4f}, acc@1: {test_1:.2f}%, acc@5: {test_5:.2f}%'
    print_and_write(message, logfile)

    close_files(logfile)

    return train_losses, train_1, train_5, val_losses, val_1, val_5, test_loss, test_1, test_5



