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
                logfile, summaryfile,
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

        best_epochs=dict() #(train, val)_best -> (epoch, printed message)

        wandb.watch(model, criterion, log='all', log_freq=10) # TODO: add arg to change

    
        for epoch in range(1, epochs+1):
            # Validation 
            val_loss, val1, val5 = evaluate_model_loss(model, val_loader, criterion, 
                                              device, topk=(1,5,), 
                                              desc=f'Val epoch {epoch-1}/{epochs}')
            val_losses.append(val_loss)
            val_accuracies_1.append(val1)
            val_accuracies_5.append(val5)

            if epoch > 1:
                best_types = check_acc_loss_1_5(val_losses, val_accuracies_1, val_accuracies_5)
                if best_types is not None:
                    for best in best_types:
                        best_name = f'val_{best}'

                        savestate = {
                        'epoch': epoch-1,
                        'batch': global_batch_counter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'loss': running_loss / (global_batch_counter % len(train_loader)),
                        # 'loss' : train_loss.avg,
                        # 'train_accuracy_1': train1.avg,
                        # 'train_accuracy_5': train5.avg,
                        'val_loss': val_loss,
                        'val_accuracy_1': val1,
                        'val_accuracy_5': val5,
                        }

                        save_model(savestate, epoch-1, 'epoch', checkpoint_path, isbesttype=best_name)
                        
                        message = f'Val Loss: {train_loss.avg:.4f}, acc@1: {train1.avg:.2f}%, acc@5: {train5.avg:.2f}%'
                        best_epochs[best_name] = (epoch-1, message) # save to dict here

            message = f'Val Loss: {val_loss:.4f}, acc@1: {val1:.2f}%, acc@5: {val5:.2f}%'
            print_and_write(message, logfile)
            
            # Training
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
                    savestate = {
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
                    }

                    wandb_save = {
                    'epoch': epoch-1,
                    'batch': global_batch_counter,
                    'loss' : train_loss.avg,
                    'train_accuracy_1': train1.avg,
                    'train_accuracy_5': train5.avg,
                    'val_loss': val_loss,
                    'val_accuracy_1': val1,
                    'val_accuracy_5': val5,
                    }

                    save_model(savestate, interval=global_batch_counter, 
                               checkpoint_type=checkpoint_type, 
                               checkpoint_path=checkpoint_path,

                            wandb_log=True, wandb_save=wandb_save, 
                            wandb_step=global_batch_counter)



            # train_loss = running_loss / len(train_loader)
            # train_accuracy = 100. * correct / total
            train_losses.append(train_loss.avg)
            train_accuracies_1.append(train1.avg)
            train_accuracies_5.append(train5.avg)

            # epoch save
            if save and checkpoint_type=='epoch' and nice_interval(epoch-1):
                savestate = {
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
                }
                wandb_save = {
                    'epoch': epoch-1,
                    'batch': global_batch_counter,
                    'loss' : train_loss.avg,
                    'train_accuracy_1': train1.avg,
                    'train_accuracy_5': train5.avg,
                    'val_loss': val_loss,
                    'val_accuracy_1': val1,
                    'val_accuracy_5': val5,
                }
                save_model(savestate, epoch-1, checkpoint_type, 
                           checkpoint_path=checkpoint_path,

                           wandb_log=True, wandb_save=wandb_save, 
                           wandb_step=global_batch_counter)
                
            # check if best at each epoch
            if epoch > 1:
                best_types = check_acc_loss_1_5(train_losses, train_accuracies_1, train_accuracies_5)
                if best_types is not None:
                    for best in best_types:
                        best_name = f'train_{best}'

                        savestate = {
                        'epoch': epoch-1,
                        'batch': global_batch_counter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'loss': running_loss / (global_batch_counter % len(train_loader)),
                        'loss' : train_loss.avg,
                        'train_accuracy_1': train1.avg,
                        'train_accuracy_5': train5.avg,
                        # 'val_loss': val_loss,
                        # 'val_accuracy_1': val1,
                        # 'val_accuracy_5': val5,
                        }

                        save_model(savestate, epoch-1, 'epoch', checkpoint_path, isbesttype=best_name)

                        message = f'Train Loss: {train_loss.avg:.4f}, acc@1: {train1.avg:.2f}%, acc@5: {train5.avg:.2f}%'
                        best_epochs[best_name] = (epoch-1, message) # save to dict here

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
                }, epoch-1, checkpoint_type, checkpoint_path)
            
        
        message = f'Final Val Loss: {val_loss:.4f}, acc@1: {val1:.2f}%, acc@5: {val5:.2f}%'
        print_and_write(message, logfile)
        wandb.log({
                    'epoch': epoch-1,
                    'batch': global_batch_counter,
                    'loss' : train_loss.avg,
                    'train_accuracy_1': train1.avg,
                    'train_accuracy_5': train5.avg,
                    'val_loss': val_loss,
                    'val_accuracy_1': val1,
                    'val_accuracy_5': val5,
                }, step=global_batch_counter)
        
        write_summary(best_epochs, logfile, summaryfile)
        
        return train_losses, train_accuracies_1, train_accuracies_5, val_losses, val_accuracies_1, val_accuracies_5


def save_model(save_state, interval, checkpoint_type, checkpoint_path=None, 
               isbesttype=None,
               wandb_log=False, wandb_save=None, wandb_step=None):
    if checkpoint_path is not None:
        if isbesttype is not None:
            torch.save(save_state, checkpoint_path.replace('.pth.tar', f'_{checkpoint_type}_best{isbesttype}.pth.tar'))
        elif checkpoint_type == "batch":
            torch.save(save_state, checkpoint_path.replace('.pth.tar', f'_batch{interval}.pth.tar'))
        elif checkpoint_type == "epoch":
            torch.save(save_state, checkpoint_path.replace('.pth.tar', f'_epoch{interval}.pth.tar'))
    
    if wandb_log:
        wandb.log(wandb_save, step=wandb_step)
        pass


def check_is_best(list, checktype):
    check = list[-1]
    if check < min(list[:-1]) and checktype=='lo':
        return True
    elif check > max(list[:-1]) and checktype=='hi':
        return True
    else:
        return False


def check_acc_loss_1_5(loss, acc1, acc5):
    
    best_types = []
    if check_is_best(loss, 'lo'):
        best_types.append('loss')
    
    if check_is_best(acc1, 'hi'):
        best_types.append('acc1')
    
    if check_is_best(acc5, 'hi'):
        best_types.append('acc5')
    
    if len(best_types) < 1:
        return None
    else:
        return best_types
    

def write_summary(best_dict, *files):
    message = ""
    
    for key, val in best_dict.items():
        item_message = f'best_{key}: epoch {val[0]}\n{val[1]}\n\n'
        message += item_message
    
    print_and_write(message, *files)

    return



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
          model_savefilename, checkpoint_type, logfile, summaryfile):

    # get optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr) if \
        args.optimizer == 'Adam' else optim.SGD(model.parameters(), 
                                                lr=args.lr, 
                                                momentum=args.momentum)
    # get loss function
    criterion = nn.CrossEntropyLoss() if args.criterion == 'CrossEntropyLoss' \
        else nn.MSELoss()
    # get device
    device = get_device()

    print_and_write(f'Training epochs: {args.epochs}, Device: {device}', logfile)
    
    # send model to device
    model = model.to(device)

    train_losses, train_1, train_5, val_losses, val_1, val_5 = train_model(model=model, 
                train_loader=train_loader, val_loader=val_loader, 
                optimizer=optimizer, criterion=criterion, device=device, 
                epochs=args.epochs, logfile=logfile, checkpoint_type=checkpoint_type,
                checkpoint_path=model_savefilename, save=args.save_model, 
                summaryfile=summaryfile)
    
    # test model
    test_loss, test_1, test_5 = evaluate_model_loss(model, test_loader, 
                            criterion, device, topk=(1, 5), desc='Final Test', 
                            wandb_log=True, loader_name='test')
    
    message = f'Test after {args.epochs} epochs\nLoss: {test_loss:.4f}, acc@1: {test_1:.2f}%, acc@5: {test_5:.2f}%'
    print_and_write(message, logfile)

    close_files(logfile)

    return train_losses, train_1, train_5, val_losses, val_1, val_5, test_loss, test_1, test_5



