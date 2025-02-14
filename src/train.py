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
        glboal_batch_counter = 0 # counter for batches processed
    
        for epoch in range(1, epochs+1):

            # Validation 
            val_loss, val1, val5 = test_model(model, val_loader, device, topk=(1,5,), desc=f'Val epoch {epoch-1}')
            val_losses.append(val_loss)
            val_accuracies_1.append(val1)
            val_accuracies_5.append(val5)

            message = f'val epoch {epoch-1}/{epochs}\nVal Loss: {val_loss}, val@1: {val1:.2f}%, val@5: {val5:.2f}%'
            print_and_write(message, logfile)
            
            # Train
            model.train()
            train1 = AverageMeter('acc1')
            train5 = AverageMeter('acc5')
            train_loss = AverageMeter('loss')

            for inputs, labels in tqdm(train_loader, desc=f'Train epoch {epoch}'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update batch counter
                global_batch_counter += 1

                # running_loss += loss.item()
                train_loss.update(loss.item(), outputs.size(0))
                train_acc1, train_acc5 = accuracy(outputs, labels, topk=(1,5))
                train1.update(train_acc1[0], outputs.size(0))
                train5.update(train_acc5[0], outputs.size(0))


                # Save checkpoint if required
                if save and checkpoint_type == "batch":
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
                    }, glboal_batch_counter, checkpoint_type, checkpoint_path)
                
                    # print_and_write(f'Saved checkpoint at batch {global_batch_counter}')

            # train_loss = running_loss / len(train_loader)
            # train_accuracy = 100. * correct / total
            train_losses.append(train_loss.avg)
            train_accuracies_1.append(train1.avg)
            train_accuracies_5.append(train5.avg)

            # epoch save
            if save and checkpoint_type=='epoch':
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
            
            message = f'train epoch {epoch}/{epochs}\nLoss: {train_loss.avg:.4f}, train@1: {train1.avg:.2f}%, train@5: {train5.avg:.2f}%'
            print_and_write(message, logfile)

        # end of training val and save if applicable
        val_loss, val1, val5 = test_model(model, val_loader, device, topk=(1,5,), desc=f'Val epoch {epoch-1}')
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
        
        message = f'final val epoch {epoch-1}/{epochs}\nVal Loss: {val_loss}, val@1: {val1:.2f}%, val@5: {val5:.2f}%'
        print_and_write(message, logfile)
        
        return train_losses, train_accuracies_1, train_accuracies_5, val_losses, val_accuracies_1, val_accuracies_5


def save_model(save_state, interval, checkpoint_type, checkpoint_path, force=False):
    if nice_interval(interval) or interval == 0 or force:
        if checkpoint_type == "batch":
            torch.save(save_state, checkpoint_path.replace('.pth.tar', f'_batch{interval}.pth.tar'))
        elif checkpoint_type == "epoch":
            torch.save(save_state, checkpoint_path.replace('.pth.tar', f'_epoch{interval}.pth.tar'))



def test_model(model, test_loader, criterion, device, topk=(1,), desc=None, print_acc=False):
     # Test accuracy
    meters = [AverageMeter(name=f'acc{k}') for k in topk]
    loss_meter = AverageMeter(name='loss')
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=desc):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            accs = accuracy(outputs, labels, topk)
            for k in range(len(topk)):
                meters[k].update(accs[k][0], outputs.size(0))
            
            loss_meter.update(criterion(outputs, labels).item(), outputs.size(0))

    outstrings = [f'Top {topk[k]}: {meters[k].avg:.2f}%' for k in range(len(topk))]
    if print_acc:
        print('\n'.join(outstrings))

    return [loss_meter.avg] + [meters[k].avg for k in range(len(meters))] 


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
        res.append(correct_k.mul(100.0/batch_size))

    return res

     

def parse_checkpoint_log_info(args):
    basic_train_info = f'{args.dataset}_batchsize{args.batch_size}'
    model_savename = build.get_model_savename(args)

    model_savedir = os.path.join(args.save_path, model_savename)
    checkpoint_filename = f'{basic_train_info}.pth.tar'
    checkpoint_type = "epoch" if args.save_epoch else "batch"
    
    if not os.path.exists(model_savedir):
        os.makedirs(model_savedir)

    # TODO: train loop take in this file, replace extension to include
    # batch number, then save model state dict to the file
    model_savefilename = os.path.join(model_savedir, checkpoint_filename)

    # logging configuration
    log_savedir = os.path.join(args.log_path, model_savename)
    if not os.path.exists(log_savedir):
        os.makedirs(log_savedir)
    logfile = make_logfile(os.path_join(log_savedir, f'{basic_train_info}.log'))

    # save commandline entry to log
    with open(os.path.join(model_savedir, 'args.json'), 'w') as file:
        json.dump(args.__dict__, file, indent=2, default=str) 
    print_and_write(f"Command line: {' '.join(sys.argv)}", logfile)

    return model_savefilename, checkpoint_type, logfile



# train: take in args
def train(args, model, train_loader, val_loader, test_loader):

    # get checkpoint info
    model_savefilename, checkpoint_type, logfile = parse_checkpoint_log_info(args)

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
    test_loss, test_1, test_5 = test_model(model, test_loader, criterion, device, topk=(1, 5),
                                           desc='Final Test')
    message = f'Test @{args.epochs}\nLoss: {test_loss}, test@1: {test_1:.2f}%, test@5: {test_5:.2f}%'
    print_and_write(message, logfile)

    close_files(logfile)

    return train_losses, train_1, train_5, val_losses, val_1, val_5, test_loss, test_1, test_5

## build model from model builder
    # create log file
## train for args.epochs (TODO: add to args)
## log model states/ loss/ accuracy at nice batch checkpoints
## save model to disk


