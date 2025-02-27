from src import datasets, train 
import src.build_models as build

import argparse
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import json

import wandb

from src.utils import *
from globals import *



def build_parser():
    """
    Main parameters for running
    """
    parser = argparse.ArgumentParser()

    ## TODO: model parameters (no projection)
    parser.add_argument('--arch', type=str, nargs='*', 
                        help='architecture string')
    # parser.add_argument('--n_classes', type=int, nargs='*', help='number of classes')
    # added way to get from dataset in build_models ::MO 2025-02-13

    parser.add_argument('--batch-norm', action='store_true', 
                        help='Include batch norm')
    parser.add_argument('--bias', action='store_true', 
                        help='add bias to conv layers')
    parser.add_argument('--normalize-weights', action='store_false',
                        help='weight normalization')
    # parse architecture from string, build model input/output dimensions from datasets,

    # classifier parameters
    parser.add_argument('--avgpool', action='store_const', 
                        default=True, const=True,
                        help='use AveragePool for classifier (default: True)')
    parser.add_argument('--avgpool-size', type=int, 
                        default=1, help='avgpool kernel size')
    parser.add_argument('--classifier-layers', type=int, nargs='*', 
                        help='linear classifier layers',
                        default=[4096, 4096, 4096])
    parser.add_argument('--classifier-bias', action='store_const', 
                        default=True, const=True, 
                        help='add bias to classifier (default: True)' )
    parser.add_argument('--no-classifier-bias', action='store_false', 
                        dest='classifier-bias', 
                        help='remove bias from the classifier')
    parser.add_argument('--classifier-droupout', type=float, default=0., 
                        help='classifier droupout rate')

    ## TODO: dataset parameters
    # which dataset, input/output data dimensions (automatic in datasets?), data augmentation options
    data_choices = ['cifar', 'mnist']
    parser.add_argument('--dataset', type=str, choices=data_choices, help='dataset name') # which dataset
    parser.add_argument('--greyscale', '--grayscale', type=bool, default=False) # whether to make greyscale

    ## TODO: training parameters
    parser.add_argument('--seed', type=int, help='seed for model initialization', default=0)
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='momentum for the optimizer (default: 0.9, only for optimizers that use momentum)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, 
                        help='weight decay for the optimizer (default: 0.0001)')

    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', 
                        help='loss function to use (default: CrossEntropyLoss)')
    # epoch, random seed, batch size, optimizer, learning rate, momentum, weight decay, scheduler, loss function, early stopping, etc.

    # scheduler parameters # not currently implemented
    parser.add_argument('--scheduler', type=str, default='StepLR', 
                        help='scheduler to use (default: StepLR)')
    parser.add_argument('--step-size', type=int, default=30,
                        help='step size for the scheduler (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='gamma for the scheduler (default: 0.1)')
    
    # early stopping parameters # not currently implemented
    parser.add_argument('--early-stopping', action='store_true',
                        help='use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stopping (default: 10)')
    parser.add_argument('--min-improvement', type=float, default=0.01,
                        help='minimum improvement for early stopping (default: 0.01)')
    parser.add_argument('--restore-best', action='store_true',
                        help='restore best model for early stopping')
    
    # logging parameters
    parser.add_argument('--save-model', action='store_const', const=True, 
                        default=True, help='save the model')
    parser.add_argument('--dont-save-model', action='store_false', 
                        dest='save-model', help="don't save model")
    parser.add_argument('--save-path', type=str, default=global_save_dir,
                        help='path to save the model (default: global_save_dir - see globals.py)')
    parser.add_argument('--load-model', action='store_true',
                        help='load the model')
    parser.add_argument('--load-path', type=str, default=global_save_dir,
                        help='path to load the model (default: global_save_dir - see globals.py)')
    
    parser.add_argument('--log-path', type=str, default=global_log_dir,
                        help='path for log files (default: global_log_dir - see globals.py)')

    parser.add_argument('--custom-name', type=str, help='custom save name')
    naming_choices = ['custom', 'default', 'both']
    parser.add_argument('--name-convention', type=str, choices=naming_choices,
                        help='how to name the model dir', default='default')
    # wandb logging
    parser.add_argument('--wandb-proj', type=str, help='wandb project name')
    

    # model checkpoint paramters
    parser.add_argument('--save-epoch', action='store_true',
                        help='save by epoch (default: save at batch counts of form [1,2,5]eK )')

    ## TODO: fine tuning/projection parameters
    # group/group action selection
    # fine tuning training parameter



    return parser

# get the args
parser = build_parser()

def get_args(*args_to_parser):
    """preprocessing args"""
    args = parser.parse_args(*args_to_parser)

    # set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # set up wandb logging
    if args.wandb_proj is not None:
        wandb.init(
            project = args.wandb_proj,
            config = args,
            name=build.get_model_savename(args)
        )

    return args


def main():
    args = get_args()
    do_code(args)


# the actual things we want to do
def do_code(args):

    # create saving/checkpoint configuration
    # get checkpoint info
    model_savefilename, checkpoint_type, logfile, summaryfile = \
        build.parse_checkpoint_log_info(args)

    # load datasets
    train_loader, val_loader, test_loader = datasets.get_dataloaders(args, 
                                                            logfile=logfile)

    # build/load model
    model = build.build_model_from_args(args)

    # train model
    loss_accs = train.train(args, model, train_loader, val_loader, test_loader,
                            model_savefilename, checkpoint_type, 
                            logfile, summaryfile)

    if args.wandb_proj is not None:
        wandb.finish()

    return loss_accs


# running
main()