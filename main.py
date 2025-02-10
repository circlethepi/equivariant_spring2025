from src import build_models, datasets, train 
import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random

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

    parser.add_argument('--batch-norm', action='store_true', 
                        help='Include batch norm')
    parser.add_argument('--bias', action='store_true', 
                        help='add bias to conv layers')
    # parse architecture from string, build model input/output dimensions from datasets,

    # classifier parameters
    parser.add_argument('--avgpool', action='store_const', 
                        default=True, const=True,
                        help='use AveragePool for classifier (default: True)')
    parser.add_argument('--avgpool-size', type=int, nargs='*',
                        default=[1, 1], help='avgpool kernel size')
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
    parser.add_argument('--seed', type=int, help='seed for model initialization')
    # epoch, random seed, 

    ## TODO: fine tuning/projection parameters
    # group/group action selection
    # fine tuning training parameter



    return parser

# get the args
parser = build_parser()

def get_args(*args_to_parser):
    """preprocessing args"""
    args = parser.parse_args(*args_to_parser)

    if args.seed is not None:
        set_seed(args.seed)

    return args


def main():
    args = get_args()
    do_code(args)


# the actual things we want to do
def do_code(args):

    # load datasets
    
    # build/load model

    # train model
    print(args.classifier_layers)


    return

# running
main()