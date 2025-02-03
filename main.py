import src 
import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random

from utils import *
from globals import *



def build_parser():
    """
    Main parameters for running
    """
    parser = argparse.ArgumentParser()

    ## TODO: model parameters (no projection)
    parser.add_argument('--arch', type=str, nargs='*', help='architecture string')
    # parse architecture from string, build model input/output dimensions from datasets,

    ## TODO: dataset parameters
    # which dataset, input/output data dimensions (automatic in datasets?), data augmentation options

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


    return

# running
main()