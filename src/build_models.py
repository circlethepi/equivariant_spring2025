import torch
import torch.nn as nn
from src.custom_blocks import ModelBuilder, CustomClassifier, CustomModel
import os
import sys
import json
import numpy as np
from src.utils import *

# loading models from disc /
# building models from arg parameters

# TODO: custom blocks:
# equivariant nonlinearities
# equivariance basis blocks

def get_input_dimensions(args):
    """get dataset dimensions (number of channels) and input image dimensions
    from args dataset information
    """
    if args.dataset.startswith("mnist"):
        channels_in = 1
    else:
        channels_in = 3
    if args.greyscale:
        channels_in = 1
    
    size_dict = {'cifar': 32, 'mnist': 28}
    image_size = size_dict[args.dataset]

    # class_dict = {'cifar100': 100,}
    n_classes = 100 if args.dataset == "cifar100" else 10
    # print(channels_in, image_size, n_classes)

    return channels_in, image_size, n_classes


def conv_output_size(conv_layer:nn.Module, im_input):
    """calculate conv output datasize for square images
    im_output = (im_input+2*padding-dilation*(kernel_size-1) -1)/stride + 1
    """
    assert conv_layer.padding is not None
    assert conv_layer.dilation is not None
    assert conv_layer.kernel_size is not None
    assert conv_layer.stride is not None
    
    pad = conv_layer.padding if isinstance(conv_layer.padding, int) else conv_layer.padding[0]
    dil = conv_layer.dilation if isinstance(conv_layer.dilation, int) else conv_layer.dilation[0]
    ker = conv_layer.kernel_size if isinstance(conv_layer.kernel_size, int) else conv_layer.kernel_size[0]
    sid = conv_layer.stride if isinstance(conv_layer.stride, int) else conv_layer.stride[0]

    im_output = np.floor(((im_input + 2*pad - dil*(ker - 1) -1)/sid) + 1)

    # print(im_input, pad, dil, ker, sid, '\nout:', im_output) 

    return im_output


#Changed model signature to just have args
def build_model_from_args(args):
    """build custom vgg-type architecture from arguments

    """
    # set seed
    set_seed(args.seed)

    custom_blocks = ModelBuilder()

    input_channels, image_size, n_classes = get_input_dimensions(args)
    data_sizes = { # keep track of layer data sizes. first entry is input data size
        'ch': [input_channels], 'im': [image_size],
    }

    # TODO: perhaps convert inner building function into its own method
    for layer in args.arch:

        # custom layer block
        custom_layer = ModelBuilder()

        if layer == "M": # MaxPool layer
            config=dict(kernel_size=2, stride=2)
            custom_layer.add_layer(nn.MaxPool2d, module_config=config)

            # include dimension tracking
            data_sizes['ch'].append(data_sizes['ch'][-1]) # same number of channels
            # calculate new spatial shape
            new_spatial = conv_output_size(custom_layer.layers[-1], 
                                           im_input=data_sizes['im'][-1])
            data_sizes['im'].append(new_spatial)
        
        else:
            # TODO: add identifiers/additional blocks into the builder here
            next = int(layer) # this means we get a size (for now)
            # print('layer sizes: ', input_channels, next)

            config = dict(in_channels=input_channels, out_channels=next, 
                          kernel_size=3, padding=1, bias=args.bias)
            
            new_layer = nn.Conv2d(**config) #Unpacks the dictionary into the function
            
            # inlcude dimension tracking
            data_sizes['ch'].append(new_layer.out_channels) # same number of channels
            # calculate new spatial shape
            new_spatial = conv_output_size(new_layer, 
                                           im_input=data_sizes['im'][-1])
            data_sizes['im'].append(new_spatial)

            if args.normalize_weights:
                custom_blocks.add_layer(\
                    nn.utils.parametrizations.weight_norm(new_layer),
                        configured=True)
                
            else:
                custom_blocks.add_layer(new_layer, configured=True)

            # TODO add args to change (kernel size)
            if args.batch_norm:
                custom_layer.add_layer(nn.BatchNorm2d, dict(num_features=next)) 
            
            # TODO: add args to change nonlinearity / custom nonlinearity block
            # TODO: add parsing args to have different nonlinearity
            custom_layer.add_layer(nn.ReLU, module_config=dict(inplace=True))

            input_channels = next
        
        custom_blocks.add_layer(custom_layer.model, configured=True)

    # make the avg pool
    if args.avgpool:
        avgpool = nn.AdaptiveAvgPool2d(args.avgpool_size)
        custom_blocks.add_layer(avgpool, configured=True)
        # include dimension tracking
        data_sizes['ch'].append(data_sizes['ch'][-1]) # same number of channels
        data_sizes['im'].append(args.avgpool_size) # pre-defined output shape
    else:
        avgpool=None

    # Use a dummy input tensor to calculate the output size
    # dummy_input = torch.randn(1, 1 if args.greyscale else 3, args.input_height, args.input_width)
    # dummy_output = custom_blocks.model(dummy_input)
    # print(dummy_output.shape())
    # output_size = dummy_output.view(1, -1).size(1)
    
    # calculate input size to classifier
    output_from_conv = int(data_sizes['ch'][-1] * data_sizes['im'][-1]**2)
    print('all channel sizes: ', data_sizes['ch'])
    print('all spatial sizes: ', data_sizes['im'])

    # print('output size of last layer: ', input_channels)
    # print('into classifier sizes: ', output_size)

    # make the classifier
    if args.classifier_dropout == 0:
        args.classifier_dropout = None
    config = dict(
        input_size = output_from_conv,
        layer_sizes = args.classifier_layers,
        n_classes = n_classes,
        # TODO: nonlinearity for classifier here too
        bias = args.classifier_bias,
        dropout = args.classifier_dropout
    )
    classifier = CustomClassifier(**config)

    # put everything together
    custom_model = CustomModel(features=custom_blocks.model, classifier=classifier,
                               avgpool=avgpool, )

    # print(custom_model.modules())
    return custom_model


def get_model_savename(args, architecture=True, dataload=True, optimizer=True):
    """
    :param args.name-convention: str must be in ["both", "default", "custom"] 
                    - determines which names to include in naming the model. 

                    "default" uses default/generated name only from the model
                    input parameters. "custom" uses custom/supplied name only 
                    from the model input paramters. "both" will use both names
                    # TODO: add adjusting which names get added to the default name
    """

    # get the architecture name
    conv_arch = "-".join(args.arch) + "-"
    batch_norm = "bn" if args.batch_norm else ""
    bias = f'{"nobias" if not args.bias else ""}'
    # avg_pool = f'A({".".join(str(k) for k in args.avgpool_size)})' if args.avgpool else " "
    avg_pool = f'A{args.avgpool_size}' if args.avgpool else ""
    classifier = f'+C{"-".join([str(k) for k in args.classifier_layers])}'
    class_bias = f'{"nocbias" if not args.classifier_bias else ""}'
    arch_name = "arch"+"".join([conv_arch, batch_norm, bias, avg_pool, classifier, class_bias])

    # get the dataset name
    dataset = args.dataset + '_'
    greyscale = 'g_' if args.greyscale else ""
    batchname = f'batch{args.batch_size}'
    data_name = "".join([dataset, greyscale, batchname])

    # get the training params name
    optim = f'{args.optimizer}_lr{args.lr}'#_wd{args.weight_decay}'
    cdrop = f'{args.classifier_dropout if args.classifier_dropout > 0 else ""}'
    opt_name = "_".join([optim, cdrop])

    # put the name together
    default = "_".join([data_name, arch_name, opt_name])
    custom = args.custom_name

    name = ""
    if args.name_convention == "default":
        name += default
    elif args.name_convention == "custom":
        name += custom
    else:
        name = custom+'_'+default

    return name


def parse_checkpoint_log_info(args):
    basic_train_info = f'{args.dataset}_batchsize{args.batch_size}'
    model_savename = get_model_savename(args)

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
    logfile = make_logfile(os.path.join(log_savedir, f'{basic_train_info}.log'))

    # save commandline entry to log
    with open(os.path.join(model_savedir, 'args.json'), 'w') as file:
        json.dump(args.__dict__, file, indent=2, default=str) 
    print_and_write(f"Command line: {' '.join(sys.argv)}", logfile)

    # creat summary file for summary including best run etc
    summaryfile = make_logfile(os.path.join(log_savedir, f'SUMMARY_{basic_train_info}.log'))

    return model_savefilename, checkpoint_type, logfile, summaryfile
