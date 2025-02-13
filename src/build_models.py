import torch
import torch.nn as nn
from src.custom_blocks import ModelBuilder, CustomClassifier, CustomModel
import os

# loading models from disc /
# building models from arg parameters

# TODO: custom blocks:
# equivariant nonlinearities
# equivariance basis blocks

#Changed model signature to just have args
def build_model_from_args(args):
    # this makes custom VGG-type architectures
    custom_blocks = ModelBuilder()

    # make the layers 
    input_channels = 1 if args.greyscale else 3
    for layer in args.arch:

        # custom layer block
        custom_layer = ModelBuilder()

        if layer == "M": # MaxPool layer
            config=dict(kernel_size=2, stride=2)
            custom_layer.add_layer(nn.MaxPool2d, module_config=config)
        else:
            # TODO: add identifiers/additional blocks into the builder here
            next = int(layer) # this means we get a size (for now)
            config = dict(in_channels=input_channels, out_channels=next, 
                          kernel_size=3, padding=1, bias=args.bias)
            new_layer = nn.Conv2d(**config) #Unpacks the dictionary into the function
            if args.weight_normalization:
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

    # Use a dummy input tensor to calculate the output size
    dummy_input = torch.randn(1, 1 if args.greyscale else 3, args.input_height, args.input_width)
    dummy_output = custom_blocks.model(dummy_input)
    output_size = dummy_output.view(1, -1).size(1)

    # make the classifier
    if args.classifier_dropout == 0:
        args.classifier_dropout = None
    config = dict(
        input_size = output_size,
        layer_sizes = args.classifier_layers,
        n_classes = args.n_classes,
        # TODO: nonlinearity for classifier here too
        bias = args.classifier_bias,
        dropout = args.classifier_dropout
    )
    classifier = CustomClassifier(**config)

    # put everything together
    custom_model = CustomModel(features=custom_blocks.model, classifier=classifier,
                               avgpool=args.avgpool, 
                               avgpool_size=args.avgpool_size)

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
    conv_arch = "-".join(args.arch)
    batch_norm = "bn" if args.batch_norm else ""
    bias = f'{"nobias" if not args.bias else ""}'
    avg_pool = f'A({".".join(str(k) for k in args.avgpool_size)})' if args.avgpool else " "
    classifier = f'+C{"-".join([str(k) for k in args.classifier_layers])}'
    class_bias = f'{"nocbias" if not args.classifier_bias else ""}'
    arch_name = "arch"+"".join([conv_arch, batch_norm, bias, avg_pool, classifier, class_bias])

    # get the dataset name
    dataset = args.dataset + '_'
    greyscale = 'g_' if args.greyscale else ""
    batchname = f'batch{args.batch_size}'
    data_name = "".join([dataset, greyscale, batchname])

    # get the training params name
    optim = f'{args.optimizer}_lr{args.lr}_wd{args.weight_decay}'
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


#Removed this function since we may want to use something else
#def get_input_channels(args):
 #   if "mnist" in args.dataset or args.greyscale:
   #     return 1
    #else:
     #   return 3
    
# loading then adding in projection layers after a model is trained