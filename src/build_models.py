import torch.nn as nn
from src.custom_blocks import ModelBuilder, CustomClassifier, CustomModel

# loading models from disc /
# building models from arg parameters

# TODO: custom blocks:
# equivariant nonlinearities
# equivariance basis blocks

def build_model_from_args(args, n_classes):
    # this makes custom VGG-type architectures
    custom_blocks = ModelBuilder()

    # make the layers 
    input_channels = get_input_channels(args)
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
            custom_layer.add_layer(nn.Conv2d, module_config=config) # TODO add args to change (kernel size)
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

    # make the classifier
    if args.classifier_dropout == 0:
        args.classifier_dropout = None
    config = dict(
        input_size = input_channels,
        layer_sizes = args.classifier_layers,
        n_classes = n_classes,
        # TODO: nonlinearity for classifier here too
        bias = args.classifier_bias,
        dropout = args.classifier_dropout
    )
    classifier = CustomClassifier(**config)

    # put everything together
    custom_model = CustomModel(features=custom_blocks.model, 
                               classifier=classifier,
                               avgpool=args.avgpool, 
                               avgpool_size=args.avgpool_size)

    # print(custom_model.modules())
    return custom_model



def get_input_channels(args):
    if "mnist" in args.dataset or args.greyscale:
        return 1
    else:
        return 3
    
# loading then adding in projection layers after a model is trained