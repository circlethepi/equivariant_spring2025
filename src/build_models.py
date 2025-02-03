import torch.nn as nn

# loading models from disc /
# building models from arg parameters
class ModelBuilder:
    """Builds a Model"""
    def __init__(self):
        self.layers=[]
    
    def add_layer(self, module_class, module_config=None):
        module_config = module_config if module_config is not None else {}
        layer = module_class(**module_config)
        self.layers.append(layer)
        return layer
    
    @property
    def model(self):
        if len(self.layers) == 0:
            return nn.Identity()
        elif len(self.layers) == 1:
            return self.layers[0]
        else:
            return nn.Sequential(*self.layers)

# TODO: custom blocks
# equivariant nonlinearities
# equivariance basis blocks

def build_model_from_args(args):
    # this makes custom VGG-type architectures
    builder = ModelBuilder()

    # make the layers 
    input_channels = get_input_channels(args)
    for layer in args.arch:
        
        if layer == "M": # MaxPool layer
            config=dict(kernel_size=2, stride=2)
            builder.add_layer(nn.MaxPool2d, module_config=config)
        else:
            # TODO: add identifiers/additional blocks into the builder here
            next = int(layer) # this means we get a size (for now)
            config = dict(in_channels=input_channels, out_channels=next, 
                          kernel_size=3, padding=1)
            builder.add_layer(nn.Conv2d, module_config=config) # TODO add args to change 
            if args.batch_norm:
                builder.add_layer(nn.BatchNorm2d, dict(num_features=next)) 
            
            # TODO: add args to change nonlinearity 
            # TODO: add parsing args to have different nonlinearity
            builder.add_layer(nn.ReLU, dict(inplace=True))

            input_channels = next
    print(builder.layers)
    return builder.model



def get_input_channels(args):
    if "mnist" in args.dataset or args.greyscale:
        return 1
    else:
        return 3
    
# loading then adding in projection layers after a model is trained