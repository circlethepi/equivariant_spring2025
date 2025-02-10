import numpy as np
import torch.nn as nn
from typing import Optional, Union


class ModelBuilder:
    """Builds a Model"""
    def __init__(self):
        self.layers=[]
    
    def add_layer(self, module_class, configured:bool=False, 
                  module_config:Optional[dict]=None):
        """adds a configured layer or configures and adds a layer

        :param module_class: nn.Module - the configured module or the class of
                             module to add
        :param configured: bool
        :param module_config: dict - config dict for the module
        """
        if not configured:
            module_config = module_config if module_config is not None else {}
            layer = module_class(**module_config)
        else:
            layer = module_class
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
    

class CustomModel(nn.Module):
    """segments a custom model"""

    def __init__(self, features:nn.Module, classifier:nn.Module, 
                 avgpool:bool=True, avgpool_size=(1,1)):
        """container for custom models"""
        super().__init__()

        self.features = features

        self.do_avgpool = avgpool
        self.avgpool_size = avgpool_size
        if self.do_avgpool:
            self.avgpool = nn.AdaptiveAvgPool2d(self.avgpool_size)

        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        if self.do_avgpool:
            x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.classifier(x)
        return x

        

# define custom blocks that we can use
class CustomClassifier(nn.Module):
    """ custom configurable MLP classifier """
    def __init__(self, input_size:int, layer_sizes:list, n_classes:int, 
                 nonlinearity=nn.ReLU(True), bias:bool=True, 
                 dropout=None):
        super().__init__()

        # config
        self.input_size = input_size
        self.layer_sizes = layer_sizes if layer_sizes is not None else []
        self.n_classes = n_classes
        self.bias = bias
        # self.nonlinearity = nonlinearity
        self.dropout = dropout
        
        if self.dropout is not None:
            assert 0 <= self.dropout <= 1
        
        # making the layers
        self.classifier_layers = None
        self.build_layers(nonlinearity)


    def build_layers(self, nonlinearity):
        # TODO: different nonlinearities for different layers?
        classifier = ModelBuilder()

        in_dimension = self.input_size
        for out_dimension in self.layer_sizes + [self.n_classes]:
            class_layer = ModelBuilder()

            config = dict(
                in_features=in_dimension,
                out_features=out_dimension,
                bias=self.bias,
            )
            class_layer.add_layer(nn.Linear, module_config=config)
            class_layer.add_layer(nonlinearity, configured=True)

            if self.dropout is not None:
                config = dict(p=self.dropout)
                class_layer.add_layer(nn.Dropout, module_config=config)
            
            classifier.add_layer(class_layer.model, configured=True)
            in_dimension = out_dimension

        self.classifier_layers = classifier.model
    
    def forward(self, x):
        return self.classifier_layers(x)


class EquivariantProjectionLayer(nn.Module):
    """
    Custom projection layer that projects previous layer weights into 
    equivariant basis for specified group
    """

    def __init__(self, group_name, previous_layer: nn.Module, rank=float("inf")):
        """
        
        """
        super().__init__()
    
    def calculate_basis(self):
        # calculate basis for transform from previous layer
        
        return
        


    def forward(self, x):

        return x