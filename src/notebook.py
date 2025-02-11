import src.custom_blocks as blocks
import src.build_models as build
import src.caluclate_basis as ebasis
import src.datasets as datasets
import src.train
import argparse

def build_model(vgg_layers=['64', 'M', '64', 'M'], 
                batch_norm=False,
                bias=False,
                classifier_layers=[512, 512],
                classifier_bias=False, 
                classifier_dropout=0,
                avgpool=True,
                avgpool_size=[1,1],
                dataset="mnist",
                greyscale=False, 
                ):
    """
    custom model for notebook
    """
    # make args
    custom_args = argparse.Namespace(arch=vgg_layers, 
                                     batch_norm=batch_norm,
                                     bias=bias,
                                     classifier_layers=classifier_layers,
                                     classifier_bias=classifier_bias,
                                     classifier_dropout=classifier_dropout,
                                     avgpool=avgpool,
                                     avgpool_size=avgpool_size,
                                     dataset=dataset,
                                     greyscale=greyscale,
                                     n_classes = 100 if dataset == "cifar100" else 10)
    

    # feed to builder
    model = build.build_model_from_args(custom_args)
    
    return model