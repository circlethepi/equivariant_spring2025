import src.custom_blocks as blocks
import src.build_models as build
import src.caluclate_basis as ebasis
import src.datasets as datasets
import src.train as train
from globals import *
import argparse
import wandb
import src.emlp_torch as emlp_torch


from src.utils import *

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
                                     seed=0,
                                     n_classes = 100 if dataset == "cifar100" else 10)
    

    # feed to builder
    model = build.build_model_from_args(custom_args)
    
    return model


# currently manually takes in the same arguments as specified in main.py
# except the default arch. 
default_arch = ['64', 'M', '64', 'M']
default_args = dict(arch=default_arch,
                batch_norm = False,
                bias=False,
                normalize_weights=False,
                avgpool=False,
                avgpool_size=1,
                classifier_layers=[4096, 4096],
                classifier_bias=False,
                classifier_dropout=0,

                dataset='mnist',
                greyscale=False,
                data_rt_inc=None,
                data_rt_fill=True,
                data_upsample=56,

                seed=0,

                epochs=100,
                batch_size=256,
                optimizer='Adam',
                lr=1e-5,
                
                criterion='CrossEntropyLoss',
                save_model=True,
                save_epoch=False,
                save_path=global_save_dir,
                log_path=global_log_dir,
                custom_name=None,
                name_convention='default',
                wandb_proj=None)


class NotebookExperiment:    

    def __init__(self, **kwargs):
        # set default values as defined immediately above
        arg_dict = default_args.copy()
        arg_dict.update(kwargs) # update any new values specified
        print(arg_dict)
        self.args = argparse.Namespace(**arg_dict)
        self.name = build.get_model_savename(self.args)

        if self.args.seed is not None:
            set_seed(self.args.seed)
        if self.args.wandb_proj is not None:
            wandb.init(project=self.args.wandb_proj,
                       config=self.args,
                       name=self.name)
            self.wandb_log=True
        
        self.savefilename, self.checkpoint, self.logfile, self.summaryfile = \
            build.parse_checkpoint_log_info(self.args)

        self.model = self.build_model()

        return


    def build_model(self):

        model = build.build_model_from_args(self.args)

        return model
    
    def get_dataloaders(self):
        return datasets.get_dataloaders(self.args, self.logfile)
    

    def train(self):

        loss_accs = train.train(self.args, self.model, *self.get_dataloaders(), 
                            self.savefilename, self.checkpoint, self.logfile, self.summaryfile)

        return loss_accs
    
    def test(self, dataloader, loader_name, step=None):

        test_vals = train.evaluate_model(self.model,
                                         dataloader,
                                         loader_name=loader_name,
                                         device=get_device(),
                                         topk=(1,5),
                                         step=step,
                                         wandb_log=self.wandb_log)

        return test_vals
    

class NotebookEMLP(NotebookExperiment):

    def __init__(self, rep_in, rep_out, group, **kwargs):
        super.__init__()

        if kwargs["ch"] is not None:
            n_ch = kwargs["ch"]
        else:
            n_ch = 640

        if kwargs["num_layers"] is not None:
            num_layers = kwargs["num_layers"]
        else:
            num_layers = 4
        
        if kwargs['name'] is not None:
            self.savename = kwargs['name'] 
        else:
            self.savename = f"emlp_{n_ch}_{num_layers}_{self.args.data_rt_inc}"

        model_savedir = os.path.join(self.args.save_path, self.savename)
        checkpoint_filename = f'{self.savename}.pth.tar'

        checkpoint_type = "epoch" if self.args.save_epoch else "batch"
        
        # self.logfile, self.summaryfile = 
