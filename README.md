# Equivariant Machine Learning Projet 8 - Projecting to Equivariance Basis

## Quick Start for Running Experiments
For use in a notebook, import the scripts:
```
from src import *
```

To build a model and train it, use the `notebook.NotebookExperiment` object. 
By default, this will load in the parameters you would otherwise provide to
`main.py` in the notebook setting. 

To see the default values, look at [notebook](src/notebook.py#46). 
A training session might look something like this:

```
test = notebook.NotebookExperiment(avgpool=False, dataset='cifar', wandb_proj='example', epochs=2)
test.train()
```



## `globals.py`
In order to use this code, you will need to verify that the following variables 
inside of `globals.py` are set correctly:
1. `global_save_dir` where to save models throughout training
2. `global_data_dir` where to put data files

## Building a Model
Currently: support for `MaxPool` and `Conv2d` layers


## To-Do
- [ ] Default values for `NotebookExperiment` to a config file?