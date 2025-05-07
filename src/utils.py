import torch
import numpy as np
import random
import os

"""
tools
"""

def set_seed(seed):
    """set the random seed everywhere relevant"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    return

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if \
                          torch.backends.mps.is_available() else "cpu")
    return device

# logging/writing things
def make_logfile(filepath):
    """opens a file (that will need to be closed)"""
    logfile = open(filepath, 'a')
    return logfile


def print_and_write(message, *files):
    """files should be open"""
    print(message)
    for file in files:
        if file is not None:
            file.write(message+'\n')
    return

def close_files(*files):
    """closes logs"""
    for file in files:
        if file is not None:
            file.close()
    return

def nice_interval(n: int):
    """checks if a number is nice
    nice numbers are of the form {1, 2, 5} * 10^k"""
    if n == 0:
        return True
    else:
        pwr = 10 ** int(np.log10(n))
        nice = True if (n == 0 or (n % pwr == 0 and n // pwr in (1, 2, 5))) else False
        return nice

def get_nice_intervals(min_val=0, max_val=None, n_intervals=None):
    """returns a list of nice intervals between min_val and max_val based on
    either the max interval value or the number of intervals. 
    """

    # make sure the inputs are valid
    assert not (max_val is None and n_intervals is None), \
        "Either `max_val` or `num_intervals` must be specified"
    
    # define the range for powers of 10
    max_pwr = int(np.log10(max_val)) if max_val is not None \
                                            else int(np.ceil(n_intervals/3))
    min_pwr = int(np.log10(min_val)) if min_val is not None and min_val > 0 \
                                                        else 0

    # get the intervals
    intervals = []
    for i in range(min_pwr, max_pwr+1):
        pwr = 10 ** i
        # get the nice intervals for this power
        for k in (1, 2, 5):
            nice = k * pwr
            # check if max value is achieved
            if max_val is not None and nice > max_val:
                break
            intervals.append(k * pwr)
            # check if nice value is achieved
            if n_intervals is not None and len(intervals) >= n_intervals:
                break
            
        if n_intervals is not None and len(intervals) >= n_intervals:
                break

    return intervals


class AverageMeter(object):
    def __init__(self, name=None, format=':.2f'):
        self.reset()
        self.name = name
        self.format = format
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.format + '} ({avg' + self.format + '})'
        return fmtstr.format(**self.__dict__)


##################################################
## PLOT FORMATTING 
##################################################
def format_number(n):
    if n <= 0:
        raise ValueError("Only positive numbers are supported.")
    
    log10 = np.log10(n)

    if int(log10) == log10:
        return f"$10^{{{int(log10)}}}$"
    
    elif n < 10000:
        return str(int(n)) if (int(n) == n) else str(n)
    
    else:
        exponent = int(np.floor(log10))
        mantissa = n / (10 ** exponent)
        return f"${mantissa:.1f} \\times 10^{{{exponent}}}$"
