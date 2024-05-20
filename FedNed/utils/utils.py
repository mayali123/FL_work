import copy
import random
import numpy as np
import pandas as pd
import logging
from numpy.testing import assert_array_almost_equal
import torch
import torch.nn.functional as F
import os
import sys
# from dataset.utils_1.dataset import Indices2Dataset
# basic function
def set_output_files(args):
    outputs_dir = (f"../Save/FedNed/outputs_{args.dataset}_{args.n_type}_{args.level_n_system}_{args.level_n_lowerb}"
                   f"_{args.level_n_upperb}")
    if args.iid == 1:
        exp_dir = os.path.join(outputs_dir, f"frac{args.join_ratio}_iid")
    else:
        exp_dir = os.path.join(outputs_dir, f"frac{args.join_ratio}_non-iid_{args.non_iid_prob_class}_{args.alpha_dirichlet}")

    models_dir = os.path.join(exp_dir, 'models')
    mkdirs(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    mkdirs(logs_dir)

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    return models_dir


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass