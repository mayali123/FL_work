import os
import sys
import os.path
import hashlib
import errno
import random
import torch
import numpy as np
import logging
from numpy.testing import assert_array_almost_equal
import copy





def set_output_files(args):
    file_name = __file__.split('/')[-3]
    # print(f"file_name:{file_name}")
    outputs_dir = (f"../Save/{file_name}/outputs_{args.dataset}_{args.n_type}_{args.level_n_system}_{args.level_n_lowerb}"
                f"_{args.level_n_upperb}")
    if args.iid == 1:
        exp_dir = os.path.join(outputs_dir, f"frac{args.frac}_iid")
    else:
        exp_dir = os.path.join(outputs_dir, f"frac{args.frac}_non-iid_{args.non_iid_prob_class}_{args.alpha_dirichlet}")
   
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


