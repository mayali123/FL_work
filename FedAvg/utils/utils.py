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
# basic function



def get_output(loader, net, args, softmax=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if softmax == True:
                outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                if criterion is not None:
                    loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                if criterion is not None:
                    loss_whole = np.concatenate(
                        (loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def set_output_files(args):
    outputs_dir = (f"../Save/FedAvg/outputs_{args.dataset}_{args.n_type}_{args.level_n_system}_{args.level_n_lowerb}"
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