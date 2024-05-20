import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist
import logging
import random
import os
import sys
from numpy.testing import assert_array_almost_equal


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def set_output_files(args):
    outputs_dir = (f"../Save/FedCorr/outputs_{args.dataset}_{args.n_type}_{args.level_n_system}_{args.level_n_lowerb}"
                f"_{args.level_n_upperb}")
    if args.iid == 1:
        exp_dir = outputs_dir + f"iid"
    else:
        exp_dir = outputs_dir + f"non-iid_{args.non_iid_prob_class}_{args.alpha_dirichlet}"
    exp_dir += f"/iter_{args.iteration1}_rnd1_{args.rounds1}_rnd2_{args.rounds2}_frac1_{args.frac1}_frac2_{args.frac2}"
    if args.mixup:
        exp_dir += "mixup"
    models_dir = os.path.join(exp_dir, 'models')
    logs_dir = os.path.join(exp_dir, 'logs')
    mkdirs(models_dir)
    mkdirs(logs_dir)

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    return models_dir

def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if latent == False:
                outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images, True)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids
