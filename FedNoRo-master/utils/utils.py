import copy
import random
import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score




def sigmoid_rampup(current, begin, end):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, begin, end)
    phase = 1.0 - (current-begin) / (end-begin)
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(rnd, begin, end):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(rnd, begin, end)


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


def get_output_and_label(loader, net, args):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()

            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)

            if i == 0:
                output_whole = np.array(outputs.cpu())
                label_whole = np.array(labels.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                label_whole = np.concatenate(
                    (label_whole, labels.cpu()), axis=0)

    return output_whole, label_whole


def cal_training_acc(prediction, noisy_labels, true_labels):
    prediction = np.array(prediction)
    noisy_labels = np.array(noisy_labels)
    true_labels = np.array(true_labels)

    acc_noisy = balanced_accuracy_score(noisy_labels, prediction)
    acc_true = balanced_accuracy_score(true_labels, prediction)

    return acc_noisy, acc_true



def set_output_files(args):
    outputs_dir = (f"../Save/FedNoRo-master/outputs_{args.dataset}_{args.n_type}_{args.level_n_system}_({args.level_n_lowerb}"
                   f"_{args.level_n_upperb})")
    if args.debug != "None":
        outputs_dir += f"/{args.debug}"
    if args.iid == 1:
        exp_dir = os.path.join(outputs_dir, f"frac1_iid")
    else:
        exp_dir = os.path.join(outputs_dir, f"frac1_non-iid_{args.non_iid_prob_class}_{args.alpha_dirichlet}")
    models_dir = os.path.join(exp_dir, 'models')
    mkdirs(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    mkdirs(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    mkdirs(tensorboard_dir)
    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(tensorboard_dir)

    return writer,models_dir

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass