import os
import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from collections import Counter

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from utils.options import args_parser
from utils.local_training import LocalUpdate, globaltest
from utils.FedAvg import FedAvg, DaAgg
from utils.utils import  set_output_files, get_output, get_current_consistency_weight



import sys
# 加载公共模块
sys.path.append("..")
from public_utils.dataset.dataset import get_dataset
from public_utils.utils.utils import add_noise,set_seed,get_model_score
from public_utils.model.build_model import build_model

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL
"""


if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed)

    # ------------------------------ output files ------------------------------
    writer, models_dir = set_output_files(args)

    # ------------------------------ dataset ------------------------------
    dataset_train, dataset_test, dict_users = get_dataset(args)
    logging.info(
        f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    logging.info(
        f"test: {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")

    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(
        args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    # --------------------- Build Models ---------------------------
    netglob = build_model(args)
    user_id = list(range(args.n_clients))
    trainer_locals = []
    for id in user_id:
        trainer_locals.append(LocalUpdate(
            args, id, copy.deepcopy(dataset_train), dict_users[id]))

    # ------------------------------ begin training ------------------------------
    set_seed(args.seed)
    logging.info("\n ---------------------begin training---------------------")
    best_performance = 0.

    # ------------------------ Stage 1: warm up ------------------------ 
    BACC = []
    best_performance = 0.
    for rnd in range(args.rounds):
        w_locals, loss_locals = [], []
        for idx in user_id:  # training over the subset
            logging.info(f"train client idx:{idx}")
            local = trainer_locals[idx]
            w_local, loss_local = local.train_LA(
                net=copy.deepcopy(netglob).to(args.device), writer=writer)

            # store every updated model
            w_locals.append(copy.deepcopy(w_local))
            loss_locals.append(copy.deepcopy(loss_local))

        w_locals_last = copy.deepcopy(w_locals)
        dict_len = [len(dict_users[idx]) for idx in user_id]
        w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        pred = globaltest(copy.deepcopy(netglob).to(
            args.device), dataset_test, args)
        acc, bacc = get_model_score(dataset_test.targets,pred)
        BACC.append(bacc)
        if bacc > best_performance:
            best_performance = bacc
        logging.info(f'eopch:{rnd} best bacc: {best_performance}, now bacc: {bacc}')
        
    BACC = np.array(BACC)
    logging.info("last:")
    logging.info(BACC[-10:].mean())
    logging.info("best:")
    logging.info(BACC.max())

    torch.cuda.empty_cache()