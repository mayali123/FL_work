#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os 
import copy
import numpy as np
import random

import torchvision
from torchvision import transforms
import torch
# from model.build_model import build_model
import logging
from utils.options import args_parser
from utils.train import get_local_update_objects, FedAvg
from utils.test import globaltest
# from utils.dataset import get_dataset
from utils.utils import set_output_files
import time


import sys
# 加载公共模块
sys.path.append("..")
from public_utils.dataset.dataset import get_dataset
from public_utils.utils.utils import add_noise,get_model_score,get_client_class_information,set_seed
from public_utils.model.build_model import build_model

if __name__ == '__main__':

    start = time.time()
    # parse args
    args = args_parser()
    args.n_clients = args.num_users

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # ------------------------------set output files path------------------------------
    model_save_path = set_output_files(args)


    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        set_seed(args.seed)

    dataset_train, dataset_test, dict_users = get_dataset(args)
    args.num_classes = args.n_classes
    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    # 得到类别信息
    get_client_class_information(dict_users, dataset_train, y_train, args)

    ##############################
    # Build model
    ##############################
    net_glob = build_model(args)
    net_glob = net_glob.to(args.device)


    ##############################
    # Training
    ##############################
    # logger = Logger(args)
    # 设置 forget_rate 
    forget_rate_schedule = []
            
    forget_rate = args.forget_rate
    exponent = 1
    forget_rate_schedule = np.ones(args.epochs) * forget_rate
    forget_rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** exponent, args.num_gradual)

    # Initialize f_G
    f_G = torch.randn(args.num_classes, args.feature_dim, device=args.device)
    
    # Initialize local update objects
    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        net_glob=net_glob,
    )
    print("in epoch")
    BACC = []
    best_performance = 0.
    for epoch in range(args.epochs):
        local_losses = []
        local_weights = []
        f_locals = []
        args.g_epoch = epoch
        
        # 调整学习率
        if (epoch + 1) in args.schedule:
            logging.info("Learning Rate Decay Epoch {}".format(epoch + 1))
            logging.info("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        logging.info(idxs_users)
        # Local Update
        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args         
            print(f"train in {idx }")
            w, loss, f_k = local.train(copy.deepcopy(net_glob).to(args.device), copy.deepcopy(f_G).to(args.device), client_num)
            print(f"{idx } train over ")
            f_locals.append(f_k)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
         
        # update global weights
        w_glob = FedAvg(local_weights) 
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)  

        # Update f_G
        sim = torch.nn.CosineSimilarity(dim=1) 
        tmp = 0
        w_sum = 0
        for i in f_locals:
            sim_weight = sim(f_G, i).reshape(args.num_classes, 1)
            # print(sim_weight.shape)
            w_sum += sim_weight
            tmp += sim_weight * i
        f_G = torch.div(tmp, w_sum)
        
        
        # test 
        net_glob.eval()
        pred = globaltest(copy.deepcopy(net_glob).to(args.device), dataset_test, args)
        acc, bacc = get_model_score(dataset_test.targets,pred)
        BACC.append(bacc)
        if bacc > best_performance:
            best_performance = bacc
        logging.info(f'eopch:{epoch} best bacc: {best_performance}, now bacc: {bacc}')

    BACC = np.array(BACC)
    logging.info("last:")
    logging.info(BACC[-10:].mean())
    logging.info("best:")
    logging.info(BACC.max())
    torch.cuda.empty_cache()