#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import numpy as np
import random
import time
import logging

import torchvision
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# from utils.dataset import get_dataset
from utils.options import args_parser
from utils.utils import set_output_files
from models.fed import LocalModelWeights
# from models.nets import get_model
# from models.build_model import build_model
from models.test import globaltest
from models.update import get_local_update_objects


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

    # ------------------------------ dataset ------------------------------
    dataset_train, dataset_test, dict_users = get_dataset(args)
    args.num_classes = args.n_classes
    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy
    # 得到类别信息
    get_client_class_information(dict_users, dataset_train, y_train, args)
    


    # Arbitrary gaussian noise
    gaussian_noise = torch.randn(1, 3, 128, 128)
    # for logging purposes
    logging_args = dict(
        batch_size=args.local_bs,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    ##############################
    # Build model
    ##############################
    # net_glob = get_model(args)
    net_glob = build_model(args)
    net_glob = net_glob.to(args.device)


    ##############################
    # Training
    ##############################
    CosineSimilarity = torch.nn.CosineSimilarity()

    forget_rate_schedule = []
    if args.method in ['coteaching', 'coteaching+']:
        num_gradual = args.warmup_epochs
        forget_rate = args.forget_rate
        exponent = 1
        forget_rate_schedule = np.ones(args.epochs) * forget_rate
        forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

    pred_user_noise_rates = [args.forget_rate] * args.num_users

    # Initialize local model weights
    fed_args = dict(
        all_clients=args.all_clients,
        num_users=args.num_users,
        method=args.fed_method,
        dict_users=dict_users,
    )
    # 用于模型聚合时
    local_weights = LocalModelWeights(net_glob=net_glob, **fed_args)
    # if args.send_2_models:
    #     local_weights2 = LocalModelWeights(net_glob=net_glob2, **fed_args)

    # Initialize local update objects
    local_update_objects = get_local_update_objects(
        args=args,
        dataset_train=dataset_train,
        dict_users=dict_users,
        noise_rates=pred_user_noise_rates,
        gaussian_noise=gaussian_noise,
    )
    for i in range(args.num_users):
        local = local_update_objects[i]
        local.weight = copy.deepcopy(net_glob.state_dict())
        
    if args.start > 0:
        model_path = f'{model_save_path}/{args.start}.pth'
        net_glob.load_state_dict(torch.load(model_path))
        epoch_start = args.start - 1
    else:
        epoch_start = 0
    
    BACC = []
    best_performance = 0.
    for epoch in range(epoch_start,args.epochs):
        # 调整学习率
        if (epoch + 1) in args.schedule:
            logging.info("Learning Rate Decay Epoch {}".format(epoch + 1))
            logging.info("{} => {}".format(args.lr, args.lr * args.lr_decay))
            args.lr *= args.lr_decay

        if len(forget_rate_schedule) > 0:
            args.forget_rate = forget_rate_schedule[epoch]

        local_losses = []
        local_losses2 = []
        args.g_epoch = epoch

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # Local Update
        logging.info(f"idxs_users:{idxs_users}")
        for client_num, idx in enumerate(idxs_users):
            local = local_update_objects[idx]
            local.args = args
            logging.info(f"idx:{idx}")
            if epoch < args.warmup_epochs:
                w, loss = local.train_phase1(copy.deepcopy(net_glob).to(args.device))
            else:
                # Get similarity, expertise values
                sim_list = []
                exp_list = []
                # 计算与其他客户端的相似性
                for user in range(args.num_users):
                    sim = CosineSimilarity(
                        local.arbitrary_output.to(args.device),
                        local_update_objects[user].arbitrary_output.to(args.device),
                    ).item()
                    exp = local_update_objects[user].expertise
                    sim_list.append(sim)
                    exp_list.append(exp)

                # Normalize similarity & expertise values
                sim_list = [(sim - min(sim_list)) / (max(sim_list) - min(sim_list)) for sim in sim_list]
                exp_list = [(exp - min(exp_list)) / (max(exp_list) - min(exp_list)) for exp in exp_list]

                # Compute & sort scores
                prev_score = args.w_alpha * exp_list[idx] + (1 - args.w_alpha)

                score_list = []
                for neighbor_idx, (exp, sim) in enumerate(zip(exp_list, sim_list)):
                    if neighbor_idx != idx:
                        score = args.w_alpha * exp + (1 - args.w_alpha) * sim
                        score_list.append([score, neighbor_idx])
                score_list.sort(key=lambda x: x[0], reverse=True)
                # print(f"score_list:{score_list}")
                # Get top-k neighbors
                neighbor_list = []
                neighbor_score_list = []
                for k in range(args.num_neighbors):
                    neighbor_score, neighbor_idx = score_list[k]
                    neighbor_net = copy.deepcopy(local_update_objects[neighbor_idx].net1)
                    neighbor_list.append(neighbor_net)
                    neighbor_score_list.append(neighbor_score)

                w, loss= local.train_phase2(copy.deepcopy(net_glob).to(args.device),
                                                prev_score,
                                                neighbor_list,
                                                neighbor_score_list)


            local_weights.update(idx, w)
            local_losses.append(copy.deepcopy(loss))
        

        w_glob = local_weights.average()  # update global weights
        net_glob.load_state_dict(w_glob, strict=False)  # copy weight to net_glob
        # if epoch % 5 == 0:
        #     torch.save(net_glob.state_dict(), f'{model_save_path}/{epoch}.pth')
        local_weights.init()

        # testing
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

