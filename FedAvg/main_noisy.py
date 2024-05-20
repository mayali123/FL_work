import os
import torch
import numpy as np
from torch.utils.data import Subset
import logging

from model.Client import Client
from model.Fed import FedAvg
from model.test import test_img, globaltest
import copy
from utils.options import args_parser
from utils.utils import set_output_files
# from model.build_model import build_model
# from dataset.dataset import get_dataset
import torch.backends.cudnn as cudnn



import sys
# 加载公共模块
sys.path.append("..")
from public_utils.dataset.dataset import get_dataset
from public_utils.utils.utils import add_noise,get_model_score,get_client_class_information,set_seed
from public_utils.model.build_model import build_model

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

    # ------------------------------set output files path------------------------------
    model_save_path = set_output_files(args)

    # ------------------------------ dataset ------------------------------
    dataset_train, dataset_test, dict_users = get_dataset(args)

    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy
    
    # 得到类别信息
    get_client_class_information(dict_users, dataset_train, y_train, args)

    # 设置模型
    net_glob = build_model(args)
    w_glob = net_glob.state_dict()
    
    # 用于选择客户端
    m = max(int(args.frac * args.num_users), 1)
    prob = [1.0/args.num_users for _ in range(args.num_users)]
    # 最好的结果
    best_performance = 0.
    BACC = []
    for rnd in range(args.rounds):
        # 选好的客户端 下标
        idxs_users = np.random.choice(range(args.num_users), m, replace=False,p=prob)
        logging.info(f"idxs_users:{idxs_users}")
        loss_client = []
        w_client = []
        # 对每个客户端进行训练
        for idx in idxs_users:
            logging.info(f"client{idx}:")
            # 每个客户端的 训练样本
            sample_idx = np.array(list(dict_users[idx]))
            # 生成对应的客户端
            client = Client(args,dataset_train,sample_idx)
            # 开始训练
            w, loss = client.train(copy.deepcopy(net_glob).to(args.device))

            w_client.append(copy.deepcopy(w))
            loss_client.append(loss)

        # 平均loss
        avg_loss = sum(loss_client)/len(loss_client)
        print("全局epoch={} 全局loss = {}".format(rnd, avg_loss))

        # 聚合
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob = FedAvg(w_client, dict_len)
        # 加载到 服务器模型中
        net_glob.load_state_dict(w_glob)

        
        # testing
        net_glob.eval()
        pred = globaltest(copy.deepcopy(net_glob).to(args.device), dataset_test, args)
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
