import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from datetime import datetime
from server import Server
# from dataset.get_cifar10 import get_cifar10
from dataset.utils_1.dataset import Indices2Dataset
# from models.model_feature import ResNet_cifar_feature

from utils.utils import set_output_files
from options import args_parser
# from models.build_model import build_model
import numpy as np
import copy
import torch
import random
import logging
import sys
# 加载公共模块
sys.path.append("..")
from public_utils.dataset.dataset import get_dataset,get_public_dataset
from public_utils.utils.utils import add_noise,get_client_class_information,set_seed
from public_utils.model.build_model import build_model

def get_train_label(data_local_training, index_list):
    trian_label_list = []
    for index in index_list:
        label = data_local_training[index][1]
        trian_label_list.append(label)
    return trian_label_list


def label_rate(test_label_list, train_label_list):
    true_num = 0
    for true_label, nos_label in zip(test_label_list, train_label_list):
        if true_label == nos_label:
            true_num += 1
    rate = true_num / len(test_label_list)
    print(f"噪声比例：{1-rate}")

def main(args):
    prev_time = datetime.now()
    
    args.n_clients = args.num_clients
    args.num_users = args.num_clients
    # ----------------------------- set GPU --------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        set_seed(args.seed)

    # ------------------------------set output files path------------------------------
    model_save_path = set_output_files(args)
    
    # ------------------------------ dataset ------------------------------
    dataset_train, dataset_test, dict_users = get_dataset(args)
    args.num_classes = args.n_classes
    global_distill_dataset = get_public_dataset()


    model = build_model(args)
    # model = ResNet_cifar_feature(resnet_size=8, scaling=4,
    #                              save_activations=False, group_norm_num_groups=None,
    #                              freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
    
    # --------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    train_data_list = []

    for idx in range(args.num_clients):
        client_sample_idx = np.array(list(dict_users[idx]))
        client_label = y_train_noisy[client_sample_idx]

        # 每一个客户端都有自己的dataset 对象
        client_train_data = Indices2Dataset(dataset_train)
        client_train_data.load(dict_users[idx], client_label)
        train_data_list.append(client_train_data)
         

    # 得到每个客户端的类别信息
    get_client_class_information(dict_users, dataset_train, y_train, args)

    server = Server(args=args,
                    train_data_list=train_data_list,
                    global_test_dataset=dataset_test,
                    global_distill_dataset=global_distill_dataset,
                    global_student=model,
                    temperature=args.temperature,
                    mini_batch_size_distillation=args.mini_batch_size_distillation,
                    lamda=args.lamda
                    )

    server.train()

    BACC = server.BACC
    BACC = np.array(BACC)
    logging.info("last:")
    logging.info(BACC[-10:].mean())
    logging.info("best:")
    logging.info(BACC.max())
    torch.cuda.empty_cache()

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "train time %02d:%02d:%02d" % (h, m, s)
    print(time_str)


if __name__ == '__main__':
    args = args_parser()
    main(args)

