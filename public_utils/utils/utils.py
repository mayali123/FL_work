import copy
import numpy as np
import pandas as pd
import logging
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from numpy.testing import assert_array_almost_equal
import os


# def add_noise(args, y_train, dict_users):
#     np.random.seed(args.seed)
#     gamma_s = np.array([0.] * args.num_users)
#     gamma_s[:int(args.level_n_system*args.num_users)] = 1.
#     np.random.shuffle(gamma_s)
#     gamma_c_initial = np.random.rand(args.num_users)
#     gamma_c_initial = (args.level_n_upperb - args.level_n_lowerb) * \
#         gamma_c_initial + args.level_n_lowerb
#     gamma_c = gamma_s * gamma_c_initial
#     y_train_noisy = copy.deepcopy(y_train)

#     if args.n_type == "instance":
#         if args.dataset == "isic2019":
#             df = pd.read_csv("your csv")
#         elif args.dataset == "ICH":
#             df = pd.read_csv("/home/mayali/My_try_code/data/ICH/ICH_softlabel.csv")
#         else:
#             raise

#         soft_label = df.iloc[:, 1:args.n_classes+1].values.astype("float")
#         real_noise_level = np.zeros(args.num_users)
#         for i in np.where(gamma_c > 0)[0]:
#             sample_idx = np.array(list(dict_users[i]))
#             soft_label_this_client = soft_label[sample_idx]
#             hard_label_this_client = y_train[sample_idx]

#             p_t = copy.deepcopy(soft_label_this_client[np.arange(
#                 soft_label_this_client.shape[0]), hard_label_this_client])
#             p_f = 1 - p_t
#             p_f = p_f / p_f.sum()
#             # Choose noisy samples base on the misclassification probability.
#             noisy_idx = np.random.choice(np.arange(len(sample_idx)), size=int(
#                 gamma_c[i]*len(sample_idx)), replace=False, p=p_f)

#             for j in noisy_idx:
#                 soft_label_this_client[j][hard_label_this_client[j]] = 0.
#                 soft_label_this_client[j] = soft_label_this_client[j] / \
#                     soft_label_this_client[j].sum()
#                 # Choose a noisy label base on the classification probability.
#                 # The noisy label is different from the initial label.
#                 y_train_noisy[sample_idx[j]] = np.random.choice(
#                     np.arange(args.n_classes), p=soft_label_this_client[j])

#             noise_ratio = np.mean(
#                 y_train[sample_idx] != y_train_noisy[sample_idx])
#             logging.info("Client %d, noise level: %.4f, real noise ratio: %.4f" % (
#                 i, gamma_c[i], noise_ratio))
#             real_noise_level[i] = noise_ratio

#     elif args.n_type == "random":
#         real_noise_level = np.zeros(args.num_users)
#         for i in np.where(gamma_c > 0)[0]:
#             sample_idx = np.array(list(dict_users[i]))
#             prob = np.random.rand(len(sample_idx))
#             noisy_idx = np.where(prob <= gamma_c[i])[0]
#             y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(
#                 0, args.n_classes, len(noisy_idx))
#             noise_ratio = np.mean(
#                 y_train[sample_idx] != y_train_noisy[sample_idx])
#             logging.info("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
#                 i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
#             real_noise_level[i] = noise_ratio

#     else:
#         raise NotImplementedError

#     return (y_train_noisy, gamma_s, real_noise_level)


def get_model_score(targets,pred,is_need_print=True):
    acc = accuracy_score(targets, pred)
    bacc = balanced_accuracy_score(targets, pred)
    cm = confusion_matrix(targets, pred)
    if is_need_print:
        logging.info(
            "******** acc: %.4f, bacc: %.4f ********" % (acc, bacc))
        logging.info(cm)

    return acc, bacc


def get_client_class_information(dict_users, dataset_train, y_train, args):
    client_num = args.num_users
    class_num = args.n_classes
    for idx in range(client_num):
        logging.info(f'客户端{idx}')
        sample_idx = list(dict_users[idx])

        class_sample_idx = [[] for _ in range(class_num)]
        for j in sample_idx:
            cls = dataset_train.targets[j]
            class_sample_idx[cls].append(j)

        client_class_num = np.zeros(class_num,dtype=int)
        client_class_noisy_num = np.zeros(class_num,dtype=int)
        for cls in range(class_num):
            class_idx = np.array(class_sample_idx[cls])
            if len(class_idx) != 0:
                client_class_num[cls] = len(class_idx)
                client_class_noisy_num[cls] =len(np.where(np.array(dataset_train.targets)[class_idx] != y_train[class_idx])[0])

        logging.info(f'各个类别数量：{client_class_num}')
        logging.info(f'各个类别噪声数量：{client_class_noisy_num}')
        logging.info(f'各个类别噪声比：{client_class_noisy_num/client_class_num}')
        logging.info(f'客户端噪声比：{np.sum(client_class_noisy_num) / np.sum(client_class_num)}')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print('m=', m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print(P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        
        y_train = y_train_noisy
    return y_train, actual_noise


def add_noise(args, y_train, dict_users):
    # 随机产生有噪声的客户端
    gamma_s = np.array([0] * args.num_users)
    gamma_s[:int(args.level_n_system * args.num_users)] = 1
    np.random.shuffle(gamma_s)
    # 随机设置有噪声客户端的噪声比率
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (args.level_n_upperb - args.level_n_lowerb) * \
                      gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial
    # 加噪之后的label
    y_train_noisy = copy.deepcopy(y_train)
    # 真实产生的噪声
    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        client_label = y_train[sample_idx]
        client_label = np.asarray([[client_label[i]] for i in range(len(client_label))])
        if args.n_type == 'pairflip':
            client_label_noisy_labels, actual_noise_rate = noisify_pairflip(client_label, gamma_c[i],
                                                                            random_state=args.seed,
                                                                            nb_classes=args.num_classes)
        elif args.n_type == 'symmetric':
            client_label_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(client_label, gamma_c[i],
                                                                                        random_state=args.seed,
                                                                                        nb_classes=args.num_classes)
        elif args.n_type == 'None':
            client_label_noisy_labels = client_label
            actual_noise_rate = 0
        else:
            raise NotImplementedError
        
        logging.info("Client %d, noise level: %.4f , real noise ratio: %.4f" % (
            i, gamma_c[i], actual_noise_rate))
        # 还原
        client_label_noisy_labels = [i[0] for i in client_label_noisy_labels]

        y_train_noisy[sample_idx] = np.array(client_label_noisy_labels)
        real_noise_level[i] = actual_noise_rate

    return (y_train_noisy, gamma_s, real_noise_level)



def get_num_of_each_class(dataset, client_idxs, n_classes):
    class_sum = np.array([0] * n_classes)
    for idx in client_idxs:
        label = dataset.targets[idx]
        class_sum[label] += 1
    return class_sum.tolist()


class avg_acc():
    def __init__(self) -> None:
        # self.clean_acc_num = None 
        # self.noisy_acc_num = None
        # self.all_num = None
        self.Flag = False
        

    def append(self,acc_info_dict) -> None:
        if not self.Flag:
            self.clean_acc_num  = copy.deepcopy(acc_info_dict['class_pre_clean_acc'])
            self.noisy_acc_num  = copy.deepcopy(acc_info_dict['class_pre_noisy_acc'])
            self.all_num  = copy.deepcopy(acc_info_dict['class_smaple_num'])
            self.Flag = True
        else:
            self.clean_acc_num  += acc_info_dict['class_pre_clean_acc']
            self.noisy_acc_num  += acc_info_dict['class_pre_noisy_acc']
            self.all_num  += acc_info_dict['class_smaple_num']
    
    def print_all_acc(self,add_info='') :
        logging.info(f"{add_info}_clean_acc_num:{np.sum(self.clean_acc_num)},{add_info}_noisy_acc_num:{np.sum(self.noisy_acc_num)},all_smaple_num:{np.sum(self.all_num)}，detect_acc = :{(np.sum(self.clean_acc_num) + np.sum(self.noisy_acc_num))/np.sum(self.all_num) }")
        logging.info(f"clean_acc_num:{self.clean_acc_num},_noisy_acc_num:{self.noisy_acc_num},all_smaple_num:{self.all_num}，detect_acc = :{(self.clean_acc_num + self.noisy_acc_num)/self.all_num }")
        
        
