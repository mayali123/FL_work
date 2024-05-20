#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import sys

def args_parser():
    parser = argparse.ArgumentParser()
    
    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='5', help='GPU to use')
    
    # training arguments
    parser.add_argument('--model', type=str,default='Resnet18', help='model name')
    parser.add_argument('--pretrained', type=int,  default=1)
    parser.add_argument('--epochs', type=int, default=1000, help="rounds of training")
    parser.add_argument('--base_lr', type=float, default=3e-4, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay size")
    parser.add_argument('--schedule', nargs='+', default=[], help='decrease learning rate at these epochs.')
    parser.add_argument('--feature_dim', type=int, help = 'feature dimension', default=512)
    
    # FL arguments
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    
    # dataset arguments
    parser.add_argument('--dataset', type=str, default='ICH', help="name of dataset")
    parser.add_argument('--iid', type=int, default=1, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float,
                        default=0.9, help='parameter for non-iid')
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=2.0, help='parameter for non-iid')

    # noise arguments
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")
    parser.add_argument('--level_n_upperb', type=float, default=0.7, help="upper bound of noise level")
    parser.add_argument('--n_type', type=str, default='None',
                        help="type of noise pairflip or symmetric or None")
    
    parser.add_argument('--num_gradual', type=int, default=10, help='T_k')
    parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate")
    
    # "Robust Federated Learning with Noisy Labels" arguments
    parser.add_argument('--T_pl', type=int, help = 'T_pl: When to start using global guided pseudo labeling', default=100)
    parser.add_argument('--lambda_cen', type=float, help = 'lambda_cen', default=1.0)
    parser.add_argument('--lambda_e', type=float, help = 'lambda_e', default=0.8)
    
    
    args = parser.parse_args()
    return args
