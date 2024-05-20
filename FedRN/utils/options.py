#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--gpu', type=str, default='2', help="GPU ID")
    parser.add_argument('--start', type=int, default=0, help="start epoch")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    # label noise method
    parser.add_argument('--method', type=str, default='fedrn',
                        choices=['default', 'selfie', 'jointoptim', 'coteaching', 'coteaching+', 'dividemix', 'fedrn'],
                        help='method name')

    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--base_lr', type=float, default=3e-4, help="learning rate")
    parser.add_argument('--schedule', nargs='+', default=[], help='decrease learning rate at these epochs.')
    parser.add_argument('--lr_decay', type=float, default=0.1, help="learning rate decay")
    parser.add_argument('--iid', type=int, default=1, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float,
                        default=0.9, help='parameter for non-iid')
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=2.0, help='parameter for non-iid')
    parser.add_argument('--fed_method', type=str, default='fedavg', choices=['fedavg'],
                        help="federated learning method")

    # model arguments
    parser.add_argument('--model', type=str, default='Resnet18', choices=['Resnet18','Resnet34'], help='model name')
    parser.add_argument('--pretrained', type=int,  default=1)

    # other arguments
    parser.add_argument('--dataset', type=str, default='ICH', help="name of dataset")
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers to load data')

    # noise label arguments
    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")
    parser.add_argument('--level_n_upperb', type=float, default=0.7, help="upper bound of noise level")
    parser.add_argument('--n_type', type=str, default='None',
                        help="type of noise pairflip or symmetric or None")


    parser.add_argument('--warmup_epochs', type=int, default=100, help='number of warmup epochs')

    # SELFIE / Joint optimization arguments
    parser.add_argument('--queue_size', type=int, default=15, help='size of history queue')
    # SELFIE / Co-teaching arguments
    parser.add_argument('--forget_rate', type=float, default=0.2, help="forget rate for co-teaching")
    # SELFIE arguments
    parser.add_argument('--uncertainty_threshold', type=float, default=0.05, help='uncertainty threshold')
    # Joint optimization arguments
    parser.add_argument('--alpha', type=float, default=1.2, help="alpha for joint optimization")
    parser.add_argument('--beta', type=float, default=0.8, help="beta for joint optimization")
    parser.add_argument('--labeling', type=str, default='soft', help='[soft, hard]')
    # MixMatch arguments
    parser.add_argument('--mm_alpha', default=4, type=float)
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--p_threshold', default=0.5, type=float)

    # FedRN
    parser.add_argument('--num_neighbors', type=int, default=2, help="number of neighbors")
    parser.add_argument('--w_alpha', type=float, help='weight alpha for our method', default=0.5)

    args = parser.parse_args()
    return args
