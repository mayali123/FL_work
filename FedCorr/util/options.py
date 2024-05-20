# python version 3.7.1
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='5', help='GPU to use')

    # federated arguments
    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--rounds1', type=int, default=200, help="rounds of training in fine_tuning stage")
    parser.add_argument('--rounds2', type=int, default=200, help="rounds of training in usual training stage")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    parser.add_argument('--frac1', type=float, default=0.05, help="fration of selected clients in preprocessing stage")
    parser.add_argument('--frac2', type=float, default=0.1, help="fration of selected clients in fine-tuning and usual training stage")

    
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--base_lr', type=float, default=3e-4, help="learning rate")
    parser.add_argument('--beta', type=float, default=5, help="coefficient for local proximal，0 for fedavg, 1 for fedprox, 5 for noise fl")

    # noise arguments
    parser.add_argument('--LID_k', type=int, default=20, help="lid")
    parser.add_argument('--mixup', action='store_true')

    parser.add_argument('--level_n_system', type=float, default=0.4, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")
    parser.add_argument('--level_n_upperb', type=float, default=0.7, help="upper bound of noise level")
    parser.add_argument('--n_type', type=str, default='symmetric',
                        help="type of noise pairflip or symmetric or None")
    
    # correction
    parser.add_argument('--relabel_ratio', type=float, default=0.5, help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5, help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1, help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")

    # ablation study
    parser.add_argument('--fine_tuning', action='store_false', help='whether to include fine-tuning stage')
    parser.add_argument('--correction', action='store_false', help='whether to correct noisy labels')

    # for FL
    parser.add_argument('--num_users', type=int, default=20, help="number of uses: K")
    parser.add_argument('--iid', type=int, default=0, help="i.i.d. or non-i.i.d.")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=10)

    # other arguments
    parser.add_argument('--dataset', type=str, default='ICH', help="name of dataset")
    parser.add_argument('--model', type=str,
                        default='Resnet18', help='model name')
    parser.add_argument('--pretrained', type=int,  default=1)
    parser.add_argument('--alpha', type=float, default=1, help="0.1,1,5")


    

    return parser.parse_args()
