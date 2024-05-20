import argparse
import os
def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='5', help='GPU to use')
    # dataset
    parser.add_argument('--dataset', type=str,
                        default='ICH', help='dataset name')
    
    parser.add_argument('--model', type=str,
                        default='Resnet18', help='model name')
    parser.add_argument('--pretrained', type=int,  default=1)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--global_rounds', type=int, default=100)
    parser.add_argument('--warm_round', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--join_ratio', type=float, default=0.5)
    parser.add_argument('--global_learning_rate', type=float, default=0.01)
    parser.add_argument('--local_learning_rate', type=float, default=3e-4)
    parser.add_argument('--local_steps', type=int, default=10)
    parser.add_argument('--threshold', default=0.95, type=float,help='pseudo label threshold')
    parser.add_argument('--lamda', type=float, default=0.12)


    # noise
    parser.add_argument('--level_n_system', type=float, default=0.5, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.1, help="lower bound of noise level")
    parser.add_argument('--level_n_upperb', type=float, default=0.3, help="upper bound of noise level")
    parser.add_argument('--n_type', type=str, default='symmetric',
                        help="type of noise pairflip or symmetric or None")

    #non-iid
    parser.add_argument('--iid', type=int, default=1)
    parser.add_argument('--non_iid_prob_class', type=float,
                        default=0.9, help='parameter for non-iid')
    parser.add_argument('--alpha_dirichlet', type=float,
                        default=2.0, help='parameter for non-iid')
    # parser.add_argument('--non_iid_alpha', type=float, default=0.7)

    #distill
    parser.add_argument('--temperature', type=float, default=2)
    parser.add_argument('--mini_batch_size_distillation', type=int, default=128)
    parser.add_argument('--ld', type=float, default=0.5, help='threshold of distillation aggregate')
    parser.add_argument('--eval_step', default=78, type=int , help='number of eval steps to run')


    args = parser.parse_args()
    return args