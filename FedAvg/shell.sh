source activate FedNoRo
# source activate FedCorr
# 测试 类别噪声异构
# python main_noisy.py --deterministic 1 --seed 42 --gpu 0 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 10 --rounds 200 --frac 1 --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type class_noisy_heterogeneous --class_n_lowerb 0.1 --class_n_upperb 0.5
# （0.3~0.5）* 1 第四个
# python main_noisy.py --deterministic 1 --seed 42 --gpu 0 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 10 --rounds 200 --frac 1 --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 
#  (0.1~0.3)*0.5
# python main_noisy.py --deterministic 1 --seed 42 --gpu 4 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 10 --rounds 200 --frac 1 --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 


# python main_noisy.py --deterministic 1 --seed 42 --gpu 5 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 10 --rounds 200 --frac 1 --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.3  --noise_type symmetric 


## ----non-iid ---
# （0.1~0.3）* 1 第五个
# python main_noisy.py --deterministic 1 --seed 42 --gpu 0 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 0 --non_iid_prob_class 0.7 --alpha_dirichlet 10 --local_ep 10 --rounds 200 --frac 1 --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 


# ------ 2024.2.26 16:37 ---------

#  (0.1~0.3)*0.5
# python main_noisy.py --deterministic 1 --seed 42 --gpu 5 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 10 --rounds 200 --frac 0.1 --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 

#  (0.3~0.5)*0.5
# python main_noisy.py --deterministic 1 --seed 42 --gpu 4 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 10 --rounds 200 --frac 1 --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 

# --------2024.2.27 -----------
# (0 ~ 0) * 0 
# python main_noisy.py --deterministic 1 --seed 42 --gpu 2 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 10 --rounds 400 --frac 0.1 --level_n_system 0  --level_n_lowerb 0  --level_n_upperb 0  --noise_type None 



# --------2024.2.28 -----------
# (0 ~ 0) * 0 
# python main_noisy.py --deterministic 1 --seed 42 --gpu 2 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 0  --level_n_lowerb 0  --level_n_upperb 0  --noise_type None 
# (0.1 ~ 0.3) * 0.5
# python main_noisy.py --deterministic 1 --seed 42 --gpu 4 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 
# (0.1 ~ 0.3) * 1
# python main_noisy.py --deterministic 1 --seed 42 --gpu 4 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 

# (0.3 ~ 0.5) * 0.5
# python main_noisy.py --deterministic 1 --seed 42 --gpu 2 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 
# (0.3 ~ 0.5) * 1
# python main_noisy.py --deterministic 1 --seed 42 --gpu 2 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 

# --------2024.2.29 -----------
# (0.1 ~ 0.3) * 0.5
# python main_for_detect.py --deterministic 1 --seed 42 --gpu 6 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 
# python detect.py --deterministic 1 --seed 42 --gpu 2 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 



# --------2024.3.10 ---------
# 噪声异构的情况 （0.5 * 0.9 + 0.5 * 0.1）* 0.5
# python main_noisy.py --deterministic 1 --seed 42 --gpu 1 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 0.5  --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type class_noisy_heterogeneous 
# sys  (0.5 ~ 0.5) * 0.5
# python main_noisy.py --deterministic 1 --seed 42 --gpu 1 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 0.5  --level_n_lowerb 0.5  --level_n_upperb 0.5  --noise_type symmetric 
# ------2024.3.14 ----------
# 噪声异构的情况 （0.5 * 0.9 + 0.5 * 0.1）* 1
# python main_noisy.py --deterministic 1 --seed 42 --gpu 1 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 1  --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type class_noisy_heterogeneous 
# sys  (0.5 ~ 0.5) * 1
# python main_noisy.py --deterministic 1 --seed 42 --gpu 2 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 1  --level_n_lowerb 0.5  --level_n_upperb 0.5  --noise_type symmetric 


# 测试用 噪声异构的情况 （0.5 * 0.9 + 0.5 * 0.1）* 0.25
# python main_noisy.py --deterministic 1 --seed 42 --gpu 1 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 0.25 --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type class_noisy_heterogeneous 

# ------- 全局噪声噪声异构  ----- 
# （0.5 * 0.9 + 0.5 * 0.1）* 0.5 
# python main_noisy.py --deterministic 1 --seed 42 --gpu 4 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 0.5 --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type overall_class_noisy_heterogeneous 
# （0.5 * 0.9 + 0.5 * 0.1）* 1 
# python main_noisy.py --deterministic 1 --seed 42 --gpu 4 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 1 --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type overall_class_noisy_heterogeneous 


# ------- sys  ----- 
# sys  (0.5 ~ 0.5) * 1
# python main_noisy.py --deterministic 1 --seed 42 --gpu 4 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 1  --level_n_lowerb 0.5  --level_n_upperb 0.5  --noise_type symmetric 



# ------2024.3.14 ----------
# 噪声异构的情况 （0.5 * 0.9 + 0.5 * 0.1）* 1
# python main_noisy.py --deterministic 1 --seed 42 --gpu 2 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 5 --rounds 200 --frac 0.5 --level_n_system 1  --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type class_noisy_heterogeneous 



python main_noisy.py --deterministic 1 --seed 42 --gpu 4 --dataset ICH  --model Resnet18 --batch_size 64 --pretrained 1 --n_clients 20  --local_ep 1 --rounds 20 --frac 1 --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3 --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type random

# (0.1~0.3)*0.5
# python main_noisy.py --deterministic 1 --seed 42 --gpu 4 --dataset ICH  --model Resnet18 --batch_size 64 --pretrained 1 --n_clients 20  --local_ep 5 --rounds 100 --frac 1 --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3 --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type random
