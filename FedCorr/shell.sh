source activate FedCorr
# （0.3~0.5）* 1 第三个
# python main.py --deterministic 1 --seed 42 --gpu 5 --iteration1 5 --rounds1 200 --rounds2 200 --local_ep 10 --frac1 1  --frac2 0.1 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.5 --noise_type symmetric --num_users 20 --iid --dataset cifar10
# python main.py --deterministic 1 --seed 42 --gpu 5 --iteration1 5 --rounds1 200 --rounds2 200 --local_ep 10 --frac1 1  --frac2 0.1 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.3 --noise_type symmetric --num_users 20 --iid --dataset cifar10
# # (0.1~0.3) * 0.5
# python main.py --deterministic 1 --seed 42 --gpu 4 --iteration1 5 --rounds1 200 --rounds2 200 --local_ep 10 --frac1 1  --frac2 0.1 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3 --noise_type symmetric --num_users 20 --iid --dataset cifar10


## ----non-iid ---
# （0.1~0.3）* 1 第六个
# python main.py --deterministic 1 --seed 42 --gpu 5 --iteration1 5 --rounds1 200 --rounds2 200 --local_ep 10 --frac1 1  --frac2 0.1 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3 --noise_type symmetric --num_users 20 --non_iid_prob_class 0.7 --alpha_dirichlet 10 --dataset cifar10




# 测试 类别噪声异构
# python main.py --deterministic 1 --seed 42 --gpu 5 --iteration1 5 --rounds1 200 --rounds2 200 --local_ep 10 --frac1 1  --frac2 0.1 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3 --noise_type class_noisy_heterogeneous --class_n_lowerb 0.1 --class_n_upperb 0.5 --num_users 20 --iid --dataset cifar10
# python main_noisy.py --deterministic 1 --seed 42 --gpu 0 --dataset cifar10 --batch_size 64  --n_clients 20 --iid 1 --local_ep 10 --rounds 200 --frac 1 --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type class_noisy_heterogeneous --class_n_lowerb 0.1 --class_n_upperb 0.5


# 2024.2.26 16:32
# SYS (0.1~0.3) * 0.5
# python main.py --deterministic 1 --seed 42 --gpu 4 --iteration1 5 --rounds1 200 --rounds2 200 --local_ep 10 --frac1 1  --frac2 0.1 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3 --noise_type symmetric --num_users 20 --iid --dataset cifar10

# SYS (0.3~0.5) * 0.5
# python main.py --deterministic 1 --seed 42 --gpu 4 --iteration1 5 --rounds1 200 --rounds2 200 --local_ep 10 --frac1 1  --frac2 0.1 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5 --noise_type symmetric --num_users 20 --iid --dataset cifar10

# SYS  0
# python main.py --deterministic 1 --seed 42 --gpu 5 --iteration1 5 --rounds1 200 --rounds2 200 --local_ep 10 --frac1 1  --frac2 0.1 --local_bs 64 --mixup  --level_n_system 0  --level_n_lowerb 0 --level_n_upperb 0 --noise_type symmetric --num_users 20 --iid --dataset cifar10


# ----- 2024.3.1 ------  
# SYS (0~0) * 0
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 0  --level_n_lowerb 0  --level_n_upperb 0 --noise_type None --num_users 20 --iid --dataset cifar10

# # SYS (0.1~0.3) * 0.5
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.1 --level_n_upperb 0.3 --noise_type symmetric --num_users 20 --iid --dataset cifar10

# # SYS  (0.1~0.3) * 1
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.1 --level_n_upperb 0.3 --noise_type symmetric --num_users 20 --iid --dataset cifar10


# # SYS (0.3~0.5) * 0.5
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.3 --level_n_upperb 0.5 --noise_type symmetric --num_users 20 --iid --dataset cifar10

# # SYS  (0.3~0.5) * 1
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.3 --level_n_upperb 0.5 --noise_type symmetric --num_users 20 --iid --dataset cifar10



# -------2024.3.11 ------
# 噪声异构的情况 （0.5 * 0.9 + 0.5 * 0.1）* 0.5
# python main.py --deterministic 1 --seed 42 --gpu 1 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 0.5  --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type class_noisy_heterogeneous  --num_users 20 --iid --dataset cifar10
# SYS (0.5~0.5) * 0.5
# python main.py --deterministic 1 --seed 42 --gpu 1 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.5 --level_n_upperb 0.5 --noise_type symmetric --num_users 20 --iid --dataset cifar10

# -------2024.3.14 ------
# 噪声异构的情况 （0.5 * 0.9 + 0.5 * 0.1）* 1
# python main.py --deterministic 1 --seed 42 --gpu 7 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 1  --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type class_noisy_heterogeneous  --num_users 20 --iid --dataset cifar10
# SYS (0.5~0.5) * 1
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.5 --level_n_upperb 0.5 --noise_type symmetric --num_users 20 --iid --dataset cifar10


# -------2024.3.14 ------
# -------- 全局噪声异构 ------- 
# 噪声异构的情况 （0.5 * 0.9 + 0.5 * 0.1）* 0.5
# python main.py --deterministic 1 --seed 42 --gpu 0 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 0.5  --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type overall_class_noisy_heterogeneous  --num_users 20 --iid --dataset cifar10
# 噪声异构的情况 （0.5 * 0.9 + 0.5 * 0.1）* 1
# python main.py --deterministic 1 --seed 42 --gpu 7 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 1 --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type overall_class_noisy_heterogeneous  --num_users 20 --iid --dataset cifar10


# ------2024.3.15 ---------
# SYS (0.5~0.5) * 1
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.5 --level_n_upperb 0.5 --noise_type symmetric --num_users 20 --iid --dataset cifar10




# pair
# (0.3~0.5) * 1
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.3 --level_n_upperb 0.5 --noise_type pairflip --num_users 20 --iid --dataset cifar10
# (0.3~0.5) * 0.5
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.3 --level_n_upperb 0.5 --noise_type pairflip --num_users 20 --iid --dataset cifar10

# non-iid (0.7,10)
# sys
# (0.3~0.5) * 1
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.3 --level_n_upperb 0.5 --noise_type symmetric --num_users 20 --iid --dataset cifar10  --non_iid_prob_class 0.7 --alpha_dirichlet 10 
# (0.3~0.5) * 0.5
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.3 --level_n_upperb 0.5 --noise_type symmetric --num_users 20 --iid --dataset cifar10  --non_iid_prob_class 0.7 --alpha_dirichlet 10 

# pair
# (0.3~0.5) * 1
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 1  --level_n_lowerb 0.3 --level_n_upperb 0.5 --noise_type pairflip --num_users 20 --iid --dataset cifar10  --non_iid_prob_class 0.7 --alpha_dirichlet 10 
# (0.3~0.5) * 0.5
# python main.py --deterministic 1 --seed 42 --gpu 2 --iteration1 5 --rounds1 90 --rounds2 100 --local_ep 5 --frac1 0.05  --frac2 0.5 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.3 --level_n_upperb 0.5 --noise_type pairflip --num_users 20 --iid --dataset cifar10  --non_iid_prob_class 0.7 --alpha_dirichlet 10 



# python main.py --deterministic 1 --seed 42 --gpu 4 --iteration1 5 --rounds1 45 --rounds2 50 --local_ep 5 --frac1 0.05  --frac2 1 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.3 --level_n_upperb 0.5  --num_users 20 --dataset ICH --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type random 


python main.py --deterministic 1 --seed 42 --gpu 4 --iteration1 2 --rounds1 2 --rounds2 2 --local_ep 1 --frac1 0.05  --frac2 1 --local_bs 64 --mixup  --level_n_system 0.5  --level_n_lowerb 0.3 --level_n_upperb 0.5  --num_users 20 --dataset ICH --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type random 
