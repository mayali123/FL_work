source activate FedCorr
# 测试案例 
# python main_fed_LNL.py --deterministic 1 --gpu 7 --verbose --seed 42  --dataset cifar10 --method fedrn --epochs 3 --warmup_epochs 1 --num_users 20 --frac 0.5  --local_ep 1 --local_bs 16 --bs 16  --model Resnet18   --num_neighbors 2  --level_n_system 0 --level_n_lowerb 0  --level_n_upperb 0  --noise_type  None

# (0,0)*0
# python main_fed_LNL.py --deterministic 1 --gpu 7 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0 --level_n_lowerb 0  --level_n_upperb 0  --noise_type  None
# # （0.1~0.3）*1
# python main_fed_LNL.py --deterministic 1 --gpu 0 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 
# （0.3~0.5）*1
# python main_fed_LNL.py --deterministic 1 --gpu 3 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 

# # （0.1~0.3）*0.5
# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 
# （0.3~0.5）*0.5
# python main_fed_LNL.py --deterministic 1 --gpu 7 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 


# 全局噪声异构
# （0.5 * 0.9 + 0.5 * 0.1）* 1 
# python main_fed_LNL.py --deterministic 1 --gpu 3 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1 --big_noisy_prob 0.5  --big_noisy 0.9  --noise_type overall_class_noisy_heterogeneous 


# （0.5~0.5）*0.5
# python main_fed_LNL.py --deterministic 1 --gpu 7 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1  --level_n_lowerb 0.5 --level_n_upperb 0.5  --noise_type symmetric 






# # （0.1~0.3）* 1 
# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type pairflip 
# （0.3~0.5）* 1
# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type pairflip 


# 测试
# # （0.1~0.3）* 1 
# python main_fed_LNL.py --deterministic 1 --gpu 1 --seed 42  --dataset cifar10 --method fedrn --epochs 3 --warmup_epochs 1 --num_users 20 --frac 0.5  --local_ep 1 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type pairflip 


# sys 
# （0.3~0.5）* 1
# python main_fed_LNL.py --deterministic 1 --gpu 1 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 
# pair
# （0.3~0.5）* 1
# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type pairflip 



# sys 
# （0.3~0.5）* 0.5
# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 
# pair
# （0.3~0.5）* 0.5
# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type pairflip 


# （non-iid） (0.7,10)
# pair
# （0.3~0.5）* 1
# python main_fed_LNL.py --deterministic 1 --gpu 4 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type pairflip --iid 0   --non_iid_prob_class 0.7 --alpha_dirichlet 10

# pair
# （0.3~0.5）* 0.5
# python main_fed_LNL.py --deterministic 1 --gpu 4 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type pairflip --iid 0   --non_iid_prob_class 0.7 --alpha_dirichlet 10

# iid
# sys 
# （0.3~0.5）* 1
# 不要微调头
# python main_fed_LNL.py --deterministic 1 --gpu 1 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 



# 用于保存 预热之后的模型 用于后面的debug
# pair
# （0.3~0.5）* 1
# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 1  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type pairflip 

# 查看一下 一些变量的shape
# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42  --dataset cifar10 --method fedrn --epochs 10 --warmup_epochs 5 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type pairflip 
# 查看一下 在干净客户端下是否能检测出来
# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42  --dataset cifar10 --method fedrn --epochs 25 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 



# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42 --start 20  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric 



# pair
# （0.3~0.5）* 1
# python main_fed_LNL.py --deterministic 1 --gpu 2 --seed 42  --dataset cifar10 --method fedrn --epochs 200 --warmup_epochs 20 --start 20 --num_users 20 --frac 0.5  --local_ep 5 --local_bs 64 --bs 64  --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type pairflip 


python main_fed_LNL.py --deterministic 1 --gpu 4 --seed 42  --dataset ICH --method fedrn --epochs 3 --warmup_epochs 1  --num_users 20 --frac 1  --local_ep 1 --local_bs 64 --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type random

# 0.5*(0.3~0.5)
# python main_fed_LNL.py --deterministic 1 --gpu 4 --seed 42  --dataset ICH --method fedrn --epochs 100 --warmup_epochs 10  --num_users 20 --frac 1  --local_ep 5 --local_bs 64 --model Resnet18 --pretrained 1  --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type random