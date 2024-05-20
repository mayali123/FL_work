source activate FedCorr
# python3 main.py --deterministic 1 --seed 42 --gpu 3  --iid 1 --bs 64 --local_bs 64  --num_users 20 --frac 0.5 --dataset cifar10 --local_ep 5 --feature_dim 512 --epochs 3 --num_gradual 1 --level_n_system 0 --level_n_lowerb 0  --level_n_upperb 0  --noise_type None 


# （0，0）*0 
# python3 main.py --deterministic 1 --seed 42 --gpu 3 --epochs 200  --bs 128 --local_bs 64  --num_users 20 --frac 0.5 --dataset cifar10 --local_ep 5 --feature_dim 512  --num_gradual 10 --level_n_system 0 --level_n_lowerb 0  --level_n_upperb 0  --noise_type None --iid 1

# （0.1，0.3）*1 
# python3 main.py --deterministic 1 --seed 42 --gpu 3 --epochs 200  --bs 128 --local_bs 64  --num_users 20 --frac 0.5 --dataset cifar10 --local_ep 5 --feature_dim 512  --num_gradual 10 --level_n_system 1 --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric --iid 1
# （0.1，0.3）*0.5 
# python3 main.py --deterministic 1 --seed 42 --gpu 3 --epochs 200  --bs 128 --local_bs 64  --num_users 20 --frac 0.5 --dataset cifar10 --local_ep 5 --feature_dim 512  --num_gradual 10 --level_n_system 0.5 --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric --iid 1
# # （0.3，0.5）*1 
# python3 main.py --deterministic 1 --seed 42 --gpu 3 --epochs 200  --bs 128 --local_bs 64  --num_users 20 --frac 0.5 --dataset cifar10 --local_ep 5 --feature_dim 512  --num_gradual 10 --level_n_system 1 --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric --iid 1
# # （0.3，0.5）*0.5 
# python3 main.py --deterministic 1 --seed 42 --gpu 3 --epochs 200  --bs 128 --local_bs 64  --num_users 20 --frac 0.5 --dataset cifar10 --local_ep 5 --feature_dim 512  --num_gradual 10 --level_n_system 0.5 --level_n_lowerb 0.3  --level_n_upperb 0.5  --noise_type symmetric --iid 1

# for test（0.1，0.3）* 0.5 
python3 main.py --deterministic 1 --seed 42 --gpu 2 --epochs 200  --local_bs 64  --num_users 20 --frac 0.5 --dataset ICH --local_ep 1 --feature_dim 512  --num_gradual 10 --level_n_system 0.5 --level_n_lowerb 0.1  --level_n_upperb 0.3  --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type symmetric

# （0.1，0.3）* 0.5
# python3 main.py --deterministic 1 --seed 42 --gpu 4 --model Resnet18 --pretrained 1  --epochs 100 --local_ep 1 --local_bs 64  --num_users 20 --frac 1 --dataset ICH --feature_dim 512    --num_gradual 10 --level_n_system 0.5 --level_n_lowerb 0.1  --level_n_upperb 0.3  --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type random
