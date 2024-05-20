source activate FedCorr
# （0.1~0.3）*1
# python main_change.py --gpu 2 --num_clients=20 --join_ratio=0.5 --global_rounds=100 --warm_round=10 --batch_size=128 --mini_batch_size_distillation=128 --temperature=2 --global_learning_rate=0.01 --local_learning_rate=0.05  --lamda=0.12 --join_ratio 1 --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 
# （0~0）*0
# python main_change.py --gpu 2 --num_clients=20 --join_ratio=0.5 --global_rounds=100 --warm_round=10 --batch_size=128 --mini_batch_size_distillation=128 --temperature=2 --global_learning_rate=0.01 --local_learning_rate=0.05  --lamda=0.12 --join_ratio 1 --level_n_system 0  --level_n_lowerb 0  --level_n_upperb 0  --noise_type None 
# (0.1~0.3)*0.5
# python main_change.py --gpu 2 --num_clients=20 --join_ratio=0.5 --global_rounds=100 --warm_round=10 --batch_size=128 --mini_batch_size_distillation=128 --temperature=2 --global_learning_rate=0.01 --local_learning_rate=0.05  --lamda=0.12 --join_ratio 1 --level_n_system 0.5  --level_n_lowerb 0.1  --level_n_upperb 0.3  --noise_type symmetric 



# for test
python main_change.py --deterministic 1 --seed 42  --gpu 4 --dataset ICH  --num_clients 20 --global_rounds 4 --warm_round 2 --local_steps 1 --batch_size 128 --mini_batch_size_distillation 128 --temperature 2 --lamda 0.12 --join_ratio 1 --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3  --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type random


# python main_change.py --deterministic 1 --seed 42  --gpu 2 --dataset ICH  --num_clients 20 --global_rounds 100 --warm_round 10 --local_steps 5 --batch_size 64 --mini_batch_size_distillation 64 --temperature 2 --lamda 0.12 --join_ratio 1 --level_n_system 1  --level_n_lowerb 0.1  --level_n_upperb 0.3  --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type random
