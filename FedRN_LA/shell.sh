source activate FedCorr

python main_fed_LNL.py --deterministic 1 --gpu 0 --seed 42  --dataset ICH --method fedrn --epochs 20 --warmup_epochs 5  --num_users 20 --frac 1  --local_ep 1 --local_bs 64 --model Resnet18   --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type symmetric

# 0.5*(0.3~0.5)
# python main_fed_LNL.py --deterministic 1 --gpu 4 --seed 42  --dataset ICH --method fedrn --epochs 100 --warmup_epochs 10  --num_users 20 --frac 1  --local_ep 5 --local_bs 64 --model Resnet18 --pretrained 1  --num_neighbors 2  --level_n_system 0.5  --level_n_lowerb 0.3  --level_n_upperb 0.5  --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0   --n_type symmetric