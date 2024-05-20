source activate FedNoRo
# (0.5~0.7)*0.4 
python train_FedNoRo.py --deterministic 1 --seed 42 --gpu 2 --dataset ICH --model Resnet18 --batch_size 64 --pretrained 1 --n_clients 20 --iid 0 --non_iid_prob_class 0.9 --alpha_dirichlet 2.0 --local_ep 1 --rounds 3 --s1 1 --level_n_system 0.4 --level_n_lowerb 0.5 --level_n_upperb 0.7 --n_type symmetric