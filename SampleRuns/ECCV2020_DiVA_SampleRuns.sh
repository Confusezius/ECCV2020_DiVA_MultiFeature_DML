"""========== CUB 200-2011 ==========="""
### ResNet50
python diva_main.py --dataset cub200 --log_online --project DiVA_SampleRuns --diva_ssl fast_moco --source_path $datapath --group CUB_DiVA-R50-512 --tau 55 --gamma 0.2 --diva_alpha_ssl 0.3 --diva_alpha_shared 0.3 --diva_alpha_intra 0.3 --diva_rho_decorrelation 1500 1500 1500 --diva_features discriminative selfsimilarity shared intra --diva_sharing random --evaltypes all --diva_moco_temperature 0.01  --diva_moco_n_key_batches 30 --n_epochs 350 --seed 0 --gpu 0 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --embed_dim 128
### Inception-BN
python diva_main.py --dataset cub200 --log_online --project DiVA_SampleRuns --diva_ssl fast_moco --source_path $datapath --group CUB_DiVA-IBN-512 --gamma 0.3 --tau 70 --diva_alpha_ssl 0.1 --diva_alpha_shared 0.1 --diva_alpha_intra 0.1 --diva_rho_decorrelation 300 300 300 --diva_features discriminative selfsimilarity shared intra --diva_sharing random --evaltypes all --diva_moco_temperature 0.01  --diva_moco_n_key_batches 30 --n_epochs 350 --seed 0 --gpu 0 --samples_per_class 2 --loss margin --batch_mining distance --arch bninception_normalize --embed_dim 128


"""========== CARS196 ==========="""
### ResNet50
python diva_main.py --dataset cars196 --log_online --project DiVA_SampleRuns --diva_ssl fast_moco --source_path $datapath --group CAR_DiVA-R50-512 --gamma 0.2 --tau 70 --diva_alpha_ssl 0.15 --diva_alpha_shared 0.15 --diva_alpha_intra 0.15 --diva_rho_decorrelation 100 100 100 --diva_features discriminative selfsimilarity shared intra --diva_sharing random --evaltypes all --diva_moco_temperature 0.01  --diva_moco_n_key_batches 30 --n_epochs 350 --seed 0 --gpu 0 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --embed_dim 128
### Inception-BN
python diva_main.py --dataset cars196 --log_online --project DiVA_SampleRuns --diva_ssl fast_moco --source_path $datapath --group CAR_DiVA-IBN-512 --gamma 0.3 --tau 160 --diva_alpha_ssl 0.15 --diva_alpha_shared 0.15 --diva_alpha_intra 0.15 --diva_rho_decorrelation 100 100 100 --diva_features discriminative selfsimilarity shared intra --diva_sharing random --evaltypes all --diva_moco_temperature 0.01  --diva_moco_n_key_batches 30 --n_epochs 350 --seed 1 --gpu 0 --samples_per_class 2 --loss margin --batch_mining distance --arch bninception_normalize --embed_dim 128


"""========== Online Products ==========="""
### ResNet50
python diva_main.py --dataset online_products --log_online --project DiVA_SampleRuns --diva_ssl fast_moco --source_path $datapath --group SOP_DiVA-R50-512 --gamma 0.3 --tau 70 --diva_alpha_ssl 0.2 --diva_alpha_shared 0.2 --diva_alpha_intra 0.2 --diva_rho_decorrelation 150 150 150 --diva_features discriminative selfsimilarity shared intra --diva_sharing random --evaltypes all --diva_moco_temperature 0.01  --diva_moco_n_key_batches 70 --n_epochs 150 --seed 0 --gpu 0 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize --embed_dim 128
### Inception-BN
python diva_main.py --dataset online_products --log_online --project DiVA_SampleRuns --diva_ssl fast_moco --source_path $datapath --group SOP_DiVA-IBN-512 --gamma 0.3 --tau 80 --diva_alpha_ssl 0.1 --diva_alpha_shared 0.1 --diva_alpha_intra 0.1 --diva_rho_decorrelation 150 150 150 --diva_features discriminative selfsimilarity shared intra --diva_sharing random --evaltypes all --diva_moco_temperature 0.01  --diva_moco_n_key_batches 70 --n_epochs 150 --seed 0 --gpu 0 --samples_per_class 2 --loss margin --batch_mining distance --arch bninception_normalize --embed_dim 128



"""==== INCLUDE DIFFERENT SELF-SUPERVISION APPROACHES ===="""
### e.g. Deep Clustering
python diva_main.py --source_path $datapath --log_online --project DiVA_Experiments --diva_rho_decorrelation 500 --diva_features discriminative dc --evaltypes all --diva_dc_update_f 2 --diva_dc_ncluster 300 --n_epochs 200 --seed 0 --gpu $gpu --bs 104 --samples_per_class 2 --loss margin --batch_mining distance --arch bninception_normalize --embed_dim 256
