CUDA_VISIBLE_DEVICES=0 python taylorized_train.py  --epochs 200  --init_seed 100  --model cnn --gap --n_channels 128 --n_layers 4  --loss logistic  --parameterization standard  --linearize  --optimizer momentum --lr 0.1 --momentum 0.0 --weight_decay 0.0  --lr_decay --decay_epoch 100 --decay_epoch_2 150 --decay_factor 0.1  --grad_norm_thresh 5  --batch_size_train 256  --batch_size_test 256  --logdir runs/CNNTHIN-LIN-lr-0.1-clip-5-bs-256  --save_steps 200  --early_save_steps 25  --early_save_till_step 200  --load_path saved_models/CNNTHIN-lr-0.1-clip-5-bs-256/0.npy  --save_path saved_models/CNNTHIN-LIN-lr-0.1-clip-5-bs-256 &
CUDA_VISIBLE_DEVICES=1 python taylorized_train.py  --epochs 200  --init_seed 100  --model cnn --gap --n_channels 128 --n_layers 4  --loss logistic  --parameterization standard  --expand_order 2  --optimizer momentum --lr 0.1 --momentum 0.0 --weight_decay 0.0  --lr_decay --decay_epoch 100 --decay_epoch_2 150 --decay_factor 0.1  --grad_norm_thresh 5  --batch_size_train 256  --batch_size_test 256  --logdir runs/CNNTHIN-QUAD-lr-0.1-clip-5-bs-256  --save_steps 200  --early_save_steps 25  --early_save_till_step 200  --load_path saved_models/CNNTHIN-lr-0.1-clip-5-bs-256/0.npy  --save_path saved_models/CNNTHIN-QUAD-lr-0.1-clip-5-bs-256 &
CUDA_VISIBLE_DEVICES=2 python taylorized_train.py  --epochs 200  --init_seed 100  --model cnn --gap --n_channels 128 --n_layers 4  --loss logistic  --parameterization standard  --expand_order 3  --optimizer momentum --lr 0.1 --momentum 0.0 --weight_decay 0.0  --lr_decay --decay_epoch 100 --decay_epoch_2 150 --decay_factor 0.1  --grad_norm_thresh 5  --batch_size_train 256  --batch_size_test 256  --logdir runs/CNNTHIN-CUBIC-lr-0.1-clip-5-bs-256  --save_steps 200  --early_save_steps 25  --early_save_till_step 200  --load_path saved_models/CNNTHIN-lr-0.1-clip-5-bs-256/0.npy  --save_path saved_models/CNNTHIN-CUBIC-lr-0.1-clip-5-bs-256 &
CUDA_VISIBLE_DEVICES=3 python taylorized_train.py  --epochs 200  --init_seed 100  --model cnn --gap --n_channels 128 --n_layers 4  --loss logistic  --parameterization standard  --expand_order 4  --optimizer momentum --lr 0.1 --momentum 0.0 --weight_decay 0.0  --lr_decay --decay_epoch 100 --decay_epoch_2 150 --decay_factor 0.1  --grad_norm_thresh 5  --batch_size_train 256  --batch_size_test 256  --logdir runs/CNNTHIN-QUARTIC-lr-0.1-clip-5-bs-256  --save_steps 200  --early_save_steps 25  --early_save_till_step 200  --load_path saved_models/CNNTHIN-lr-0.1-clip-5-bs-256/0.npy  --save_path saved_models/CNNTHIN-QUARTIC-lr-0.1-clip-5-bs-256
