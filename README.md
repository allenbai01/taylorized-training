# taylorized-training

This repository contains code for training Taylorized neural networks from the paper

[Taylorized Training: Towards Better Approximation of Neural Network Training at Finite Width](https://arxiv.org/abs/2002.04010) by Yu Bai, Ben Krause, Huan Wang, Caiming Xiong, and Richard Socher, 2020.

## Prerequisites
Requires Python >=3.6 and the following prerequisites.
```
# install tensorflow and tensorboard
pip install tensorflow-gpu, tensorflow_datasets, tensorboardX

# install jax
# note: please check newest version number for jaxlib
PYTHON_VERSION=cp36  # alternatives: cp36, cp37, cp38
CUDA_VERSION=cuda100  # alternatives: cuda92, cuda100, cuda101, cuda102
PLATFORM=linux_x86_64  # alternatives: linux_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.42-$PYTHON_VERSION-none-$PLATFORM.whl
pip install --upgrade jax  # install jax

# install neural_tangents
pip install neural_tangents
```
Our code is written in [Jax](https://github.com/google/jax) and uses the Taylorization functionality from [Neural Tangents](https://github.com/google/neural-tangents). We currently require Tensorflow in addition for some (very light) data processing routines.

## Training neural networks and their Taylorizations
1. Train a 4-layer CNN with 128 channels per layer, saving the model parameters and test logits.
```
python taylorized_train.py \
    --epochs 200 \
    --init_seed 100 \
    --model cnn --gap --n_channels 128 --n_layers 4 \
    --loss logistic \
    --parameterization standard \
    --optimizer momentum --lr 0.1 --momentum 0.0 --weight_decay 0.0 \
    --lr_decay --decay_epoch 100 --decay_epoch_2 150 --decay_factor 0.1 \
    --grad_norm_thresh 5 \
    --batch_size_train 256 \
    --batch_size_test 256 \
    --logdir runs/CNNTHIN-lr-0.1-clip-5-bs-256 \
    --save_steps 200 \
    --early_save_steps 25 \
    --early_save_till_step 200 \
    --save_path saved_models/CNNTHIN-lr-0.1-clip-5-bs-256
```

2. Train {linearized, quadratic, cubic, quartic} Taylorized versions of the 4-layer CNN from the same initialization (with parallelization over CUDA devices.)

    *Note*: Taylorized training uses the exact same random seeds as full training (for SGD and data augmentation noise.)
```
CUDA_VISIBLE_DEVICES=0 python taylorized_train.py  --linearize  --epochs 200  --init_seed 100  --model cnn --gap --n_channels 128 --n_layers 4  --loss logistic  --parameterization standard  --optimizer momentum --lr 0.1 --momentum 0.0 --weight_decay 0.0  --lr_decay --decay_epoch 100 --decay_epoch_2 150 --decay_factor 0.1  --grad_norm_thresh 5  --batch_size_train 256  --batch_size_test 256  --logdir runs/CNNTHIN-LIN-lr-0.1-clip-5-bs-256  --save_steps 200  --early_save_steps 25  --early_save_till_step 200  --load_path saved_models/CNNTHIN-lr-0.1-clip-5-bs-256/0.npy  --save_path saved_models/CNNTHIN-LIN-lr-0.1-clip-5-bs-256 &
CUDA_VISIBLE_DEVICES=1 python taylorized_train.py  --expand_order 2  --epochs 200  --init_seed 100  --model cnn --gap --n_channels 128 --n_layers 4  --loss logistic  --parameterization standard  --optimizer momentum --lr 0.1 --momentum 0.0 --weight_decay 0.0  --lr_decay --decay_epoch 100 --decay_epoch_2 150 --decay_factor 0.1  --grad_norm_thresh 5  --batch_size_train 256  --batch_size_test 256  --logdir runs/CNNTHIN-QUAD-lr-0.1-clip-5-bs-256  --save_steps 200  --early_save_steps 25  --early_save_till_step 200  --load_path saved_models/CNNTHIN-lr-0.1-clip-5-bs-256/0.npy  --save_path saved_models/CNNTHIN-QUAD-lr-0.1-clip-5-bs-256 &
CUDA_VISIBLE_DEVICES=2 python taylorized_train.py  --expand_order 3  --epochs 200  --init_seed 100  --model cnn --gap --n_channels 128 --n_layers 4  --loss logistic  --parameterization standard  --optimizer momentum --lr 0.1 --momentum 0.0 --weight_decay 0.0  --lr_decay --decay_epoch 100 --decay_epoch_2 150 --decay_factor 0.1  --grad_norm_thresh 5  --batch_size_train 256  --batch_size_test 256  --logdir runs/CNNTHIN-CUBIC-lr-0.1-clip-5-bs-256  --save_steps 200  --early_save_steps 25  --early_save_till_step 200  --load_path saved_models/CNNTHIN-lr-0.1-clip-5-bs-256/0.npy  --save_path saved_models/CNNTHIN-CUBIC-lr-0.1-clip-5-bs-256 &
CUDA_VISIBLE_DEVICES=3 python taylorized_train.py  --expand_order 4  --epochs 200  --init_seed 100  --model cnn --gap --n_channels 128 --n_layers 4  --loss logistic  --parameterization standard  --optimizer momentum --lr 0.1 --momentum 0.0 --weight_decay 0.0  --lr_decay --decay_epoch 100 --decay_epoch_2 150 --decay_factor 0.1  --grad_norm_thresh 5  --batch_size_train 256  --batch_size_test 256  --logdir runs/CNNTHIN-QUARTIC-lr-0.1-clip-5-bs-256  --save_steps 200  --early_save_steps 25  --early_save_till_step 200  --load_path saved_models/CNNTHIN-lr-0.1-clip-5-bs-256/0.npy  --save_path saved_models/CNNTHIN-QUARTIC-lr-0.1-clip-5-bs-256
```

3. Monitor the results on Tensorboard.
```
tensorboard --logdir runs
```

Further setups (such as training WideResNets) can be found in the provided shell scripts.
