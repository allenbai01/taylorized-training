from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax.api import value_and_grad
from jax import random

import numpy as onp
from jax.experimental.stax import logsoftmax
from jax.experimental import optimizers
import optimizers as myopt

import neural_tangents as nt

import tensorflow_datasets as tfds

import os, sys
from tensorboardX import SummaryWriter
from models import CNNStandard, WideResnet
from utils import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10',
                    help='Dataset')
parser.add_argument('--data_augment', dest='data_augment', action='store_true')
parser.add_argument('--no_data_augment', dest='data_augment', action='store_false')
parser.set_defaults(data_augment=True)
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--param_seed', type=int, default=0,
                    help='Seed for initializing network.')
parser.add_argument('--init_seed', type=int, default=None,
                    help='Seed at epoch 0. Determines all following seeds for batching and data augmentation.')
parser.add_argument('--seed_separator', type=int, default=1048576,
                    help='Separator for seeds (for deterministic mini-batching).')
parser.add_argument('--batch_size_train', type=int, default=128,
                    help='Batch size for training')
parser.add_argument('--batch_size_test', type=int, default=128,
                    help='Batch size for testing')
parser.add_argument('--linearize', action='store_true',
                    help='Linearize the network')
parser.add_argument('--expand_order', type=int, default=-1,
                    help='Expand the network to higher orders (requires >= 2)')
parser.add_argument('--lr', type=float, default=1.0,
                    help='Learning rate')
parser.add_argument('--linear_lr_decay', action='store_true',
                    help='Use linear learning rate decay schedule.')
parser.add_argument('--lr_decay', action='store_true',
                    help='Use customized learning rate decay schedule.')
parser.add_argument('--decay_epoch', type=int, default=None,
                    help='First epoch to decay the learning rate.')
parser.add_argument('--decay_epoch_2', type=int, default=None,
                    help='Second epoch to decay the learning rate.')
parser.add_argument('--decay_factor', type=float, default=0.1,
                    help='Factor for decaying the learning rate.')
parser.add_argument('--lr_power_of_two', type=float, default=None,
                    help='Learning rate in power of two')
parser.add_argument('--momentum', type=float, default=0.0,
                    help='Momentum')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay')
parser.add_argument('--grad_norm_thresh', type=float, default=0.0,
                    help='gradient norm clipping')
parser.add_argument('--delta_decay', type=float, default=0.95,
                    help='Delta decay (for Hessian-free optimizer).')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='Optimizer')
parser.add_argument('--loss', type=str, default='logistic',
                    help='Loss function (one of logistic/squared)')
parser.add_argument('--centralize_y', action='store_true',
                    help='Centralize the one-hot label encoding to have values (-1/K, (K-1)/K)')
parser.add_argument('--model', type=str, default='cnn',
                    help='Model architecture (cnn/cquad-gp)')
parser.add_argument('--nonlinearity', type=str, default='relu',
                    help='Nonlinearity')
parser.add_argument('--parameterization', type=str, default='standard',
                    help='Model parameterization (standard/ntk/taylor).')
parser.add_argument('--param_order', type=int, default=None,
                    help='Order if use Taylor parameterization.')
parser.add_argument('--batchnorm', action='store_true',
                    help='Use BatchNorm in resnet model.')
parser.add_argument('--wideresnet_bs', type=int, default=10,
                    help='Block size for WideResnet. 2 -- WRN16; 4 -- WRN28.')
parser.add_argument('--wideresnet_k', type=int, default=10,
                    help='k value for WideResnet.')
parser.add_argument('--wideresnet_channels', type=int, default=1024,
                    help='Initial number of channels for WideResnet.')
parser.add_argument('--gap', action='store_true',
                    help='Use Global Average Pooling in CNNs.')
parser.add_argument('--frozen', action='store_true',
                    help='Freeze first and last layer in training')
parser.add_argument('--n_channels', type=int, default=128,
                    help='Number of channels in cnn')
parser.add_argument('--n_quad_features', type=int, default=512,
                    help='Number of quadratic features')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of layers in cnn.')
parser.add_argument('--disp_steps', type=int, default=200,
                    help='Display metrics per x steps.')
parser.add_argument('--logdir', type=str, default=None,
                    help='Directory for tensorboard log file')
parser.add_argument('--save_path', type=str, default=None,
                    help='Path for saving models')
parser.add_argument('--save_steps', type=int, default=None,
                    help='Save the model per x steps.')
parser.add_argument('--early_save_steps', type=int, default=50,
                    help='Save the model per x steps before early_save_till_step')
parser.add_argument('--early_save_till_step', type=int, default=4000)
parser.add_argument('--save_at_end_epoch', action='store_true',
                    help='Save model parameters at the end of the epoch.')
parser.add_argument('--load_path', type=str, default=None,
                    help='Path for loading a pretrained model.')
parser.add_argument('--load_path_param0', type=str, default=None,
                    help='Path for loading a model to taylor expand around.')
parser.add_argument('--show_shapes', action='store_true',
                    help='Show network shapes')

args = parser.parse_args()
if args.lr_power_of_two is not None:
    args.lr = 2 ** args.lr_power_of_two
if args.data == 'cifar10':
    args.num_classes = 10
    args.x_shape = (32, 32, 3)
if args.data == 'cifar100':
    args.num_classes = 100
    args.x_shape = (32, 32, 3)
print(args)

if args.save_path is not None:
    try:
        os.makedirs(args.save_path)
    except OSError:
        pass
    else:
        print("Successfully created the directory %s" % args.save_path)

# Load the dataset #
if args.data == 'cifar10':
    train_data = tfds.load('cifar10', split=tfds.Split.TRAIN)
    test_data = tfds.load('cifar10', split=tfds.Split.TEST)
    n_train, n_test = 50000, 10000
elif args.data == 'cifar100':
    train_data = tfds.load('cifar100', split=tfds.Split.TRAIN)
    test_data = tfds.load('cifar100', split=tfds.Split.TEST)
    n_train, n_test = 50000, 10000

# Define and create the network #
if args.model == 'cnn':
    init_fn, f = CNNStandard(args.n_channels, args.n_layers,
                             data=args.data, gap=args.gap,
                             nonlinearity=args.nonlinearity,
                             parameterization=args.parameterization,
                             order=args.param_order)
elif args.model == 'wideresnet':
    init_fn, f = WideResnet(
        args.wideresnet_bs, args.wideresnet_k, args.num_classes,
        channels=args.wideresnet_channels,
        nonlinearity=args.nonlinearity,
        parameterization=args.parameterization,
        order=args.param_order
    )
else:
    pass
key = random.PRNGKey(args.param_seed)
output_shape, params = init_fn(key, (-1, *args.x_shape))
args.flatten = False

# Examine parameter shapes
if args.show_shapes:
    params_flat, _ = tree_flatten(params)
    print([x.shape for x in params_flat])
    print(f"Total number of parameters={sum([np.prod(x.shape) for x in params_flat])}")
    sys.exit()

# Load pre-saved model #
if args.load_path is not None:
    params = load_jax_params(params, args.load_path)

# (Optionally) Taylorize the network #
tb_flag = 'f'
params_0 = copy_jax_array(params)
if args.load_path_param0 is not None:
    params_0 = load_jax_params(params_0, args.load_path_param0)
if args.linearize:
    f = nt.linearize(f, params_0)
    tb_flag = 'f_lin'
elif args.expand_order >= 2:
    f = nt.taylor_expand(f, params_0, args.expand_order)
    if args.expand_order == 2:
        tb_flag = 'f_quad'
    elif args.expand_order == 3:
        tb_flag = 'f_cubic'
    else:
        tb_flag = f'f_order_{args.expand_order}'


# Define optimizer, loss, and accuracy
# Learning rate decay schedule
if args.lr_decay and args.decay_epoch is not None:
    def lr_schedule(epoch):
        if epoch < args.decay_epoch:
            return args.lr
        elif args.decay_epoch_2 is not None and epoch >= args.decay_epoch_2:
            return args.lr * (args.decay_factor ** 2)
        else:
            return args.lr * args.decay_factor
    lr = lr_schedule
elif args.linear_lr_decay:
    def lr_schedule(epoch):
        return args.lr * (args.epochs - epoch) / args.epochs
    lr = lr_schedule
else:
    lr = args.lr

if args.optimizer == 'sgd':
    opt_init, opt_apply, get_params = myopt.sgd(lr)
elif args.optimizer == 'momentum':
    opt_init, opt_apply, get_params = myopt.momentum(lr, args.momentum,
                                                     weight_decay=args.weight_decay)
elif args.optimizer == 'adagrad':
    opt_init, opt_apply, get_params = optimizers.adagrad(lr, args.momentum)
elif args.optimizer == 'adam':
    opt_init, opt_apply, get_params = optimizers.adam(lr)

state = opt_init(params)

if args.loss == 'logistic':
    loss = lambda fx, y: np.mean(-np.sum(logsoftmax(fx) * y, axis=1))
elif args.loss == 'squared':
    loss = lambda fx, y: np.mean(np.sum((fx - y) ** 2, axis=1))
value_and_grad_loss = jit(value_and_grad(lambda params, x, y: loss(f(params, x), y)))
loss_fn = jit(lambda params, x, y: loss(f(params, x), y))
accuracy_sum = jit(lambda fx, y: np.sum(np.argmax(fx, axis=1) == np.argmax(y, axis=1)))

# Create tensorboard writer
writer = SummaryWriter(logdir=args.logdir)

# Train the network
global_step, running_count = 0, 0
running_loss, running_loss_g = 0., 0.
if args.save_path is not None:
    save_path = os.path.join(args.save_path, f'{global_step}.npy')
    save_jax_params(params, save_path)
    test_logits = onp.zeros((n_test, args.num_classes), dtype=onp.float32)
    test_loader = tfds.as_numpy(test_data.batch(args.batch_size_test))
    start_ind = 0
    for j, test_batch in enumerate(test_loader):
        test_batch = process_data(test_batch, split='test',
                                  centralize_y=args.centralize_y,
                                  num_classes=args.num_classes)
        X_test, Y_test = test_batch['image'], test_batch['label']
        fx = f(params, X_test)
        test_logits[start_ind:(start_ind + X_test.shape[0]), :] = fx
        start_ind += X_test.shape[0]
    save_path_logits = os.path.join(args.save_path, f'test_logits_{global_step}.npy')
    onp.save(save_path_logits, test_logits)


seed = args.init_seed
for epoch in range(args.epochs):
    print(f"Epoch {epoch}, seed={seed}")
    train_loader = tfds.as_numpy(train_data.shuffle(n_train, seed=seed).batch(args.batch_size_train))
    X, Y = None, None
    for i, batch in enumerate(train_loader):
        batch = process_data(batch, flatten=args.flatten, centralize_y=args.centralize_y,
                             split='train' if args.data_augment else 'test', seed=seed*args.seed_separator+i,
                             num_classes=args.num_classes)
        X, Y = batch['image'], batch['label']
        params_curr = get_params(state)
        loss_curr, grad_curr = value_and_grad_loss(params_curr, X, Y)
        # monitor gradient norm
        grad_norm = optimizers.l2_norm(grad_curr)
        writer.add_scalar(f'grad_norm/{tb_flag}', grad_norm.item(), global_step)
        if np.isnan(loss_curr):
            sys.exit()
        running_loss += loss_curr
        running_count += 1
        if args.grad_norm_thresh > 0:
            grad_curr = optimizers.clip_grads(grad_curr, args.grad_norm_thresh)
        state = opt_apply(epoch, grad_curr, state)
        global_step += 1
        print(f"Step {global_step}, training loss={loss_curr:.4f}, grad norm={grad_norm:.4f}")

        # Evaluate on the test set
        if global_step % args.save_steps == 0 \
                or global_step % args.early_save_steps == 0 and global_step <= args.early_save_till_step:
            test_loader = tfds.as_numpy(test_data.batch(args.batch_size_test))
            acc_f, loss_test = 0., 0.
            acc_g, loss_test_g = 0., 0.
            params_curr = get_params(state)
            start_ind = 0
            for j, test_batch in enumerate(test_loader):
                test_batch = process_data(test_batch, split='test',
                                          centralize_y=args.centralize_y,
                                          num_classes=args.num_classes)
                X_test, Y_test = test_batch['image'], test_batch['label']
                fx = f(params_curr, X_test)
                if args.save_path is not None:
                    test_logits[start_ind:(start_ind + X_test.shape[0]), :] = fx
                    start_ind += X_test.shape[0]
                loss_test += loss(fx, Y_test) * X_test.shape[0]
                acc_f += accuracy_sum(fx, Y_test)
            acc_f = acc_f / n_test
            loss_test = loss_test / n_test

            # Print and log results
            loss_train = running_loss / running_count
            running_loss, running_loss_g, running_count = 0., 0., 0
            writer.add_scalar(f'loss/train/{tb_flag}', loss_train.item(), global_step)
            writer.add_scalar(f'loss/test/{tb_flag}', loss_test.item(), global_step)
            writer.add_scalar(f'accuracy/test/{tb_flag}', acc_f.item(), global_step)
            print(f'Step {global_step}\n'
                  f'train_loss_{tb_flag}={loss_train:.4f}, '
                  f'test_loss_{tb_flag}={loss_test:.4f}, '
                  f'test_acc_{tb_flag}={acc_f:.4f}')

            # Compute distance between params and params_lin, relative to params_0
            dists_nn, rel_dists_nn = param_dist(params_curr, params_0)
            for j in range(len(dists_nn)):
                writer.add_scalar(f'dist/absolute/{j}', dists_nn[j].item(), global_step)
                writer.add_scalar(f'dist/relative/{j}', rel_dists_nn[j].item(), global_step)

            # Save params and test logits to file
            if args.save_path is not None:
                save_path = os.path.join(args.save_path, f'{global_step}.npy')
                save_jax_params(params_curr, save_path)
                save_path_logits = os.path.join(args.save_path, f'test_logits_{global_step}.npy')
                onp.save(save_path_logits, test_logits)

    if args.save_at_end_epoch:
        save_path = os.path.join(args.save_path, f'epoch_{epoch}.npy')
        save_jax_params(params_curr, save_path)
    seed += 1
writer.close()
