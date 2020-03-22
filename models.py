from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax, Softplus)

from jax.nn import sigmoid, swish

from neural_tangents import stax
import jax.experimental.stax as jax_stax
from layers import MyConv, MyDense

Swish = jax_stax.elementwise(swish)
def swish_ten(x):
    return x * sigmoid(10 * x)
Swishten = jax_stax.elementwise(swish_ten)


def CNNStandard(n_channels, L, filter=(3, 3), data='cifar10',
                gap=True,
                nonlinearity='relu',
                parameterization='standard', order=None):
    if data == 'cifar10':
        num_classes = 10
    if data == 'cifar100':
        num_classes = 100
    if nonlinearity == 'relu':
        nonlin = Relu
    elif nonlinearity == 'swish':
        nonlin = Swish
    init_fn, f = jax_stax.serial(
        *[jax_stax.serial(
            MyConv(n_channels, filter,
                   parameterization=parameterization, order=order),
            nonlin,
          ) for _ in range(L)]
    )
    if gap:
        init_fn, f = jax_stax.serial(
            (init_fn, f),
            stax.GlobalAvgPool()[:2],
            MyDense(num_classes,
                    parameterization=parameterization, order=order)
        )
    else:
        init_fn, f = jax_stax.serial(
            (init_fn, f),
            stax.Flatten()[:2],
            MyDense(num_classes,
                    parameterization=parameterization, order=order)
        )
    return init_fn, f


def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False,
                    nonlin=Relu,
                    parameterization='standard', order=None):
    Main = jax_stax.serial(
        nonlin,
        MyConv(channels, (3, 3), strides, padding='SAME',
               parameterization=parameterization, order=order),
        nonlin,
        MyConv(channels, (3, 3), padding='SAME',
               parameterization=parameterization, order=order)
    )
    Shortcut = Identity if not channel_mismatch else MyConv(
        channels, (3, 3), strides, padding='SAME',
        parameterization=parameterization, order=order)
    return jax_stax.serial(FanOut(2),
                           jax_stax.parallel(Main, Shortcut),
                           FanInSum)


def WideResnetGroup(n, channels, strides=(1, 1),
                    nonlin=Relu,
                    parameterization='standard', order=None):
    blocks = []
    blocks += [WideResnetBlock(channels, strides, channel_mismatch=True,
                               nonlin=nonlin,
                               parameterization=parameterization, order=order)]
    for _ in range(n - 1):
        blocks += [WideResnetBlock(channels, (1, 1),
                                   nonlin=nonlin,
                                   parameterization=parameterization, order=order)]
    return jax_stax.serial(*blocks)


def WideResnet(block_size, k, num_classes,
               channels=1024,
               nonlinearity='relu',
               parameterization='standard', order=None):
    if nonlinearity == 'relu':
        nonlin = Relu
    elif nonlinearity == 'swish':
        nonlin = Swish
    return jax_stax.serial(
        MyConv(channels, (3, 3), padding='SAME',
               parameterization=parameterization, order=order),
        WideResnetGroup(block_size, int(16 * k),
                        nonlin=nonlin,
                        parameterization=parameterization, order=order),
        WideResnetGroup(block_size, int(32 * k), (2, 2),
                        nonlin=nonlin,
                        parameterization=parameterization, order=order),
        WideResnetGroup(block_size, int(64 * k), (2, 2),
                        nonlin=nonlin,
                        parameterization=parameterization, order=order),
        AvgPool((8, 8)),
        Flatten,
        MyDense(num_classes,
                parameterization=parameterization, order=order)
    )


def ConvBlock(kernel_size, filters, strides=(2, 2),
              batchnorm=True, parameterization='standard', nonlin=Relu):
    ks = kernel_size
    filters1, filters2, filters3 = filters
    if parameterization == 'standard':
        def MyConv(*args, **kwargs):
            return Conv(*args, **kwargs)
    elif parameterization == 'ntk':
        def MyConv(*args, **kwargs):
            return stax.Conv(*args, **kwargs)[:2]
    if batchnorm:
        Main = jax_stax.serial(
            MyConv(filters1, (1, 1), strides),
            BatchNorm(), nonlin,
            MyConv(filters2, (ks, ks), padding='SAME'),
            BatchNorm(), nonlin,
            MyConv(filters3, (1, 1)),
            BatchNorm()
        )
        Shortcut = jax_stax.serial(
            MyConv(filters3, (1, 1), strides),
            BatchNorm()
        )
    else:
        Main = jax_stax.serial(
            MyConv(filters1, (1, 1), strides),
            nonlin,
            MyConv(filters2, (ks, ks), padding='SAME'),
            nonlin,
            MyConv(filters3, (1, 1))
        )
        Shortcut = jax_stax.serial(
            MyConv(filters3, (1, 1), strides)
        )
    return jax_stax.serial(FanOut(2), jax_stax.parallel(Main, Shortcut), FanInSum, nonlin)


def IdentityBlock(kernel_size, filters, batchnorm=True,
                  parameterization='standard', nonlin=Relu):
    ks = kernel_size
    filters1, filters2 = filters
    if parameterization == 'standard':
        def MyConv(*args, **kwargs):
            return Conv(*args, **kwargs)
    elif parameterization == 'ntk':
        def MyConv(*args, **kwargs):
            return stax.Conv(*args, **kwargs)[:2]
    def make_main(input_shape):
        # the number of output channels depends on the number of input channels
        if batchnorm:
            return jax_stax.serial(
                MyConv(filters1, (1, 1)),
                BatchNorm(), nonlin,
                MyConv(filters2, (ks, ks), padding='SAME'),
                BatchNorm(), nonlin,
                MyConv(input_shape[3], (1, 1)),
                BatchNorm()
            )
        else:
            return jax_stax.serial(
                MyConv(filters1, (1, 1)), nonlin,
                MyConv(filters2, (ks, ks), padding='SAME'), nonlin,
                MyConv(input_shape[3], (1, 1))
            )
    Main = jax_stax.shape_dependent(make_main)
    return jax_stax.serial(FanOut(2), jax_stax.parallel(Main, Identity), FanInSum, nonlin)


# ResNet architectures compose layers and ResNet blocks
def ResNet50(num_classes, batchnorm=True,
             parameterization='standard', nonlinearity='relu'):
    # Define layer constructors
    if parameterization == 'standard':
        def MyGeneralConv(*args, **kwargs):
            return GeneralConv(*args, **kwargs)
        def MyDense(*args, **kwargs):
            return Dense(*args, **kwargs)
    elif parameterization == 'ntk':
        def MyGeneralConv(*args, **kwargs):
            return stax._GeneralConv(*args, **kwargs)[:2]
        def MyDense(*args, **kwargs):
            return stax.Dense(*args, **kwargs)[:2]
    # Define nonlinearity
    if nonlinearity == 'relu':
        nonlin = Relu
    elif nonlinearity == 'swish':
        nonlin = Swish
    elif nonlinearity == 'swishten':
        nonlin = Swishten
    elif nonlinearity == 'softplus':
        nonlin = Softplus
    return jax_stax.serial(
        MyGeneralConv(('NHWC', 'HWIO', 'NHWC'), 64, (7, 7), strides=(2, 2), padding='SAME'),
        BatchNorm() if batchnorm else Identity,
        nonlin, MaxPool((3, 3), strides=(2, 2)),
        ConvBlock(3, [64, 64, 256], strides=(1, 1), batchnorm=batchnorm,
                  parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [64, 64], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [64, 64], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        ConvBlock(3, [128, 128, 512], batchnorm=batchnorm,
                  parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [128, 128], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [128, 128], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [128, 128], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        ConvBlock(3, [256, 256, 1024], batchnorm=batchnorm,
                  parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [256, 256], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [256, 256], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [256, 256], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [256, 256], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [256, 256], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        ConvBlock(3, [512, 512, 2048], batchnorm=batchnorm,
                  parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [512, 512], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        IdentityBlock(3, [512, 512], batchnorm=batchnorm,
                      parameterization=parameterization, nonlin=nonlin),
        stax.GlobalAvgPool()[:-1],
        MyDense(num_classes)
    )
