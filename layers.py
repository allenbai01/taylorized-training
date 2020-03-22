from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import lax

import jax.numpy as np

from neural_tangents import stax
from neural_tangents.stax import _same_pad_for_filter_shape, _randn
import jax.experimental.stax as jax_stax

import enum

class Padding(enum.Enum):
    CIRCULAR = 'CIRCULAR'
    SAME = 'SAME'
    VALID = 'VALID'

_CONV_DIMENSION_NUMBERS = ('NHWC', 'HWIO', 'NHWC')

def TaylorConv(out_chan,
               filter_shape,
               strides=None,
               padding=Padding.VALID.name,
               W_std=1.0,
               W_init=_randn(1.0),
               b_std=0.0,
               b_init=_randn(1.0),
               order=2):
    """Layer construction function for a convolution layer with Taylorized parameterization.
    Based on `jax.experimental.stax.GeneralConv`. Has a similar API apart from:
    Args:
        padding: in addition to `VALID` and `SAME' padding, supports `CIRCULAR`, not
        available in `jax.experimental.stax.GeneralConv`.
    """
    assert(isinstance(order, int) and order >= 1)
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    lhs_spec, rhs_spec, out_spec = dimension_numbers

    one = (1,) * len(filter_shape)
    strides = strides or one

    padding = Padding(padding)
    init_padding = padding
    if padding == Padding.CIRCULAR:
        init_padding = Padding.SAME

    def input_total_dim(input_shape):
        return input_shape[lhs_spec.index('C')] * np.prod(filter_shape)

    ntk_init_fn, _ = jax_stax.GeneralConv(dimension_numbers, out_chan, filter_shape,
                                          strides, init_padding.name, W_init, b_init)

    def taylor_init_fn(rng, input_shape):
        output_shape, (W, b) = ntk_init_fn(rng, input_shape)
        norm = W_std / (input_total_dim(input_shape) ** ((order-1)/(2*order+2)))
        return output_shape, (W * norm, b * b_std)

    def apply_fn(params, inputs, **kwargs):
        W, b = params
        norm = W_std / (input_total_dim(inputs.shape) ** (1/(order+1)))
        b_rescale = b_std

        apply_padding = padding
        if padding == Padding.CIRCULAR:
            apply_padding = Padding.VALID
            non_spatial_axes = (dimension_numbers[0].index('N'),
                                dimension_numbers[0].index('C'))
            spatial_axes = tuple(i for i in range(inputs.ndim)
                                 if i not in non_spatial_axes)
            inputs = _same_pad_for_filter_shape(inputs, filter_shape, strides,
                                                spatial_axes, 'wrap')

        return norm * lax.conv_general_dilated(
            inputs,
            W,
            strides,
            apply_padding.name,
            dimension_numbers=dimension_numbers) + b_rescale * b

    return taylor_init_fn, apply_fn


def MyConv(*args, parameterization='standard', order=None, **kwargs):
    """Wrapper for convolutional layer with different parameterizations."""
    if parameterization == 'standard':
        return jax_stax.Conv(*args, **kwargs)
    elif parameterization == 'ntk':
        return stax.Conv(*args, b_std=1.0, **kwargs)[:2]
    elif parameterization == 'taylor':
        return TaylorConv(*args, b_std=1.0, order=order, **kwargs)


def TaylorDense(out_dim,
                W_std=1.,
                b_std=0.,
                W_init=_randn(1.0),
                b_init=_randn(1.0),
                order=2):
    assert (isinstance(order, int) and order >= 1)

    ntk_init_fn, _ = jax_stax.Dense(out_dim, W_init, b_init)

    def taylor_init_fn(rng, input_shape):
        output_shape, (W, b) = ntk_init_fn(rng, input_shape)
        norm = W_std / (input_shape[-1] ** ((order-1)/(2*order+2)))
        return output_shape, (W * norm, b * b_std)

    def apply_fn(params, inputs, **kwargs):
        W, b = params
        norm = W_std / (inputs.shape[-1] ** (1/(order+1)))
        return norm * np.dot(inputs, W) + b_std * b

    return taylor_init_fn, apply_fn


def MyDense(*args, parameterization='standard', order=None, **kwargs):
    """Wrapper for dense layer with different parameterizations."""
    if parameterization == 'standard':
        return jax_stax.Dense(*args, **kwargs)
    elif parameterization == 'ntk':
        return stax.Dense(*args, b_std=1.0, **kwargs)[:2]
    elif parameterization == 'taylor':
        return TaylorDense(*args, b_std=1.0, order=order, **kwargs)