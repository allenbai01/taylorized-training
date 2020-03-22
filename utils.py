from jax import numpy as np
from jax.api import jit
from jax.tree_util import tree_flatten, tree_unflatten
import tensorflow as tf

def param_dist(param_1, param_2, param_0=None):
    """
    Computes the layer-wise distance between two sets of Jax parameters.
    Return:
        dists: list of absolute distances
        rel_dists: list of relative (to param_0) distances
    """
    if param_0 == None:
        param_0 = param_2
    param_1_flat, _ = tree_flatten(param_1)
    param_2_flat, _ = tree_flatten(param_2)
    param_0_flat, _ = tree_flatten(param_0)
    dists = [np.linalg.norm(p1 - p2) for (p1, p2) in zip(param_1_flat, param_2_flat)]
    norms = [np.linalg.norm(p0) for p0 in param_0_flat]
    rel_dists = [dists[i] / norms[i] for i in range(len(dists))]
    return dists, rel_dists


def save_jax_params(params, save_path):
    """
    Save parameter values to a path.
    """
    params_flat, _ = tree_flatten(params)
    np.save(save_path, params_flat)


def load_jax_params(params, load_path):
    """
    Load parameter values from a path.
    """
    params_flat, params_tree = tree_flatten(params)
    params_load_flat = np.load(load_path, allow_pickle=True)
    params_load_flat = [np.asarray(p) for p in params_load_flat]
    # for (p, p_load) in zip(params_flat, params_load_flat):
    #     p *= 0.0
    #     p += p_load
    params = tree_unflatten(params_tree, params_load_flat)
    return params


def copy_jax_array(params):
    params_flat, params_tree = tree_flatten(params)
    params_new = [np.array(p) for p in params_flat]
    return tree_unflatten(params_tree, params_new)


def tf_cifar_augment_image(image, split='train', seed=None):
    """Image augmentation suitable for CIFAR-10/100.
    As described in https://arxiv.org/pdf/1608.06993v3.pdf (page 5).
    Args:
        image: a 4D tf Tensor of shape NxHxWxC
    Returns:
        Tensor of the same shape as image.
    """
    image = tf.cast(image, tf.float32) / 255.0
    if split == 'train':
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)
        image = tf.image.random_crop(image, [image.shape[0], 32, 32, 3],
                                     seed=seed if seed is not None else seed)
        image = tf.image.random_flip_left_right(image, seed=seed+1 if seed is not None else seed)
    image = (image - tf.reshape(tf.constant([0.4914, 0.4822, 0.4465]), [1, 1, 1, 3])) \
            / tf.reshape(tf.constant([0.2023, 0.1994, 0.2010]), [1, 1, 1, 3])
    return image


def process_data(data_chunk, flatten=False,
                 one_hot_y=True, centralize_y=False,
                 split='train', seed=None, num_classes=10):
    """Flatten the images and one-hot encode the labels."""
    image, label = data_chunk['image'], data_chunk['label']
    samples = image.shape[0]
    # pdb.set_trace()
    image = np.array(tf_cifar_augment_image(image, split=split, seed=seed).numpy(), dtype=np.float32)
    # if flatten:
    #     image = np.array(np.reshape(image, (samples, -1)), dtype=np.float32)
    # else:
    #     image = np.array(image, dtype=np.float32)
    # image = (image - np.mean(image)) / np.std(image)
    if one_hot_y:
        label = np.eye(num_classes)[label]
        if centralize_y:
            label -= np.mean(label, axis=1, keepdims=True)
    return {'image': image, 'label': label}


@jit
def saxpby_jax(a, x, b, y):
    x_flat, x_tree = tree_flatten(x)
    y_flat, _ = tree_flatten(y)
    return tree_unflatten(x_tree, [a * xx + b * yy for (xx, yy) in zip(x_flat, y_flat)])