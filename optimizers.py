import jax.experimental.optimizers as jax_opt
import jax.numpy as np
import copy


@jax_opt.optimizer
def sgd(step_size):
    step_size = jax_opt.make_schedule(step_size)

    def init(x0):
        return copy.deepcopy(x0)

    def update(i, g, x):
        return x - step_size(i) * g

    def get_params(x):
        return x

    return init, update, get_params


@jax_opt.optimizer
def momentum(step_size, mass, weight_decay=0.):
    step_size = jax_opt.make_schedule(step_size)

    def init(x0):
        v0 = np.zeros_like(x0)
        return x0, v0

    def update(i, g, state):
        x, velocity = state
        if weight_decay != 0.:
            g = g + weight_decay * x
        velocity = mass * velocity + g
        x = x - step_size(i) * velocity
        return x, velocity

    def get_params(state):
        x, _ = state
        return x

    return init, update, get_params