import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
import jax
from jax.nn import relu
from functools import partial
from jax.scipy.special import logsumexp
import time
from ironhide.data.mnist import num_pixels, train_images, train_labels, test_images, test_labels
from ironhide.utilities.helpers import accuracy, one_hot


layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 10
batch_size = 128
n_targets = 10
num_labels = 10


@partial(vmap, in_axes=(None, 0))
def forward(parameters, x):
    activations = x
    for w, b in parameters[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = parameters[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


@jit
def cost_func(parameters, x, y):
    preds = forward(parameters, x)
    return -jnp.mean(preds * y)


@jit
def update(parameters, x, y):
    grads = grad(cost_func)(parameters, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(parameters, grads)]


global parameters
parameters = [
    (np.random.standard_normal((n, m)) * 1 / (n ** 0.5), np.random.standard_normal((n,)) * 1 / (n ** 0.5))
    for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
]

class MiniMLP:
    def __init__(self, layer_sizes):
        pass

    def predict(self, x):
        return jnp.argmax(forward(parameters, x), axis=1)

    def fit(self, data_stream, val_x, val_y):
        for epoch in range(num_epochs):
            start_time = time.time()
            for x, y in data_stream:
                x = jnp.reshape(x, (len(x), num_pixels))
                y = one_hot(y, num_labels)
                parameters = update(parameters, x, y)
            epoch_time = time.time() - start_time

            if val_x is not None and val_y is not None:
                test_acc = accuracy(self.predict(val_x), val_y)
                print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
                print("Validation set accuracy {}".format(test_acc))