###
# References
# Google Vanilla RNN:
# https://github.com/google-research/computation-thru-dynamics/blob/master/integrator_rnn_tutorial/rnn.py
# https://github.com/google-research/computation-thru-dynamics/blob/master/notebooks/Integrator%20RNN%20Tutorial.ipynb
# https://calvinfeng.gitbook.io/machine-learning-notebook/supervised-learning/recurrent-neural-network/recurrent_neural_networks


###


# References
# https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html
from jax.scipy.special import logsumexp
from jax.nn import relu
from functools import partial
from jax import grad, jit, vmap
import jax.numpy as jnp
import jax
from ironhide.utilities.helpers import accuracy, one_hot
from ironhide.data.mnist import get_train_batches, num_pixels, train_images, train_labels, test_images, test_labels
import time
import numpy as np



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
def update(parameters, x, y, step_size=0.025):
    grads = grad(cost_func)(parameters, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(parameters, grads)]

num_epochs=10
num_labels=10

class MiniMLP:
    def __init__(self, layer_sizes):
        initializer = jax.nn.initializers.normal(1.0)
        self.parameters = [
            (np.random.standard_normal((n, m)) * 1 / (n ** 0.5), np.random.standard_normal((n,)) * 1 / (n ** 0.5))
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
        ]


    def predict(self, x):
        predicted_class = jnp.argmax(forward(self.parameters, x), axis=1)
        return predicted_class

    def fit(self):
        for epoch in range(num_epochs):
            start_time = time.time()
            for x, y in get_train_batches():
                print(y.sum())
                x = jnp.reshape(x, (len(x), num_pixels))
                y = one_hot(y, num_labels)
                self.parameters = update(self.parameters, x, y)
            epoch_time = time.time() - start_time

            train_acc = accuracy(self.predict(train_images), train_labels)
            test_acc = accuracy(self.predict(test_images), test_labels)
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))