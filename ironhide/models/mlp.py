from jax.scipy.special import logsumexp
from jax.nn import relu
from functools import partial
from jax import grad, jit, vmap
import jax.numpy as jnp
import jax
from jax.tree_util import register_pytree_node_class
from ironhide.utilities.helpers import batchify
import jaxopt


@register_pytree_node_class
class ToyMLP:
    def __init__(self, parameters=None, layer_sizes=None):
        self.parameters = parameters
        self.layer_sizes = layer_sizes

        if (not self.parameters) and self.layer_sizes:
            self.initialize_params()

    def initialize_params(self):
        initializer = jax.nn.initializers.normal(0.01)
        self.parameters = [
            [initializer(jax.random.PRNGKey(42), (n, m), dtype=jnp.float32),
             initializer(jax.random.PRNGKey(42), (n,), dtype=jnp.float32)]
            for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]

    @partial(vmap, in_axes=(None, None, 0))
    def forward(self, parameters, x):
        activations = x
        for w, b in parameters[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = relu(outputs)

        final_w, final_b = parameters[-1]
        logits = jnp.dot(final_w, activations) + final_b
        return logits - logsumexp(logits)

    @jit
    def cost_func(self, params, x, y):
        preds = self.forward(params, x)
        return -jnp.mean(preds * y)

    @jit
    def update(self, x, y, step_size):
        grads = grad(self.cost_func)(self.parameters, x, y)
        return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(self.parameters, grads)]

    def predict(self, x):
        return jnp.argmax(self.forward(self.parameters, x), axis=1)

    def fit(self, train_x, train_y, num_epochs=50, step_size=0.05, use_jaxopt=False):
        if use_jaxopt:
            solver = jaxopt.GradientDescent(fun=self.cost_func, stepsize=step_size)
            params, state = solver.run(self.parameters, x=train_x, y=train_y)
            self.parameters = params
        else:
            for epoch in range(num_epochs):
                for x, y in batchify(train_x, train_y):
                    self.parameters = self.update(x, y, step_size)

    def tree_flatten(self):
        children = (self.parameters, self.layer_sizes)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)