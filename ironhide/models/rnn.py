import jax
import jax.numpy as jnp
from jax import vmap
import optax
from functools import partial
import equinox as eqx
from ironhide.utilities.helpers import get_batch_with_seq_length


cost_func = jax.vmap(optax.softmax_cross_entropy, in_axes=(0, 0))
make_sequence_embedding = jax.vmap(lambda w_embeds, one_hot_word: w_embeds @ one_hot_word, in_axes=(None, 0))
loss_and_gradient = eqx.filter_value_and_grad(lambda model, inputs, targets: cost_func(model(inputs), targets).mean(axis=0))
batched_loss_and_gradient = jax.vmap(loss_and_gradient, in_axes=(None, 0, 0))
filter_jit_batched_loss_and_gradient = eqx.filter_jit(batched_loss_and_gradient)


class ToyRNN(eqx.Module):
    w_embedding: jax.Array
    w_h_hprev: jax.Array
    w_h_embedding: jax.Array
    h_bias: jax.Array
    w_output_h: jax.Array
    output_bias: jax.Array
    vocab_size: int
    embed_size: int
    hidden_size: int

    def __init__(self, layer_sizes):
        self.vocab_size, self.embed_size, self.hidden_size = layer_sizes
        w_embeds_key, w_hh_key, w_hx_key, w_s_key = jax.random.split(jax.random.PRNGKey(42), num=4)

        initializer = jax.nn.initializers.variance_scaling(1,  distribution="truncated_normal", mode='fan_avg')

        self.w_embedding = initializer(shape=(self.embed_size, self.vocab_size), key=w_embeds_key)

        self.w_h_hprev = initializer(shape=(self.hidden_size, self.hidden_size), key=w_hh_key)
        self.w_h_embedding = initializer(shape=(self.hidden_size, self.embed_size), key=w_hx_key)
        self.h_bias = jnp.zeros((self.hidden_size,))

        self.w_output_h = initializer(shape=(self.vocab_size, self.hidden_size), key=w_s_key)
        self.output_bias = jnp.zeros((self.vocab_size,))

    def __call__(self, sent):
        embedding = make_sequence_embedding(self.w_embedding, sent)
        initial_h = jnp.zeros((self.hidden_size,))

        def f(h_prev, word_embedding):
            h_curr = jax.nn.tanh(
                self.w_h_hprev @ h_prev
                + self.w_h_embedding @ word_embedding
                + self.h_bias
            )
            curr_outputs = self.w_output_h @ h_curr + self.output_bias
            return h_curr, curr_outputs

        _, out = jax.lax.scan(f, initial_h, embedding)

        return out

    def fit(self, data_stream, lr=0.01, seq_length=20, max_iter=500):
        opt = optax.chain(
            optax.clip(1),
            optax.adamw(learning_rate=lr),
        )
        opt_state = opt.init(self)
        batch_one_hot_sentence = vmap(partial(jax.nn.one_hot, num_classes=self.vocab_size))

        training_losses = []
        for i, data in enumerate(get_batch_with_seq_length(data_stream, seq_length)):
            inputs, targets = data
            batched_inputs = batch_one_hot_sentence(jnp.array(inputs))
            batched_targets = batch_one_hot_sentence(jnp.array(targets))

            loss, grads = filter_jit_batched_loss_and_gradient(self, batched_inputs, batched_targets)
            avg_grads = jax.tree_map(lambda g: g.mean(axis=0), grads)

            updates, opt_state = opt.update(avg_grads, opt_state, params=self)
            self = eqx.apply_updates(self, updates)
            mean_loss = loss.mean()
            training_losses.append(mean_loss)
            print(mean_loss)
            if i >= max_iter:
                break
        return training_losses