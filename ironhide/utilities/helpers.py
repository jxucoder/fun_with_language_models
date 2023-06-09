import jax.numpy as jnp
import numpy as np


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(preds, targets):
    target_class = jnp.argmax(targets, axis=1)
    return jnp.mean(preds == target_class)


def batchify(train_x, train_y, batch_size=256):
    num_samples = len(train_x)
    num_batches = num_samples // batch_size

    batched_train_x = np.array_split(train_x[:num_batches * batch_size], num_batches)
    batched_train_y = np.array_split(train_y[:num_batches * batch_size], num_batches)

    return zip(batched_train_x, batched_train_y)


def get_batch_with_seq_length(data, seq_length):
    for i in range(0, len(data[0]) - seq_length, seq_length):
        inputs = data[:, i: i + seq_length]
        targets = data[:, (i + 1): (i + 1) + seq_length]
        yield (inputs, targets)