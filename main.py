import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)


def interleave_two_arrays(a, b):
    # https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    c = jnp.empty((a.size, b.size), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c


def relu(x):
    # x -> (d_model)
    # output -> (d_model)
    return jnp.max(0, x)


def softmax(x):
    # x -> (d_model)
    # output -> (d_model)
    return jnp.exp(x) / jnp.sum(jnp.exp(x))


def layer_norm(a):
    # https://arxiv.org/pdf/1607.06450.pdf
    u = jnp.mean(a)
    o = jnp.sqrt(jnp.mean((a - u) ** 2))
    return o


def positional_embeddings(pos, d_model):
    # TODO: do we start indexing at 0 or 1, I'm assuming it's implied as 1 by the paper
    # since we are using mathematical notation (not that it will make a difference
    # anyways)
    odd_indices = jnp.arange(2, d_model + 1, 2)
    even_indices = jnp.arange(1, d_model + 1, 2)

    odd_values = jnp.cos(pos / jnp.power(10000, 2 * odd_indices / d_model))
    even_values = jnp.sin(pos / jnp.power(10000, 2 * even_indices / d_model))

    return interleave_two_arrays(odd_values, even_values)


def feed_forward_network(x, W1, b1, W2, b2):
    # X -> (d_model)
    # W1 -> (d_model, d_ff)
    # b1 -> (d_ff)
    # W2 -> (d_ff, d_model)
    # b1 -> (d_model)

    # output -> (d_model)
    return relu(x @ W1 + b1) @ W2 + b2


def scaled_dot_product_attention(Q, K, V):
    # Q -> (seq_len, d_k)
    # K -> (seq_len, d_k)
    # V -> (seq_len, d_v)

    # output -> (seq_len, d_v)
    d_k = K.shape[-1]
    return softmax(Q @ K.T / jnp.sqrt(d_k)) @ V


def multihead_attention(Q, K, V, WQ, WK, WV, WO):
    # Q -> (seq_len, d_k)
    # K -> (seq_len, d_k)
    # V -> (seq_len, d_v)

    # WQi -> (h, d_model, d_k)
    # WKi -> (h, d_model, d_k)
    # WVi -> (h, d_model, d_v)
    # WO  -> (h * d_v, d_model)

    # h = number of attentions heads
    # output -> (seq_len, d_model)

    h = WQ.shape[0]
    heads = []
    for i in range(h):
        heads.append(
            scaled_dot_product_attention(Q=Q @ WQ[i], K=K @ WK[i], V=V @ WV[i])
        )
    return jnp.concatenate(heads) @ WO
