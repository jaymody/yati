import jax
import jax.numpy as jnp


def xavier_init(key, shape, gain=1.0):
    # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
    assert len(shape) == 2
    a = gain * jnp.sqrt(6.0 / (shape[0] + shape[1]))
    return a * jax.random.normal(key, shape)


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


def final_linear_layer(x, W1, b1):
    return softmax(x @ W1 + b1)


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


def encoder_layer():
    # X -> (seq_len, d_model)

    # ---- MultiHead Attention ----
    # out1 = multihead_attention(X, X, X)
    # out2 = norm(X + out1)

    # ---- Feed Forward ----
    # out3 = feed_forward_network(out2)
    # out4 = norm(out2 + out3)

    # output -> (seq_len, d_model)
    pass


def decoder_layer():
    # X -> (seq_len, d_model)
    # Z -> (seq_len, d_model) which is the encoder outputs

    # ---- Masked MultiHead Attention ----
    # TODO: figure out how masking is implemented (should be done at sdp atn layer)
    # out1 = multihead_attention(X, X, X)
    # out2 = norm(X + out1)

    # ---- MultiHead Attention ----
    # out3 = multihead_attention(out2, Z, Z)
    # out4 = norm(X + out2)

    # ---- Feed Forward ----
    # out5 = feed_forward_network(out4)
    # out6 = norm(out4 + out5)

    # output -> (seq_len, d_model)
    pass


def create_feed_forward_network_weights(key, d_model, d_ff):
    key, W1_subkey = jax.random.split(key)
    key, b1_subkey = jax.random.split(key)
    key, W2_subkey = jax.random.split(key)
    key, b2_subkey = jax.random.split(key)

    return key, {
        "W1": xavier_init(W1_subkey, (d_model, d_ff)),
        "b1": xavier_init(b1_subkey, (d_ff,)),
        "W2": xavier_init(W2_subkey, (d_ff, d_model)),
        "b2": xavier_init(b2_subkey, (d_model,)),
    }


def create_final_linear_layer_weights(key, d_model, n_out):
    key, W_subkey = key.split(key)
    key, b_subkey = key.split(key)
    return key, {
        "W": xavier_init(W_subkey, (d_model, n_out)),
        "b": xavier_init(b_subkey, (n_out,)),
    }


def create_mutlihead_attention_weights(key, d_model, d_k, d_v, h):
    key, *WQ_subkeys = jax.random.split(key, h)
    key, *WK_subkeys = jax.random.split(key, h)
    key, *WV_subkeys = jax.random.split(key, h)
    key, WO_subkey = jax.random.split(key)

    return key, {
        "WQ": jnp.stack([xavier_init(subkey, (d_model, d_k)) for subkey in WQ_subkeys]),
        "WK": jnp.stack([xavier_init(subkey, (d_model, d_k)) for subkey in WK_subkeys]),
        "WV": jnp.stack([xavier_init(subkey, (d_model, d_v)) for subkey in WV_subkeys]),
        "WO": xavier_init(WO_subkey, (h * d_v, d_model)),
    }


def create_encoder_layer(key, d_model, d_k, d_v, d_ff, h):
    key, multihead_attention_weights = create_mutlihead_attention_weights(
        key, d_model, d_k, d_v, h
    )
    key, feed_forward_network_weights = create_feed_forward_network_weights(
        key, d_model, d_ff
    )
    return key, {
        "multihead_attention_weights": multihead_attention_weights,
        "feed_forward_network_weights": feed_forward_network_weights,
    }


def create_decoder_layer(key, d_model, d_k, d_v, d_ff, h):
    key, multihead_attention_weights = create_mutlihead_attention_weights(
        key, d_model, d_k, d_v, h
    )
    key, masked_multihead_attention_weights = create_mutlihead_attention_weights(
        key, d_model, d_k, d_v, h
    )
    key, feed_forward_network_weights = create_feed_forward_network_weights(
        key, d_model, d_ff
    )
    return key, {
        "multihead_attention_weights": multihead_attention_weights,
        "masked_multihead_attention_weights": masked_multihead_attention_weights,
        "feed_forward_network_weights": feed_forward_network_weights,
    }


def create_transformer_weights(
    key, d_model, d_k, d_v, d_ff, h, n_enc_layers, n_dec_layers, n_out
):
    encoder_stack = []
    for _ in range(n_enc_layers):
        key, layer = create_encoder_layer(key, d_model, d_k, d_v, d_ff, h)
        encoder_stack.append(layer)

    decoder_stack = []
    for _ in range(n_dec_layers):
        key, layer = create_decoder_layer(key, d_model, d_k, d_v, d_ff, h)
        decoder_stack.append(layer)

    key, final_linear_layer_weights = create_final_linear_layer_weights(
        key, d_model, n_out
    )
    return key, {
        "encoder_stack": encoder_stack,
        "decoder_stack": decoder_stack,
        "final_linear_layer_weights": final_linear_layer_weights,
    }
