import jax
import jax.numpy as jnp


def xavier_init(key, shape, gain=1.0):
    # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
    assert len(shape) == 2
    a = gain * jnp.sqrt(6.0 / (shape[0] + shape[1]))
    return a * jax.random.normal(key, shape)


def relu(x):
    # x -> (d_model)
    # output -> (d_model)
    return jnp.max(0, x)


def softmax(x):
    # x -> (d_model)
    # output -> (d_model)
    return jnp.exp(x) / jnp.sum(jnp.exp(x))


def layer_norm(x, gamma, beta, eps=1e-8):
    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    return gamma * (x - jnp.mean(x)) / (jnp.std(x) + eps) + beta


def positional_embedding(pos, d_model, dtype=jnp.float32):
    # TODO: do we start indexing at 0 or 1, I'm assuming it's implied as 1 by the paper
    # since we are using mathematical notation (not that it will make a difference
    # anyways, but it does change the result of the equation slightly)
    odd_indices = jnp.arange(2, d_model + 1, 2)
    even_indices = jnp.arange(1, d_model + 1, 2)

    embedding = jnp.empty((d_model,), dtype=dtype)
    embedding = embedding.at[odd_indices - 1].set(
        jnp.cos(pos / jnp.power(10000, 2 * odd_indices / d_model))
    )
    embedding = embedding.at[even_indices - 1].set(
        jnp.sin(pos / jnp.power(10000, 2 * even_indices / d_model))
    )

    return embedding


def position_wise_ffn(x, W1, b1, W2, b2):
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
    # out3 = position_wise_ffn(out2)
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
    # out5 = position_wise_ffn(out4)
    # out6 = norm(out4 + out5)

    # output -> (seq_len, d_model)
    pass


def create_layer_norm_params():
    return {"gamma": jnp.array(1), "beta": jnp.array(0)}


def create_position_wise_ffn(key, d_model, d_ff):
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
    key, position_wise_ffn = create_position_wise_ffn(key, d_model, d_ff)
    return key, {
        "multihead_attention_weights": multihead_attention_weights,
        "layer_norm1": create_layer_norm_params(),
        "position_wise_ffn": position_wise_ffn,
        "layer_norm2": create_layer_norm_params(),
    }


def create_decoder_layer(key, d_model, d_k, d_v, d_ff, h):
    key, multihead_attention_weights = create_mutlihead_attention_weights(
        key, d_model, d_k, d_v, h
    )
    key, masked_multihead_attention_weights = create_mutlihead_attention_weights(
        key, d_model, d_k, d_v, h
    )
    key, position_wise_ffn = create_position_wise_ffn(key, d_model, d_ff)
    return key, {
        "multihead_attention_weights": multihead_attention_weights,
        "layer_norm1": create_layer_norm_params(),
        "masked_multihead_attention_weights": masked_multihead_attention_weights,
        "layer_norm2": create_layer_norm_params(),
        "position_wise_ffn": position_wise_ffn,
        "layer_norm3": create_layer_norm_params(),
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
