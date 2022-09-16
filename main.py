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


def create_embedding_lookup_table(key, n, d):
    # n -> number of embeddings to create
    # d -> dimension of each embedding
    return jax.random.normal(key, (n, d))


def create_pad_mask(x, pad_idx):
    # x -> (seq_len)
    # output -> (seq_len, seq_len), positions that are True will be masked out
    return (x == pad_idx).reshape(1, -1) | (x == pad_idx).reshape(-1, 1)


def create_illegal_connections_mask(seq_len):
    # output -> (seq_len, seq_len), positions that are True will be masked out
    return ~jnp.tri(seq_len, seq_len, k=0, dtype=jnp.bool_)


def create_mask(x, pad_idx):
    # x -> (seq_len)
    # output -> (seq_len, seq_len), positions that are True will be masked out
    return create_pad_mask(x, pad_idx) | create_illegal_connections_mask(x.shape[0])


def layer_norm(x, gamma, beta, eps=1e-8):
    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    return gamma * (x - jnp.mean(x)) / (jnp.std(x) + eps) + beta


def embedding_lookup(token_indices, embedding_lookup_table):
    return embedding_lookup_table[token_indices]


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


def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q -> (seq_len, d_k)
    # K -> (seq_len, d_k)
    # V -> (seq_len, d_v)
    # mask -> (seq_len, seq_len)
    #   mask[i][j] = True means mask this connection (illegal connection)
    #   mask[i][j] = False means don't mask this connection (valid connection)
    #   mask = None means no masking (in other words, every connection is valid)

    # output -> (seq_len, d_v)
    assert mask.dtype == jnp.bool_

    d_k = K.shape[-1]
    mask = 0 if mask is None else mask * -jnp.inf
    return softmax((Q @ K.T / jnp.sqrt(d_k)) + mask) @ V


def multihead_attention(Q, K, V, WQ, WK, WV, WO, mask):
    # Q -> (seq_len, d_k)
    # K -> (seq_len, d_k)
    # V -> (seq_len, d_v)
    # mask -> (seq_len, seq_len)

    # WQi -> (h, d_model, d_k)
    # WKi -> (h, d_model, d_k)
    # WVi -> (h, d_model, d_v)
    # WO  -> (h * d_v, d_model)

    # h = number of attentions heads
    # output -> (seq_len, d_model)

    # TODO: don't use a for loop here, you can probably implement this via vectorized
    # functions
    h = WQ.shape[0]
    heads = []
    for i in range(h):
        heads.append(
            scaled_dot_product_attention(
                Q=Q @ WQ[i], K=K @ WK[i], V=V @ WV[i], mask=mask
            )
        )
    return jnp.concatenate(heads) @ WO


def encoder_layer(
    X,
    src_mask,
    multihead_attention_weights,
    layer_norm1_params,
    position_wise_ffn_weights,
    layer_norm2_params,
):
    # X -> (seq_len, d_model)
    # output -> (seq_len, d_model)

    # multihead attention
    prev = X
    out = multihead_attention(
        Q=X, K=X, V=X, mask=src_mask, **multihead_attention_weights
    )
    out = layer_norm(prev + out, **layer_norm1_params)

    # position wise ffn
    prev = out
    out = position_wise_ffn(out, **position_wise_ffn_weights)
    out = layer_norm(prev + out, **layer_norm2_params)

    return out


def decoder_layer(
    X,
    Z,
    trg_mask,
    src_mask,
    masked_multihead_attention_weights,
    layer_norm1_params,
    multihead_attention_weights,
    layer_norm2_params,
    position_wise_ffn_weights,
    layer_norm3_params,
):
    # X -> (seq_len, d_model)
    # Z -> (seq_len, d_model) which is the encoder outputs
    # trg_mask -> (seq_len, seq_len) which is the mask for the first attn block
    # src_mask -> (seq_len, seq_len) which is the mask for the second attn block
    # output -> (seq_len, d_model)

    # masked multihead attention
    prev = X
    out = multihead_attention(
        Q=X, K=X, V=X, mask=trg_mask, **masked_multihead_attention_weights
    )
    out = layer_norm(prev + out, **layer_norm1_params)

    # multihead attention
    prev = out
    out = multihead_attention(
        Q=out, K=Z, V=Z, mask=src_mask, **multihead_attention_weights
    )
    out = layer_norm(prev + out, **layer_norm2_params)

    # position wise ffn
    prev = out
    out = position_wise_ffn(out, **position_wise_ffn_weights)
    out = layer_norm(prev + out, **layer_norm3_params)

    return out


def create_layer_norm_params():
    return {"gamma": jnp.array(1), "beta": jnp.array(0)}


def create_position_wise_ffn_weights(key, d_model, d_ff):
    W1_subkey, W2_subkey = jax.random.split(key)

    return {
        "W1": xavier_init(W1_subkey, (d_model, d_ff)),
        "b1": jnp.zeros((d_ff,)),
        "W2": xavier_init(W2_subkey, (d_ff, d_model)),
        "b2": jnp.zeros((d_model,)),
    }


def create_final_linear_layer_weights(key, d_model, n_out):
    return {
        "W": xavier_init(key, (d_model, n_out)),
        "b": jnp.zeros((n_out,)),
    }


def create_mutlihead_attention_weights(key, d_model, d_k, d_v, h):
    key, *WQ_subkeys = jax.random.split(key, h + 1)
    key, *WK_subkeys = jax.random.split(key, h + 1)
    key, *WV_subkeys = jax.random.split(key, h + 1)
    WO_subkey = key

    return {
        "WQ": jnp.stack([xavier_init(subkey, (d_model, d_k)) for subkey in WQ_subkeys]),
        "WK": jnp.stack([xavier_init(subkey, (d_model, d_k)) for subkey in WK_subkeys]),
        "WV": jnp.stack([xavier_init(subkey, (d_model, d_v)) for subkey in WV_subkeys]),
        "WO": xavier_init(WO_subkey, (h * d_v, d_model)),
    }


def create_encoder_layer(key, d_model, d_k, d_v, d_ff, h):
    subkeys = jax.random.split(key, 2)

    multihead_attention_weights = create_mutlihead_attention_weights(
        subkeys[0], d_model, d_k, d_v, h
    )
    position_wise_ffn_weights = create_position_wise_ffn_weights(
        subkeys[1], d_model, d_ff
    )
    return {
        "multihead_attention_weights": multihead_attention_weights,
        "layer_norm1_params": create_layer_norm_params(),
        "position_wise_ffn_weights": position_wise_ffn_weights,
        "layer_norm2_params": create_layer_norm_params(),
    }


def create_decoder_layer(key, d_model, d_k, d_v, d_ff, h):
    subkeys = jax.random.split(key, 3)

    multihead_attention_weights = create_mutlihead_attention_weights(
        subkeys[0], d_model, d_k, d_v, h
    )
    masked_multihead_attention_weights = create_mutlihead_attention_weights(
        subkeys[1], d_model, d_k, d_v, h
    )
    position_wise_ffn_weights = create_position_wise_ffn_weights(
        subkeys[2], d_model, d_ff
    )
    return key, {
        "masked_multihead_attention_weights": masked_multihead_attention_weights,
        "layer_norm1_params": create_layer_norm_params(),
        "multihead_attention_weights": multihead_attention_weights,
        "layer_norm2_params": create_layer_norm_params(),
        "position_wise_ffn_weights": position_wise_ffn_weights,
        "layer_norm3_params": create_layer_norm_params(),
    }


def create_transformer_weights(
    seed, d_model, d_k, d_v, d_ff, h, n_enc_layers, n_dec_layers, n_out
):
    key = jax.random.PRNGKey(seed)
    key, *enc_keys = jax.random.split(key, n_enc_layers + 1)
    key, *dec_keys = jax.random.split(key, n_dec_layers + 1)
    final_layer_key = key

    encoder_stack = [
        create_encoder_layer(enc_keys[i], d_model, d_k, d_v, d_ff, h)
        for i in range(n_enc_layers)
    ]

    decoder_stack = [
        create_decoder_layer(dec_keys[i], d_model, d_k, d_v, d_ff, h)
        for i in range(n_dec_layers)
    ]

    final_linear_layer_weights = create_final_linear_layer_weights(
        final_layer_key, d_model, n_out
    )
    return {
        "encoder_stack": encoder_stack,
        "decoder_stack": decoder_stack,
        "final_linear_layer_weights": final_linear_layer_weights,
    }
