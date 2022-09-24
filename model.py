import jax
import jax.numpy as jnp


################################
##### Activation Functions #####
################################
def relu(x):
    # x -> any shape
    # output -> same shape as x
    return jnp.maximum(0, x)


def softmax(x):
    # # x -> (d_model)
    # # output -> (d_model)

    # we can write softmax as: jnp.exp(x) / jnp.sum(jnp.exp(x))
    # however with large enough inputs the exponent will become very large
    # and cause numerical instability (we will start to get nan values and such)
    #
    # so we use a more numerically stable version of softmax, where we shift the input
    # by the max value such that the max value is now 0
    #
    # see Practical Issues section of https://cs231n.github.io/linear-classify/#softmax-classifier
    x = x - jnp.max(x)  # make our calculation numerically stable
    return jnp.exp(x) / jnp.sum(jnp.exp(x))


################################
#### Positional Embeddings #####
################################
def create_positional_embeddings(seq_len: int, d_model: int):
    def create_positional_embedding(d_model: int, pos: int):
        # TODO: do we start indexing at 0 or 1, I'm assuming it's implied as 1 by the paper
        # since we are using mathematical notation (not that it will make a difference
        # anyways, but it does change the result of the equation slightly)

        odd_indices = jnp.arange(2, d_model + 1, 2)
        even_indices = jnp.arange(1, d_model + 1, 2)

        positional_embedding = jnp.empty((d_model,))
        positional_embedding = positional_embedding.at[odd_indices - 1].set(
            jnp.cos(pos / jnp.power(10000, 2 * odd_indices / d_model))
        )
        positional_embedding = positional_embedding.at[even_indices - 1].set(
            jnp.sin(pos / jnp.power(10000, 2 * even_indices / d_model))
        )

        # output -> (d_model)
        return positional_embedding

    # output -> (seq_len, d_model)
    return jax.lax.map(
        lambda pos: create_positional_embedding(d_model, pos), jnp.arange(seq_len)
    )


################################
######### Basic Layers #########
################################
def layer_norm(x, gamma, beta, eps: float = 1e-8):
    # x -> (d_model)
    # gamma -> (d_model)
    # beta -> (d_model)
    # output -> same shape as x
    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    return gamma * (x - jnp.mean(x)) / (jnp.std(x) + eps) + beta


def embedding_lookup(token_indices, embedding_lookup_table):
    # token_indices -> (seq_len) of type int
    # embedding_lookup_table -> (vocab_size, d_model)
    return embedding_lookup_table[token_indices]


def position_wise_ffn(X, W1, b1, W2, b2):
    # X -> (seq_len, d_model)
    # W1 -> (d_model, d_ff)
    # b1 -> (d_ff)
    # W2 -> (d_ff, d_model)
    # b1 -> (d_model)

    # output -> (seq_len, d_model)

    def ffn(x, W1, b1, W2, b2):
        # x -> (d_model)
        # output -> (d_model)
        return relu(x @ W1 + b1) @ W2 + b2

    return jax.vmap(ffn, (0, None, None, None, None), 0)(X, W1, b1, W2, b2)


def final_linear_layer(X, final_linear_layer_matrix):
    # X -> (trg_seq_len, d_model)
    # W -> (d_model, trg_vocab_size)
    # b -> (trg_vocab_size)

    # output -> (trg_seq_len, trg_vocab_size)

    def ffn(x, final_linear_layer_matrix):
        # x -> (d_model)
        # output -> (trg_vocab_size)
        return softmax(x @ final_linear_layer_matrix)

    return jax.vmap(ffn, (0, None), 0)(X, final_linear_layer_matrix)


def scaled_dot_product_attention(Q, K, V, mask):
    # Q -> (in_seq_len, d_k)
    # K -> (out_seq_len, d_k)
    # V -> (out_seq_len, d_v)
    # mask -> (in_seq_len, out_seq_len)
    #   mask[i][j] = -jpn.inf means mask this connection (illegal connection)
    #   mask[i][j] = 0 means don't mask this connection (valid connection)

    # output -> (out_seq_len, d_v)

    d_k = K.shape[-1]
    return softmax((Q @ K.T / jnp.sqrt(d_k)) + mask) @ V


def multihead_attention(Q, K, V, WQ, WK, WV, WO, mask):
    # Q -> (in_seq_len, d_model)
    # K -> (out_seq_len, d_model)
    # V -> (out_seq_len, d_model)
    # mask -> (in_seq_len, out_seq_len)

    # WQ -> (h, d_model, d_k)
    # WK -> (h, d_model, d_k)
    # WV -> (h, d_model, d_v)
    # WO -> (h * d_v, d_model)

    # h = number of attentions heads
    # output -> (out_seq_len, d_model)

    Q = Q @ WQ  # (in_seq_len, d_model) @ (h, d_model, d_k) -> (h, in_seq_len, d_k)
    K = K @ WK  # (in_seq_len, d_model) @ (h, d_model, d_k) -> (h, in_seq_len, d_k)
    V = V @ WV  # (in_seq_len, d_model) @ (h, d_model, d_v) -> (h, in_seq_len, d_v)

    # heads -> (h, out_seq_len, d_v)
    heads = jax.vmap(scaled_dot_product_attention, (0, 0, 0, None), 0)(Q, K, V, mask)

    return jnp.hstack(heads) @ WO


################################
### Parameter Initialization ###
################################
def xavier_init(key, shape, gain: float = 1.0):
    # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
    assert len(shape) == 2
    a = gain * jnp.sqrt(6.0 / (shape[0] + shape[1]))
    return a * jax.random.normal(key, shape)


def initialize_layer_norm_params(d_model: int):
    return {"gamma": jnp.ones((d_model,)), "beta": jnp.zeros((d_model,))}


def initialize_position_wise_ffn_params(key, d_model: int, d_ff: int):
    W1_subkey, W2_subkey = jax.random.split(key)

    return {
        "W1": xavier_init(W1_subkey, (d_model, d_ff)),
        "b1": jnp.zeros((d_ff,)),
        "W2": xavier_init(W2_subkey, (d_ff, d_model)),
        "b2": jnp.zeros((d_model,)),
    }


def initialize_mutlihead_attention_params(
    key, d_model: int, d_k: int, d_v: int, h: int
):
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


def initialize_encoder_layer(key, d_model, d_ff, h):
    subkeys = jax.random.split(key, 2)

    multihead_attention_params = initialize_mutlihead_attention_params(
        subkeys[0],
        d_model=d_model,
        d_k=d_model // h,
        d_v=d_model // h,
        h=h,
    )
    position_wise_ffn_params = initialize_position_wise_ffn_params(
        subkeys[1],
        d_model,
        d_ff,
    )
    return {
        "multihead_attention_params": multihead_attention_params,
        "layer_norm1_params": initialize_layer_norm_params(d_model),
        "position_wise_ffn_params": position_wise_ffn_params,
        "layer_norm2_params": initialize_layer_norm_params(d_model),
    }


def initialize_decoder_layer(key, d_model, d_ff, h):
    subkeys = jax.random.split(key, 3)

    multihead_attention_params = initialize_mutlihead_attention_params(
        subkeys[0],
        d_model=d_model,
        d_k=d_model // h,
        d_v=d_model // h,
        h=h,
    )
    masked_multihead_attention_params = initialize_mutlihead_attention_params(
        subkeys[1],
        d_model=d_model,
        d_k=d_model // h,
        d_v=d_model // h,
        h=h,
    )
    position_wise_ffn_params = initialize_position_wise_ffn_params(
        subkeys[2],
        d_model,
        d_ff,
    )
    return {
        "masked_multihead_attention_params": masked_multihead_attention_params,
        "layer_norm1_params": initialize_layer_norm_params(d_model),
        "multihead_attention_params": multihead_attention_params,
        "layer_norm2_params": initialize_layer_norm_params(d_model),
        "position_wise_ffn_params": position_wise_ffn_params,
        "layer_norm3_params": initialize_layer_norm_params(d_model),
    }


def initialize_transformer_params(
    seed,
    src_vocab_size,
    trg_vocab_size,
    d_model,
    d_ff,
    h,
    n_enc_layers,
    n_dec_layers,
):
    key = jax.random.PRNGKey(seed)
    key, src_embedding_key = jax.random.split(key)
    key, trg_embedding_key = jax.random.split(key)
    key = jax.random.PRNGKey(seed)
    key, *enc_keys = jax.random.split(key, n_enc_layers + 1)
    key, *dec_keys = jax.random.split(key, n_dec_layers + 1)
    final_layer_key = key

    src_embeddings_table = xavier_init(src_embedding_key, (src_vocab_size, d_model))

    trg_embeddings_table = xavier_init(trg_embedding_key, (trg_vocab_size, d_model))

    encoder_stack = [
        initialize_encoder_layer(enc_keys[i], d_model, d_ff, h)
        for i in range(n_enc_layers)
    ]

    decoder_stack = [
        initialize_decoder_layer(dec_keys[i], d_model, d_ff, h)
        for i in range(n_dec_layers)
    ]

    final_linear_layer_matrix = xavier_init(final_layer_key, (d_model, trg_vocab_size))

    return {
        "src_embeddings_table": src_embeddings_table,
        "trg_embeddings_table": trg_embeddings_table,
        "encoder_stack": encoder_stack,
        "decoder_stack": decoder_stack,
        "final_linear_layer_matrix": final_linear_layer_matrix,
    }


def initialize_transformer_params_with_shared_weight_matrix(
    seed,
    vocab_size,
    d_model,
    d_ff,
    h,
    n_enc_layers,
    n_dec_layers,
):
    params_dict = initialize_transformer_params(
        seed=seed,
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        h=h,
        n_enc_layers=n_enc_layers,
        n_dec_layers=n_dec_layers,
    )
    del params_dict["trg_embeddings_table"]
    del params_dict["final_linear_layer_matrix"]
    params_dict["shared_weight_matrix"] = params_dict.pop("src_embeddings_table")
    return params_dict


################################
#### Encoder/Decoder Layers ####
################################
def encoder_layer(
    X,
    src_mask,
    multihead_attention_params,
    layer_norm1_params,
    position_wise_ffn_params,
    layer_norm2_params,
):
    # X -> (seq_len, d_model)
    # output -> (seq_len, d_model)

    # multihead attention
    prev = X
    out = multihead_attention(
        Q=X, K=X, V=X, mask=src_mask, **multihead_attention_params
    )
    out = layer_norm(prev + out, **layer_norm1_params)

    # position wise ffn
    prev = out
    out = position_wise_ffn(out, **position_wise_ffn_params)
    out = layer_norm(prev + out, **layer_norm2_params)

    return out


def decoder_layer(
    X,
    Z,
    trg_mask,
    src_mask,
    masked_multihead_attention_params,
    layer_norm1_params,
    multihead_attention_params,
    layer_norm2_params,
    position_wise_ffn_params,
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
        Q=X, K=X, V=X, mask=trg_mask, **masked_multihead_attention_params
    )
    out = layer_norm(prev + out, **layer_norm1_params)

    # multihead attention
    prev = out
    out = multihead_attention(
        Q=out, K=Z, V=Z, mask=src_mask, **multihead_attention_params
    )
    out = layer_norm(prev + out, **layer_norm2_params)

    # position wise ffn
    prev = out
    out = position_wise_ffn(out, **position_wise_ffn_params)
    out = layer_norm(prev + out, **layer_norm3_params)

    return out


################################
###### Masking Functions #######
################################
def create_pad_mask(x, pad_idx):
    # x -> (seq_len)
    # output -> (seq_len, seq_len), positions that are True are to be masked out
    return (x == pad_idx).reshape(1, -1) | (x == pad_idx).reshape(-1, 1)


def create_illegal_connections_mask(seq_len):
    # output -> (seq_len, seq_len), positions that are True are to be masked out
    return ~jnp.tri(seq_len, seq_len, k=0, dtype=jnp.bool_)


def create_mask(x, pad_idx):
    # x -> (seq_len)
    # output -> (seq_len, seq_len), positions that are True are to be masked out
    return create_pad_mask(x, pad_idx) | create_illegal_connections_mask(x.shape[0])


################################
######### Transformer ##########
################################
def encoder(src_token_ids, src_embeddings_table, encoder_stack):
    src_seq_len = src_token_ids.shape[0]

    # (src_seq_len, src_seq_len)
    # TODO: pad masking, also maybe this should only be done once between enc and dec?
    src_mask = jnp.zeros((src_seq_len, src_seq_len))

    # (src_seq_len) -> (src_seq_len, d_model)
    src_embeddings = embedding_lookup(src_token_ids, src_embeddings_table)

    # (src_seq_len, d_model) -> (src_seq_len, d_model)
    src_embeddings += create_positional_embeddings(
        src_embeddings.shape[0], src_embeddings.shape[1]
    )

    Z = src_embeddings
    for encoder_layer_params in encoder_stack:
        # (src_seq_len, d_model) -> (src_seq_len, d_model)
        Z = encoder_layer(Z, src_mask=src_mask, **encoder_layer_params)

    # (src_seq_len, d_model)
    return Z


def decoder(
    trg_token_ids, Z, trg_embeddings_table, decoder_stack, final_linear_layer_matrix
):
    src_seq_len = Z.shape[0]
    trg_seq_len = trg_token_ids.shape[0]

    # (src_seq_len, src_seq_len)
    # TODO: pad masking, also maybe this should only be done once between enc and dec?
    src_mask = jnp.zeros((trg_seq_len, src_seq_len))

    # (trg_seq_len, trg_seq_len)
    trg_mask = create_illegal_connections_mask(seq_len=trg_seq_len) * -jnp.inf

    # (trg_seq_len) -> (trg_seq_len, d_model)
    trg_embeddings = embedding_lookup(trg_token_ids, trg_embeddings_table)

    # (trg_seq_len, d_model) -> (trg_seq_len, d_model)
    trg_embeddings += create_positional_embeddings(
        trg_embeddings.shape[0], trg_embeddings.shape[1]
    )

    X = trg_embeddings
    for decoder_layer_params in decoder_stack:
        # (trg_seq_len, d_model) -> (trg_seq_len, d_model)
        X = decoder_layer(
            X, Z=Z, trg_mask=trg_mask, src_mask=src_mask, **decoder_layer_params
        )

    # (trg_seq_len, d_model) -> (trg_seq_len, trg_vocab_size)
    next_token_probabilities = final_linear_layer(X, final_linear_layer_matrix)

    # (trg_seq_len, trg_vocab_size)
    return next_token_probabilities


def transformer_forward_fn(
    src_token_ids,
    trg_token_ids,
    src_embeddings_table,
    trg_embeddings_table,
    encoder_stack,
    decoder_stack,
    final_linear_layer_matrix,
):
    Z = encoder(
        src_token_ids,
        src_embeddings_table,
        encoder_stack,
    )
    next_token_probabilities = decoder(
        trg_token_ids,
        Z,
        trg_embeddings_table,
        decoder_stack,
        final_linear_layer_matrix,
    )
    return next_token_probabilities


def transformer_predict_fn(
    src_token_ids,
    sos_idx,
    max_sequence_length,
    src_embeddings_table,
    trg_embeddings_table,
    encoder_stack,
    decoder_stack,
    final_linear_layer_matrix,
):
    # encoder forward pass
    Z = encoder(src_token_ids, src_embeddings_table, encoder_stack)

    # iterative decode
    # start with just [sos_idx] as the decoder input, each step add the token with the
    # highest predicted next token probability to the decoder input and repeat until a
    # sequence of length max_sequence_length is constructed
    trg_vocab_size = trg_embeddings_table.shape[0]
    trg_token_ids = jnp.array([sos_idx])
    logits = jnp.empty((max_sequence_length, trg_vocab_size))
    for i in range(max_sequence_length):
        # decoder forward pass
        probabilities = decoder(
            trg_token_ids,
            Z,
            trg_embeddings_table,
            decoder_stack,
            final_linear_layer_matrix,
        )

        # next_token_probability_distribution[i] = probability the next token is i
        next_token_probability_distribution = probabilities[-1]

        # token idx of the token that has the highest probability of being next
        predicted_next_token_idx = jnp.argmax(next_token_probability_distribution)

        # append probability dist to logits so we can return it to the caller
        logits = logits.at[i].set(next_token_probability_distribution)

        # append the predict next token to the decoder input for the next iteration
        trg_token_ids = jnp.concatenate(
            [trg_token_ids, jnp.array([predicted_next_token_idx])]
        )

    return logits
