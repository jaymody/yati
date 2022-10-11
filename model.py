import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PyTree, UInt

PRNGKeyType = UInt[Array, "2"]


################################
#### Positional Embeddings #####
################################
def create_positional_embeddings(
    seq_len: int, d_model: int
) -> Float[Array, "seq_len d_model"]:
    def create_positional_embedding(d_model, pos) -> Float[Array, "d_model"]:
        # NOTE: we start indexing 2i and pos from 0 instead of from 1, this shouldn't
        # make a difference (just shifts the curves one dim over)

        odd_indices = jnp.arange(1, d_model, 2)
        even_indices = jnp.arange(0, d_model, 2)

        positional_embedding = jnp.empty((d_model,))
        positional_embedding = positional_embedding.at[odd_indices].set(
            jnp.cos(pos / jnp.power(10000, even_indices / d_model))
        )
        positional_embedding = positional_embedding.at[even_indices].set(
            jnp.sin(pos / jnp.power(10000, even_indices / d_model))
        )

        return positional_embedding

    return jax.lax.map(
        lambda pos: create_positional_embedding(d_model, pos), jnp.arange(seq_len)
    )


################################
######### Basic Layers #########
################################
def layer_norm(
    x: Float[Array, "... d_model"],
    gamma: Float[Array, "d_model"],
    beta: Float[Array, "d_model"],
    eps: float = 1e-8,
) -> Float[Array, "... d_model"]:
    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    # [..., None] is a fancy trick to reshape the array
    # such that the last dimension is 1 for broadcasting
    numerator = x - jnp.mean(x, axis=-1)[..., None]
    denominator = jnp.sqrt(jnp.var(x, axis=-1)[..., None] + eps)
    return gamma * (numerator / denominator) + beta


def embedding_lookup(
    token_indices: Int[Array, "seq_len"],
    embedding_lookup_table: Float[Array, "vocab_size d_model"],
) -> Float[Array, "seq_len d_model"]:
    return embedding_lookup_table[token_indices]


def position_wise_ffn(
    X: Float[Array, "seq_len d_model"],
    W1: Float[Array, "d_model d_ff"],
    W2: Float[Array, "d_ff d_model"],
    b1: Float[Array, "d_ff"],
    b2: Float[Array, "d_model"],
) -> Float[Array, "seq_len d_model"]:
    return jax.nn.relu(X @ W1 + b1) @ W2 + b2


def final_linear_layer(
    X: Float[Array, "seq_len d_model"],
    final_linear_layer_matrix: Float[Array, "d_model vocab_size"],
) -> Float[Array, "seq_len vocab_size"]:
    # we skip the softmax for a couple reasons:
    #   1. when taking the argmax for predictions, it makes no difference as softmax is
    #      a monotonic function
    #   2. when computing cross entropy loss, taking jnp.log(jnp.softmax(x)) is
    #      numerically instable compared to jax.nn.log_softmax
    return X @ final_linear_layer_matrix


def scaled_dot_product_attention(
    Q: Float[Array, "trg_seq_len d_k"],
    K: Float[Array, "src_seq_len d_k"],
    V: Float[Array, "src_seq_len d_v"],
    mask: Float[Array, "trg_seq_len src_seq_len"],
) -> Float[Array, "trg_seq_len d_v"]:
    d_k = K.shape[-1]
    return jax.nn.softmax((Q @ K.T / jnp.sqrt(d_k)) + mask) @ V


def multihead_attention(
    Q: Float[Array, "trg_seq_len d_model"],
    K: Float[Array, "src_seq_len d_model"],
    V: Float[Array, "src_seq_len d_model"],
    WQ: Float[Array, "h d_model d_k"],
    WK: Float[Array, "h d_model d_k"],
    WV: Float[Array, "h d_model d_v"],
    WO: Float[Array, "d_model d_model"],  # TODO: should be (h x d_v,  d_model)
    mask: Float[Array, "trg_seq_len src_seq_len"],
) -> Float[Array, "trg_seq_len d_model"]:
    Q = Q @ WQ  # (trg_seq_len, d_model) @ (h, d_model, d_k) -> (h, trg_seq_len, d_k)
    K = K @ WK  # (src_seq_len, d_model) @ (h, d_model, d_k) -> (h, src_seq_len, d_k)
    V = V @ WV  # (src_seq_len, d_model) @ (h, d_model, d_v) -> (h, src_seq_len, d_v)

    # heads -> (h, trg_seq_len, d_v)
    heads = jax.vmap(scaled_dot_product_attention, (0, 0, 0, None), 0)(Q, K, V, mask)

    return jnp.hstack(heads) @ WO


################################
### Parameter Initialization ###
################################
def xavier_uniform(key: PRNGKeyType, shape: tuple[int, ...], gain: float = 1.0):
    # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
    assert len(shape) == 2
    a = gain * jnp.sqrt(6.0 / (shape[0] + shape[1]))
    return jax.random.uniform(key, shape, minval=-a, maxval=a)


def initialize_layer_norm_params(d_model: int):
    return {"gamma": jnp.ones((d_model,)), "beta": jnp.zeros((d_model,))}


def initialize_position_wise_ffn_params(key: PRNGKeyType, d_model: int, d_ff: int):
    W1_subkey, W2_subkey = jax.random.split(key)

    return {
        "W1": xavier_uniform(W1_subkey, (d_model, d_ff)),
        "b1": jnp.zeros((d_ff,)),
        "W2": xavier_uniform(W2_subkey, (d_ff, d_model)),
        "b2": jnp.zeros((d_model,)),
    }


def initialize_mutlihead_attention_params(
    key: PRNGKeyType, d_model: int, d_k: int, d_v: int, h: int
):
    key, *WQ_subkeys = jax.random.split(key, h + 1)
    key, *WK_subkeys = jax.random.split(key, h + 1)
    key, *WV_subkeys = jax.random.split(key, h + 1)
    WO_subkey = key

    return {
        "WQ": jnp.stack(
            [xavier_uniform(subkey, (d_model, d_k)) for subkey in WQ_subkeys]
        ),
        "WK": jnp.stack(
            [xavier_uniform(subkey, (d_model, d_k)) for subkey in WK_subkeys]
        ),
        "WV": jnp.stack(
            [xavier_uniform(subkey, (d_model, d_v)) for subkey in WV_subkeys]
        ),
        "WO": xavier_uniform(WO_subkey, (h * d_v, d_model)),
    }


def initialize_encoder_layer(key: PRNGKeyType, d_model: int, d_ff: int, h: int):
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


def initialize_decoder_layer(key: PRNGKeyType, d_model: int, d_ff: int, h: int):
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
    seed: int,
    src_vocab_size: int,
    trg_vocab_size: int,
    d_model: int,
    d_ff: int,
    h: int,
    n_enc_layers: int,
    n_dec_layers: int,
):
    key = jax.random.PRNGKey(seed)
    key, src_embedding_key = jax.random.split(key)
    key, trg_embedding_key = jax.random.split(key)
    key, *enc_keys = jax.random.split(key, n_enc_layers + 1)
    key, *dec_keys = jax.random.split(key, n_dec_layers + 1)
    final_layer_key = key

    src_embeddings_table = jax.random.normal(
        src_embedding_key, (src_vocab_size, d_model)
    )

    trg_embeddings_table = jax.random.normal(
        trg_embedding_key, (trg_vocab_size, d_model)
    )

    encoder_stack = [
        initialize_encoder_layer(enc_keys[i], d_model, d_ff, h)
        for i in range(n_enc_layers)
    ]

    decoder_stack = [
        initialize_decoder_layer(dec_keys[i], d_model, d_ff, h)
        for i in range(n_dec_layers)
    ]

    final_linear_layer_matrix = xavier_uniform(
        final_layer_key, (d_model, trg_vocab_size)
    )

    return {
        "src_embeddings_table": src_embeddings_table,
        "trg_embeddings_table": trg_embeddings_table,
        "encoder_stack": encoder_stack,
        "decoder_stack": decoder_stack,
        "final_linear_layer_matrix": final_linear_layer_matrix,
    }


def initialize_transformer_params_with_shared_weight_matrix(
    seed: int,
    vocab_size: int,
    d_model: int,
    d_ff: int,
    h: int,
    n_enc_layers: int,
    n_dec_layers: int,
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
def sublayer_add_and_norm(x, sublayer_fn, layer_norm_params):
    # TODO: in the original paper they do LayerNorm(x + Sublayer(x))
    # but in the actual implementation in tensor2tensor (and other implementations) use
    # x + Sublayer(LayerNorm(x)) (they LayerNorm the input rather than the output)
    # see: http://disq.us/p/1s2bpmf
    return x + sublayer_fn(layer_norm(x, **layer_norm_params))


def encoder_layer(
    X: Float[Array, "src_seq_len d_model"],
    src_mask: Float[Array, "src_seq_len src_seq_len"],
    multihead_attention_params: PyTree,
    layer_norm1_params: PyTree,
    position_wise_ffn_params: PyTree,
    layer_norm2_params: PyTree,
) -> Float[Array, "src_seq_len d_model"]:
    # multihead attention
    out = sublayer_add_and_norm(
        x=X,
        sublayer_fn=lambda x: multihead_attention(
            Q=x, K=x, V=x, mask=src_mask, **multihead_attention_params
        ),
        layer_norm_params=layer_norm1_params,
    )

    # position wise ffn
    out = sublayer_add_and_norm(
        x=out,
        sublayer_fn=lambda x: position_wise_ffn(x, **position_wise_ffn_params),
        layer_norm_params=layer_norm2_params,
    )

    return out


def decoder_layer(
    X: Float[Array, "trg_seq_len d_model"],
    Z: Float[Array, "src_seq_len d_model"],
    trg_mask: Float[Array, "trg_seq_len trg_seq_len"],
    src_mask: Float[Array, "trc_seq_len src_seq_len"],
    masked_multihead_attention_params: PyTree,
    layer_norm1_params: PyTree,
    multihead_attention_params: PyTree,
    layer_norm2_params: PyTree,
    position_wise_ffn_params: PyTree,
    layer_norm3_params: PyTree,
) -> Float[Array, "trg_seq_len d_model"]:
    # masked multihead attention
    out = sublayer_add_and_norm(
        x=X,
        sublayer_fn=lambda x: multihead_attention(
            Q=x, K=x, V=x, mask=trg_mask, **masked_multihead_attention_params
        ),
        layer_norm_params=layer_norm1_params,
    )

    # multihead attention
    out = sublayer_add_and_norm(
        x=out,
        sublayer_fn=lambda x: multihead_attention(
            Q=x, K=Z, V=Z, mask=src_mask, **multihead_attention_params
        ),
        layer_norm_params=layer_norm2_params,
    )

    # position wise ffn
    out = sublayer_add_and_norm(
        x=out,
        sublayer_fn=lambda x: position_wise_ffn(x, **position_wise_ffn_params),
        layer_norm_params=layer_norm3_params,
    )

    return out


################################
###### Masking Functions #######
################################
def create_pad_mask(
    x: Int[Array, "seq_len"],
    nrows: int,
    pad_idx: int,
) -> Bool[Array, "nrows seq_len"]:
    # positions that are True are to be masked out
    return jnp.repeat((x == pad_idx).reshape(1, -1), nrows, axis=0)


def create_illegal_connections_mask(seq_len: int) -> Bool[Array, "seq_len seq_len"]:
    # positions that are True are to be masked out
    return ~jnp.tri(seq_len, seq_len, k=0, dtype=jnp.bool_)


def create_masks(
    src_token_ids: Int[Array, "src_seq_len"],
    trg_token_ids: Int[Array, "trg_seq_len"],
    pad_idx: int,
    eps: float = -1.0e9,
) -> tuple[
    Float[Array, "src_seq_len src_seq_len"],
    Float[Array, "trg_seq_len src_seq_len"],
    Float[Array, "trg_seq_len trg_seq_len"],
]:
    src_seq_len = src_token_ids.shape[0]
    trg_seq_len = trg_token_ids.shape[0]

    encoder_src_mask = create_pad_mask(src_token_ids, src_seq_len, pad_idx) * eps
    decoder_src_mask = create_pad_mask(src_token_ids, trg_seq_len, pad_idx) * eps
    decoder_trg_mask = (
        create_pad_mask(trg_token_ids, trg_seq_len, pad_idx)
        | create_illegal_connections_mask(trg_seq_len)
    ) * eps
    return encoder_src_mask, decoder_src_mask, decoder_trg_mask


################################
######### Transformer ##########
################################
def encoder(
    src_token_ids: Int[Array, "src_seq_len"],
    src_mask: Float[Array, "src_seq_len src_seq_len"],
    src_embeddings_table: Float[Array, "src_vocab_size d_model"],
    encoder_stack: PyTree,
) -> Float[Array, "src_seq_len d_model"]:
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
    trg_token_ids: Int[Array, "trg_seq_len"],
    Z: Float[Array, "src_seq_len d_model"],
    src_mask: Float[Array, "trg_seq_len src_seq_len"],
    trg_mask: Float[Array, "trg_seq_len trg_seq_len"],
    trg_embeddings_table: Float[Array, "trg_vocab_size d_model"],
    decoder_stack: PyTree,
    final_linear_layer_matrix: Float[Array, "d_model trg_vocab_size"],
) -> Float[Array, "trg_seq_len trg_vocab_size"]:
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
    logits = final_linear_layer(X, final_linear_layer_matrix)

    # (trg_seq_len, trg_vocab_size)
    return logits


def transformer_forward_fn(
    src_token_ids: Int[Array, "src_seq_len"],
    trg_token_ids: Int[Array, "trg_seq_len"],
    src_embeddings_table: Float[Array, "src_vocab_size d_model"],
    trg_embeddings_table: Float[Array, "trg_vocab_size d_model"],
    encoder_stack: PyTree,
    decoder_stack: PyTree,
    final_linear_layer_matrix: Float[Array, "d_model trg_vocab_size"],
    pad_idx: int,
) -> Float[Array, "trg_seq_len trg_vocab_size"]:
    # TODO: maybe the caller should be responsible for creating the masks?
    encoder_src_mask, decoder_src_mask, decoder_trg_mask = create_masks(
        src_token_ids, trg_token_ids, pad_idx
    )
    Z = encoder(
        src_token_ids=src_token_ids,
        src_mask=encoder_src_mask,
        src_embeddings_table=src_embeddings_table,
        encoder_stack=encoder_stack,
    )
    logits = decoder(
        trg_token_ids=trg_token_ids,
        Z=Z,
        src_mask=decoder_src_mask,
        trg_mask=decoder_trg_mask,
        trg_embeddings_table=trg_embeddings_table,
        decoder_stack=decoder_stack,
        final_linear_layer_matrix=final_linear_layer_matrix,
    )
    return logits


def transformer_predict_fn(
    src_token_ids: Int[Array, "src_seq_len"],
    max_sequence_length: int,
    src_embeddings_table: Float[Array, "src_vocab_size d_model"],
    trg_embeddings_table: Float[Array, "trg_vocab_size d_model"],
    encoder_stack: PyTree,
    decoder_stack: PyTree,
    final_linear_layer_matrix: Float[Array, "d_model trg_vocab_size"],
    sos_idx: int,
    pad_idx: int,
) -> Float[Array, "max_sequence_length trg_vocab_size"]:
    # encoder forward pass
    encoder_src_mask, _, _ = create_masks(src_token_ids, src_token_ids, pad_idx)
    Z = encoder(
        src_token_ids=src_token_ids,
        src_mask=encoder_src_mask,
        src_embeddings_table=src_embeddings_table,
        encoder_stack=encoder_stack,
    )

    # iterative decode
    # start with just [sos_idx] as the decoder input, each step add the token with the
    # highest predicted next token probability to the decoder input and repeat until a
    # sequence of length max_sequence_length is constructed
    trg_vocab_size = trg_embeddings_table.shape[0]
    trg_token_ids = jnp.ones(max_sequence_length, dtype=jnp.int32) * pad_idx
    trg_token_ids = trg_token_ids.at[0].set(sos_idx)
    output_probabilities = jnp.empty((max_sequence_length, trg_vocab_size))
    for i in range(max_sequence_length):
        # decoder forward pass
        _, decoder_src_mask, decoder_trg_mask = create_masks(
            src_token_ids, trg_token_ids, pad_idx
        )
        logits = decoder(
            trg_token_ids=trg_token_ids,
            Z=Z,
            src_mask=decoder_src_mask,
            trg_mask=decoder_trg_mask,
            trg_embeddings_table=trg_embeddings_table,
            decoder_stack=decoder_stack,
            final_linear_layer_matrix=final_linear_layer_matrix,
        )

        # next_token_probability_distribution[i] = probability the next token is i
        next_token_probability_distribution = jax.nn.softmax(logits[-1])

        # save probabilities for the function return
        output_probabilities = output_probabilities.at[i].set(
            next_token_probability_distribution
        )

        # token idx of the token that has the highest probability of being next
        predicted_next_token_idx = jnp.argmax(next_token_probability_distribution)

        # append the predict next token to the decoder input for the next iteration
        trg_token_ids = trg_token_ids.at[i + 1].set(predicted_next_token_idx)

        # TODO: figure how to end early if predicted_next_token_idx == eos_idx
        # in a way that works with jit

    return logits
