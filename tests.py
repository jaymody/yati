import jax
import jax.numpy as jnp

from model import (
    initialize_transformer_params,
    transformer_forward_fn,
    transformer_predict_fn,
)


def test_initialization_does_not_reuse_keys():
    """Tests that we didn't reuse any keys during initialization."""
    from inspect import getfullargspec, getmembers, isfunction

    used_keys = set()

    def check_key_is_not_reused_wrapper(func):
        argspec = getfullargspec(func)
        if "key" not in argspec.args:
            return func
        key_arg_index = argspec.args.index("key")

        def wrapper(*args, **kwargs):
            if len(args) > key_arg_index:  # check if key is in args
                key = args[key_arg_index]
            else:  # otherwise it should be in kwargs
                key = kwargs["key"]

            key = tuple(key.tolist())  # turn key into a hashable type
            assert key not in used_keys, f"key {key} is reused by {func.__name__}"
            used_keys.add(key)

            return func(*args, **kwargs)

        return wrapper

    for name, func in getmembers(jax.random, isfunction):
        setattr(jax.random, name, check_key_is_not_reused_wrapper(func))

    initialize_transformer_params(
        seed=123,
        src_vocab_size=200,
        trg_vocab_size=300,
        d_model=512,
        d_ff=2048,
        h=8,
        n_enc_layers=6,
        n_dec_layers=6,
    )


def test_forward_fn_and_predict_fn():
    params = initialize_transformer_params(
        seed=123,
        src_vocab_size=200,
        trg_vocab_size=300,
        d_model=512,
        d_ff=2048,
        h=8,
        n_enc_layers=6,
        n_dec_layers=6,
    )

    logits = transformer_forward_fn(
        src_token_ids=jnp.array([11, 10, 90, 7, 101]),
        trg_token_ids=jnp.array([254, 6, 10, 40, 105, 10, 43]),
        **params,
        pad_idx=0,
    )
    print(logits.shape)

    logits = transformer_predict_fn(
        src_token_ids=jnp.array([11, 8, 90, 5, 101, 99]),
        **params,
        sos_idx=1,
        max_sequence_length=10,
        pad_idx=0,
    )
    print(jnp.argmax(logits, axis=-1))
