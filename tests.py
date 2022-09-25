import inspect

import jax
import jax.numpy as jnp
import pytest

from model import (
    initialize_transformer_params,
    transformer_forward_fn,
    transformer_predict_fn,
)


class BlockJaxKeyReuse:
    """Context manager that throws an error if a key is reused by a jax.random function.

    Example Usage:

    ```python
    # throws an error
    with BlockJaxKeyReuse():
        key = jax.random.PRNGKey(123)
        a = jax.random.normal(key, (2, 5))
        b = jax.random.normal(key, (2, 5))  # throws an error here, key was already used
        print(a + b)
    ```

    ```python
    # passes
    with BlockJaxKeyReuse():
        key = jax.random.PRNGKey(123)
        key, a_subkey, b_subkey = jax.random.split(key, 3)
        a = jax.random.normal(a_subkey, (2, 5))
        b = jax.random.normal(b_subkey, (2, 5))
        print(a + b)
    ```
    """

    def __init__(self):
        self.jax_random_functions = {
            name: func
            for name, func in inspect.getmembers(jax.random, inspect.isfunction)
        }
        self.used_keys = set()

    def add_key_reuse_check(self, func):
        argspec = inspect.getfullargspec(func)
        if "key" not in argspec.args:
            return func
        key_arg_index = argspec.args.index("key")

        def wrapper(*args, **kwargs):
            if len(args) > key_arg_index:  # check if key is in args
                key = args[key_arg_index]
            else:  # otherwise it should be in kwargs
                key = kwargs["key"]

            key = tuple(key.tolist())  # turn key into a hashable type
            assert key not in self.used_keys, f"key {key} was already been used"
            self.used_keys.add(key)

            return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        # add wrappers to all jax.random functions
        for name, func in self.jax_random_functions.items():
            setattr(jax.random, name, self.add_key_reuse_check(func))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore original jax.random functions
        for name, func in self.jax_random_functions.items():
            setattr(jax.random, name, func)


def test_BlockJaxKeyReuse():
    with pytest.raises(AssertionError):
        with BlockJaxKeyReuse():
            key = jax.random.PRNGKey(123)
            a = jax.random.normal(key, (2, 5))
            b = jax.random.normal(key, (2, 5))

    with pytest.raises(AssertionError):
        with BlockJaxKeyReuse():
            key = jax.random.PRNGKey(123)
            key, subkey = jax.random.normal(key, (2, 5))
            key = jax.random.PRNGKey(123)
            key, subkey = jax.random.normal(key, (2, 5))

    with pytest.raises(AssertionError):
        with BlockJaxKeyReuse():
            key = jax.random.PRNGKey(123)
            key, a_subkey, b_subkey = jax.random.split(key, 3)
            a = jax.random.normal(key, (2, 5))
            b = jax.random.normal(a_subkey, (2, 5))
            c = jax.random.normal(b_subkey, (2, 5))
            key, subkey = jax.random.split(key)

    # the following tests should not throw any errors
    with BlockJaxKeyReuse():
        key = jax.random.PRNGKey(123)
        key, a_subkey, b_subkey = jax.random.split(key, 3)
        a = jax.random.normal(a_subkey, (2, 5))
        b = jax.random.normal(b_subkey, (2, 5))

    with BlockJaxKeyReuse():
        key = jax.random.PRNGKey(123)
        key, subkey = jax.random.split(key)
        a_subkey, b_subkey = jax.random.split(subkey)
        a = jax.random.normal(a_subkey, (2, 5))
        b = jax.random.normal(b_subkey, (2, 5))

    with BlockJaxKeyReuse():
        key = jax.random.PRNGKey(123)
        key, a_subkey, b_subkey = jax.random.split(key, 3)
        a = jax.random.normal(key, (2, 5))
        b = jax.random.normal(a_subkey, (2, 5))
        c = jax.random.normal(b_subkey, (2, 5))

    with BlockJaxKeyReuse():
        key = jax.random.PRNGKey(123)
        key, subkey = jax.random.split(key)
        key, subkey = jax.random.split(key)


def test_initialization_does_not_reuse_keys():
    """Tests that we didn't reuse any keys during initialization."""
    with BlockJaxKeyReuse():
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
