import jax
import jax.numpy as jnp

from model import (
    initialize_transformer_params,
    transformer_forward_fn,
    transformer_predict_fn,
)


def test_initialization_is_random():
    """
    Tests that the parameter initialization is indeed random and that we didn't mess up
    jax.random.split anywhere.

    After running this function, we expect that the initialized parameters do not have
    any repeated values (since we are only generating around ~1000 numbers). Or, if
    there are, run it again with a different seed to make sure the same frequencies
    don't show up (if they do, that could be an indication you messed something up).

    As a reference, we also generate a simple list with the same number of parameters
    to see if it has repeated values (and how frequently).
    """

    def flatten_and_concat(a):
        if isinstance(a, jnp.ndarray):
            return a.flatten().tolist()

        if isinstance(a, dict):
            a = a.values()

        l = []
        for x in a:
            l += flatten_and_concat(x)
        return l

    def value_counts(a):
        b = jnp.asarray(jnp.unique(a, return_counts=True)).T
        return b[b[:, 1].argsort()][::-1]

    _seed = 127
    weights = initialize_transformer_params(
        seed=_seed,
        src_vocab_size=5,
        trg_vocab_size=10,
        d_model=8,
        d_ff=16,
        h=4,
        n_enc_layers=2,
        n_dec_layers=2,
    )
    weights_flattened = jnp.array(flatten_and_concat(weights))
    num_params = weights_flattened.shape[0]
    print(f"num params: {num_params}\n")
    print(value_counts(weights_flattened)[:20])
    value_counts_dict = {float(k): int(v) for k, v in value_counts(weights_flattened)}

    # remove 0.0 and 1.0 since those are repeated from bias + layer norm initializations
    del value_counts_dict[0.0]
    del value_counts_dict[1.0]

    # see assert message below
    assert all(v == 1 for v in value_counts_dict.values()), (
        "Some weights are repeated, this may not mean there is a bug. This check is "
        "dependent on the initialization code. If a change to the code was made, this "
        "test can break even if the initialization code doens't reuse keys. Try "
        "manually running this test with a different seed to make sure the same "
        "frequencies don't show up."
    )

    # for comparison, generate a list of random numbers with the same number of
    # parameters and see if it has repeated values
    key = jax.random.PRNGKey(_seed)
    test = jax.random.normal(key, (num_params,))
    print()
    print(value_counts(test)[:20])


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
