import jax
import jax.numpy as jnp

from main import create_transformer_weights


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

    _seed = 111
    weights = create_transformer_weights(
        _seed, 512 // 128, 512 // 128, 512 // 128, 2048 // 128, 2, 2, 2, 10
    )
    weights_flattened = jnp.array(flatten_and_concat(weights))
    num_params = weights_flattened.shape[0]
    print(f"num params: {num_params}\n")
    print(value_counts(weights_flattened)[:20])

    key = jax.random.PRNGKey(_seed)
    test = jax.random.normal(key, (num_params,))
    print()
    print(value_counts(test)[:20])
