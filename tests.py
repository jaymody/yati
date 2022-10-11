import inspect
import math
import os
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from scipy.stats import ks_2samp

from model import (
    create_masks,
    create_positional_embeddings,
    final_linear_layer,
    initialize_transformer_params,
    layer_norm,
    position_wise_ffn,
    scaled_dot_product_attention,
    transformer_forward_fn,
    transformer_predict_fn,
    xavier_uniform,
)

_SEED = 123456789
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed(_SEED)
torch.backends.cudnn.deterministic = True
os.environ["PYTHONHASHSEED"] = str(_SEED)


def jax_to_torch(arr):
    return torch.from_numpy(np.array(arr))


def torch_to_jax(tensor):
    return jnp.array(tensor.detach().numpy())


def allclose(torch_tensor, jax_array, atol=1e-6, rtol=1e-6):
    # maybe also check not nan, inf, or zero?
    return np.allclose(torch_tensor.detach().numpy(), np.array(jax_array), atol, rtol)


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
            key = jax.random.PRNGKey(_SEED)
            a = jax.random.normal(key, (2, 5))
            b = jax.random.normal(key, (2, 5))

    with pytest.raises(AssertionError):
        with BlockJaxKeyReuse():
            key = jax.random.PRNGKey(_SEED)
            key, subkey = jax.random.normal(key, (2, 5))
            key = jax.random.PRNGKey(_SEED)
            key, subkey = jax.random.normal(key, (2, 5))

    with pytest.raises(AssertionError):
        with BlockJaxKeyReuse():
            key = jax.random.PRNGKey(_SEED)
            key, a_subkey, b_subkey = jax.random.split(key, 3)
            a = jax.random.normal(key, (2, 5))
            b = jax.random.normal(a_subkey, (2, 5))
            c = jax.random.normal(b_subkey, (2, 5))
            key, subkey = jax.random.split(key)

    # the following tests should not throw any errors
    with BlockJaxKeyReuse():
        key = jax.random.PRNGKey(_SEED)
        key, a_subkey, b_subkey = jax.random.split(key, 3)
        a = jax.random.normal(a_subkey, (2, 5))
        b = jax.random.normal(b_subkey, (2, 5))

    with BlockJaxKeyReuse():
        key = jax.random.PRNGKey(_SEED)
        key, subkey = jax.random.split(key)
        a_subkey, b_subkey = jax.random.split(subkey)
        a = jax.random.normal(a_subkey, (2, 5))
        b = jax.random.normal(b_subkey, (2, 5))

    with BlockJaxKeyReuse():
        key = jax.random.PRNGKey(_SEED)
        key, a_subkey, b_subkey = jax.random.split(key, 3)
        a = jax.random.normal(key, (2, 5))
        b = jax.random.normal(a_subkey, (2, 5))
        c = jax.random.normal(b_subkey, (2, 5))

    with BlockJaxKeyReuse():
        key = jax.random.PRNGKey(_SEED)
        key, subkey = jax.random.split(key)
        key, subkey = jax.random.split(key)


def test_initialization_does_not_reuse_keys():
    """Tests that we didn't reuse any keys during initialization."""
    with BlockJaxKeyReuse():
        initialize_transformer_params(
            seed=_SEED,
            src_vocab_size=200,
            trg_vocab_size=300,
            d_model=512,
            d_ff=2048,
            h=8,
            n_enc_layers=6,
            n_dec_layers=6,
        )


def test_forward_fn_and_predict_fn():
    """Tests forward fn and predict fn run without running into errors."""
    params = initialize_transformer_params(
        seed=_SEED,
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

    logits = transformer_predict_fn(
        src_token_ids=jnp.array([11, 8, 90, 5, 101, 99]),
        **params,
        sos_idx=1,
        max_sequence_length=10,
        pad_idx=0,
    )


def test_layer_norm():
    eps = 1e-6
    seq_len = 5
    d_model = 10

    X = torch.normal(0, 1, (seq_len, d_model))

    torch_layer_norm = torch.nn.LayerNorm(d_model, eps)
    torch_output = torch_layer_norm(X)

    jax_output = layer_norm(
        x=torch_to_jax(X),
        gamma=torch_to_jax(torch_layer_norm.weight),
        beta=torch_to_jax(torch_layer_norm.bias),
        eps=eps,
    )

    assert allclose(torch_output, jax_output)


def test_position_wise_ffn():
    class PositionwiseFeedForward(torch.nn.Module):
        """Positionwise Feed Forward as implemented by 'The Annotated Transformer', see:
        https://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks
        """

        def __init__(self, d_model, d_ff, dropout=0.1):
            super(PositionwiseFeedForward, self).__init__()
            self.w_1 = torch.nn.Linear(d_model, d_ff)
            self.w_2 = torch.nn.Linear(d_ff, d_model)
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, x):
            return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))

    seq_len = 10
    d_model = 512
    d_ff = 2048

    X = torch.normal(0, 1, (seq_len, d_model))

    torch_ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0)
    torch_output = torch_ffn(X)

    jax_output = position_wise_ffn(
        X=torch_to_jax(X),
        W1=torch_to_jax(torch_ffn.w_1.weight).T,
        b1=torch_to_jax(torch_ffn.w_1.bias),
        W2=torch_to_jax(torch_ffn.w_2.weight).T,
        b2=torch_to_jax(torch_ffn.w_2.bias),
    )

    assert allclose(torch_output, jax_output)


def test_final_linear_layer():
    class Generator(torch.nn.Module):
        """Last transformer layer as implemented by 'The Annotated Transformer', see:
        https://nlp.seas.harvard.edu/2018/04/03/attention.html#model-architecture
        """

        def __init__(self, d_model, vocab):
            super(Generator, self).__init__()
            self.proj = torch.nn.Linear(d_model, vocab)

        def forward(self, x):
            return torch.nn.functional.log_softmax(self.proj(x), dim=-1)

    seq_len = 10
    d_model = 512
    vocab_size = 100

    X = torch.normal(0, 1, (seq_len, d_model))

    torch_layer = Generator(d_model, vocab_size)
    # TODO: torch final linear layer uses bias, we don't, should we?
    torch.nn.init.zeros_(torch_layer.proj.bias)  # zero out bias since we don't use it
    torch_output = torch_layer(X)

    jax_output = jax.nn.log_softmax(
        final_linear_layer(torch_to_jax(X), torch_to_jax(torch_layer.proj.weight).T),
        axis=-1,
    )

    assert allclose(torch_output, jax_output)


def test_create_positional_embeddings():
    class PositionalEncoding(torch.nn.Module):
        """Positional encodings as implemented by 'The Annotated Transformer.', see:
        https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
        """

        def __init__(self, d_model, dropout, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = torch.nn.Dropout(p=dropout)

            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            x = x + torch.autograd.Variable(
                self.pe[:, : x.size(1)], requires_grad=False
            )
            return self.dropout(x)

    d_model = 20
    seq_len = 100

    dummy_embedding = torch.zeros((1, seq_len, d_model))
    torch_output = PositionalEncoding(d_model, 0).forward(dummy_embedding).squeeze()

    jax_output = create_positional_embeddings(seq_len=seq_len, d_model=d_model)

    assert allclose(torch_output, jax_output, rtol=1e-5)


def test_xavier_uniform():
    def _test(gain, shape, seed):
        torch_output = torch.empty(shape)
        torch.nn.init.xavier_uniform_(torch_output, gain=gain)

        key = jax.random.PRNGKey(seed)
        jax_output = xavier_uniform(key, shape, gain=gain)

        # high p value means samples are more likely taken from the same distribution
        torch_output_flat_numpy = torch_output.numpy().flatten()
        jax_output_flat_numpy = np.array(jax_output).flatten()
        assert ks_2samp(torch_output_flat_numpy, jax_output_flat_numpy).pvalue > 0.01

    # we need to use sufficiently large array so our number of samples pulled from the
    # underlying distributions are not too small (this would make it hard to determine
    # if two distributions are similar, since there are too few examples)
    _test(gain=2.3, shape=(2000, 100), seed=_SEED)
    _test(gain=2.3, shape=(2000, 100), seed=_SEED + 1)
    _test(gain=1.0, shape=(5000, 200), seed=_SEED + 2)
    _test(gain=0.5, shape=(1000, 500), seed=_SEED + 3)
    _test(gain=1e6, shape=(525, 500), seed=_SEED + 4)
    _test(gain=1e-6, shape=(200, 20), seed=_SEED + 5)


def test_create_masks():
    eps = -1.0
    encoder_src_mask, decoder_src_mask, decoder_trg_mask = create_masks(
        src_token_ids=jnp.array([1, 2, 3, 0, 0]),
        trg_token_ids=jnp.array([4, 5, 6, 7, 0, 0]),
        pad_idx=0,
        eps=eps,
    )

    expected_encoder_src_mask = jnp.array(
        [
            [0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, eps, eps],
        ]
    )
    expected_decoder_src_mask = jnp.array(
        [
            [0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, eps, eps],
        ]
    )
    expected_decoder_trg_mask = jnp.array(
        [
            [0.0, eps, eps, eps, eps, eps],
            [0.0, 0.0, eps, eps, eps, eps],
            [0.0, 0.0, 0.0, eps, eps, eps],
            [0.0, 0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, 0.0, eps, eps],
            [0.0, 0.0, 0.0, 0.0, eps, eps],
        ]
    )
    assert jnp.all(encoder_src_mask == expected_encoder_src_mask)
    assert jnp.all(decoder_src_mask == expected_decoder_src_mask)
    assert jnp.all(decoder_trg_mask == expected_decoder_trg_mask)


def test_scaled_dot_product_attention():
    def attention(query, key, value, mask=None, dropout=None):
        """Scaled Dot Product Attention as implemented by 'The Annotated Transformer',
        see: https://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    d_k = 512
    d_v = 1028
    q_len = 200
    k_len = 100
    q = torch.randn((q_len, d_k))
    k = torch.randn((k_len, d_k))
    v = torch.randn((k_len, d_v))

    # for torch implementation, mask at 0 is filled with -inf
    # for jax implementation, the mask gets added without alteration
    torch_mask = torch.bernoulli(torch.ones((q_len, k_len)) * 0.1)
    jax_mask = torch_to_jax(1 - torch_mask) * -1e9

    torch_output = attention(q, k, v, torch_mask)[0]

    jax_output = scaled_dot_product_attention(
        torch_to_jax(q), torch_to_jax(k), torch_to_jax(v), jax_mask
    )

    assert allclose(torch_output, jax_output, rtol=1e-5)


def test_multihead_attention():
    # TODO: implement this
    pass


def test_cross_entropy_loss():
    from train import cross_entropy_loss

    num_examples = 1000
    num_classes = 12
    x = torch.randn((num_examples, num_classes))
    y = torch.randint(0, num_classes, (num_examples,))

    torch_loss_fn = torch.nn.CrossEntropyLoss()
    torch_output = torch_loss_fn(x, y)

    jax_output = cross_entropy_loss(torch_to_jax(x), torch_to_jax(y))

    assert allclose(torch_output, jax_output)
