from functools import partial

import jax
import jax.numpy as jnp
from tokenizers import Tokenizer

from model import (
    initialize_transformer_params_with_shared_weight_matrix,
    transformer_forward_fn,
    transformer_predict_fn,
)
from utils import (
    PAD_index,
    SOS_index,
    create_unsorted_sorted_char_pairs,
    load_wmt_2014_pairs,
    train_tokenizer,
)

model_kwargs_for_size = {
    "base": {
        "d_model": 512,
        "d_ff": 2048,
        "h": 8,
        "n_enc_layers": 6,
        "n_dec_layers": 6,
    },
    "tiny": {
        "d_model": 128,
        "d_ff": 512,
        "h": 4,
        "n_enc_layers": 3,
        "n_dec_layers": 3,
    },
}


# TODO: why does jitting this function break when called by loss fn and with
# static argnames set to pad_idx
def forward_fn_batched(src_token_ids, trg_token_ids, params_dict, pad_idx):
    # src_token_ids -> (batch_size, src_seq_len)
    # trg_token_ids -> (batch_size, trg_seq_len)
    # params_dict -> dict of params
    # output -> (batch_size, trg_seq_len, vocab_size)
    return jax.vmap(
        transformer_forward_fn, in_axes=(0, 0, None, None, None, None, None, None)
    )(
        src_token_ids,
        trg_token_ids,
        params_dict["shared_weight_matrix"],
        params_dict["shared_weight_matrix"],
        params_dict["encoder_stack"],
        params_dict["decoder_stack"],
        params_dict["shared_weight_matrix"].T,
        pad_idx,
    )


# TODO: why is this so slow to compile and run?
@partial(jax.jit, static_argnames=["max_seq_len", "sos_idx", "pad_idx"])
def predict_fn_batched(src_token_ids, params_dict, max_seq_len, sos_idx, pad_idx):
    # src_token_ids -> (batch_size, src_seq_len)
    # params_dict -> dict of params
    # output -> (batch_size, trg_seq_len)
    return jax.vmap(
        transformer_predict_fn,
        in_axes=(0, None, None, None, None, None, None, None, None),
    )(
        src_token_ids,
        max_seq_len,
        params_dict["shared_weight_matrix"],
        params_dict["shared_weight_matrix"],
        params_dict["encoder_stack"],
        params_dict["decoder_stack"],
        params_dict["shared_weight_matrix"].T,
        sos_idx,
        pad_idx,
    )


@partial(jax.jit, static_argnames=["pad_idx"])
def loss_fn(src_token_ids, trg_token_ids, params_dict, pad_idx):
    # src_token_ids -> (batch_size, src_seq_len)
    # trg_token_ids -> (batch_size, trg_seq_len)
    # params_dict -> dict of params
    # output -> loss as a scalar
    next_token_probabilities = forward_fn_batched(
        src_token_ids, trg_token_ids, params_dict, pad_idx
    )

    # cross entropy loss
    batch_size = src_token_ids.shape[0]
    vocab_size = next_token_probabilities.shape[-1]
    logits = next_token_probabilities[:, :-1, :]  # skip last token
    labels = jax.nn.one_hot(trg_token_ids[:, 1:], vocab_size)  # skip first token
    loss = jnp.sum(labels * -jnp.log(logits)) / batch_size
    return loss


@partial(jax.jit, static_argnames=["pad_idx"])
def loss_and_grad_fn_jitted(src_token_ids, trg_token_ids, params_dict, pad_idx):
    return jax.value_and_grad(loss_fn, argnums=2)(
        src_token_ids, trg_token_ids, params_dict, pad_idx
    )


def clip_gradients(grad, min_val, max_val):
    return jax.tree_util.tree_map(lambda x: jnp.clip(x, min_val, max_val), grad)


def update_step(params_dict, grad, lr):
    return jax.tree_util.tree_map(lambda w, g: w - lr * g, params_dict, grad)


def compute_loss_and_accuracy(
    all_src_token_ids,
    all_trg_token_ids,
    params_dict,
    batch_size,
    pad_idx,
    sos_idx,
    max_seq_len,
):
    running_loss = 0
    n_total = 0
    for src_token_ids, trg_token_ids in get_batches(
        all_src_token_ids, all_trg_token_ids, batch_size
    ):
        batch_size = src_token_ids.shape[0]
        n_total += batch_size
        running_loss += (
            loss_fn(src_token_ids, trg_token_ids, params_dict, pad_idx) * batch_size
        )
    return running_loss / n_total, -1  # TODO: implement accuracy, probably bleu score


def pad_token_ids(token_ids, max_seq_len: int, pad_idx: int):
    # token_ids -> (seq_len)
    # output -> (max_seq_len)
    seq_len = token_ids.shape[0]
    assert seq_len <= max_seq_len
    return jnp.pad(
        token_ids,
        (0, max_seq_len - seq_len),
        mode="constant",
        constant_values=pad_idx,
    )


def encode_and_pad(texts, tokenizer, max_seq_len, pad_index):
    # texts -> list of strings
    # output -> (batch_size, max_seq_len)
    return jnp.array(
        [
            pad_token_ids(jnp.array(enc.ids), max_seq_len, pad_index)
            for enc in tokenizer.encode_batch(texts)
        ]
    )


def get_src_trg_token_ids(pairs, src_tokenizer, trg_tokenizer, max_seq_len):
    src_texts, trg_texts = zip(*pairs)
    src_token_ids = encode_and_pad(src_texts, src_tokenizer, max_seq_len, PAD_index)
    trg_token_ids = encode_and_pad(trg_texts, trg_tokenizer, max_seq_len, PAD_index)
    return src_token_ids, trg_token_ids


def get_batches(train_X, train_y, batch_size):
    for i in range(0, len(train_X), batch_size):
        yield train_X[i : i + batch_size], train_y[i : i + batch_size]


def train(
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
    src_tokenizer: Tokenizer,
    trg_tokenizer: Tokenizer,
    params_dict: dict,
    max_seq_len: int,
    train_batch_size: int,
    val_batch_size: int,
    test_batch_size: int,
    num_train_steps: int,
    val_every_n_steps: int,
    lr: float,
):
    # encode string sequences to ids
    train_src_token_ids, train_trg_token_ids = get_src_trg_token_ids(
        train_pairs, src_tokenizer, trg_tokenizer, max_seq_len
    )
    val_src_token_ids, val_trg_token_ids = get_src_trg_token_ids(
        val_pairs, src_tokenizer, trg_tokenizer, max_seq_len
    )
    test_src_token_ids, test_trg_token_ids = get_src_trg_token_ids(
        test_pairs, src_tokenizer, trg_tokenizer, max_seq_len
    )

    # compute loss and accuracy and train and val sets
    def compute_and_print_eval_metrics(params_dict):
        train_loss, train_acc = compute_loss_and_accuracy(
            train_src_token_ids,
            train_trg_token_ids,
            params_dict,
            train_batch_size,
            PAD_index,
            SOS_index,
            max_seq_len,
        )
        val_loss, val_acc = compute_loss_and_accuracy(
            val_src_token_ids,
            val_trg_token_ids,
            params_dict,
            val_batch_size,
            PAD_index,
            SOS_index,
            max_seq_len,
        )
        print(f"    train_loss = {train_loss}")
        print(f"    train_acc = {train_acc}")
        print(f"    val_loss = {val_loss}")
        print(f"    val_acc = {val_acc}")

    # compute metrics before model is trained for reference
    print("--- eval metrics before training ---")
    compute_and_print_eval_metrics(params_dict)

    # train loop
    train_step_i = 0
    while train_step_i < num_train_steps:
        for src_token_ids, trg_token_ids in get_batches(
            train_src_token_ids, train_trg_token_ids, train_batch_size
        ):
            loss, grad = loss_and_grad_fn_jitted(
                src_token_ids, trg_token_ids, params_dict, PAD_index
            )
            params_dict = update_step(params_dict, grad, lr)
            print()
            print(f"--- step {train_step_i} / {num_train_steps} ---")
            print(f"loss of batch = {loss}")

            # eval metrics
            if train_step_i % val_every_n_steps == 0:
                compute_and_print_eval_metrics(params_dict)

            train_step_i += 1


# def train_wmt2014(
#     src_lang: str = "en",
#     trg_lang: str = "de",
#     tokenizer_type: str = "bpe",
#     vocab_size: int = 32000,
#     max_seq_len: int = 256,  # TODO: change this after doing EDA on the dataset
#     train_batch_size: int = 128,
#     val_batch_size: int = 256,
#     test_batch_size: int = 256,
#     seed: int = 123,
#     n_epochs: int = 10,
#     lr: float = 1e-4,
#     fast_dev_run: bool = True,
# ):
#     n_train_pairs = 500 if fast_dev_run else None
#     n_val_pairs = 100 if fast_dev_run else None
#     n_test_pairs = 100 if fast_dev_run else None

#     # get data
#     train_pairs = load_wmt_2014_pairs(src_lang, trg_lang, "train")[:n_train_pairs]
#     val_pairs = load_wmt_2014_pairs(src_lang, trg_lang, "validation")[:n_val_pairs]
#     test_pairs = load_wmt_2014_pairs(src_lang, trg_lang, "test")[:n_test_pairs]

#     # we used a shared tokenizer between src and trg, but only train it on the
#     # train pairs (if val and test has a token that is not learned from train pairs
#     # we want it to show up as the UNK_token to better represent predict time accuracy)
#     tokenizer = train_tokenizer(
#         texts=[text for pair in train_pairs for text in pair],
#         tokenizer_type=tokenizer_type,
#         vocab_size=vocab_size,
#     )

#     params_dict = initialize_transformer_params_with_shared_weight_matrix(
#         seed=seed,
#         vocab_size=vocab_size,
#         **model_kwargs_for_size["tiny" if fast_dev_run else "base"],
#     )

#     train(
#         train_pairs=train_pairs,
#         val_pairs=val_pairs,
#         test_pairs=test_pairs,
#         src_tokenizer=tokenizer,
#         trg_tokenizer=tokenizer,
#         params_dict=params_dict,
#         max_seq_len=max_seq_len,
#         train_batch_size=train_batch_size,
#         val_batch_size=val_batch_size,
#         test_batch_size=test_batch_size,
#         n_epochs=n_epochs,
#         lr=lr,
#     )


def train_charsort(
    min_length: int = 5,
    max_length: int = 22,
    train_batch_size: int = 128,
    val_batch_size: int = 256,
    test_batch_size: int = 256,
    seed: int = 123,
    num_train_steps: int = 100000,
    lr: float = 1e-3,
    val_every_n_steps: int = 50,
    fast_dev_run: bool = True,
):
    n_train_pairs = 500 if fast_dev_run else 10000
    n_val_pairs = 100 if fast_dev_run else 2000
    n_test_pairs = 100 if fast_dev_run else 2000

    # get data
    train_pairs = create_unsorted_sorted_char_pairs(
        n_train_pairs, min_length, max_length, seed
    )
    val_pairs = create_unsorted_sorted_char_pairs(
        n_val_pairs, min_length, max_length, seed
    )
    test_pairs = create_unsorted_sorted_char_pairs(
        n_test_pairs, min_length, max_length, seed
    )

    # we used a shared tokenizer between src and trg, but only train it on the
    # train pairs (if val and test has a token that is not learned from train pairs
    # we want it to show up as the UNK_token to better represent predict time accuracy)
    tokenizer = train_tokenizer(
        texts=[text for pair in train_pairs for text in pair],
        tokenizer_type="charlevel",
        vocab_size=None,
    )
    vocab_size = tokenizer.get_vocab_size()

    params_dict = initialize_transformer_params_with_shared_weight_matrix(
        seed=seed,
        vocab_size=vocab_size,
        **model_kwargs_for_size["tiny" if fast_dev_run else "base"],
    )

    train(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        src_tokenizer=tokenizer,
        trg_tokenizer=tokenizer,
        params_dict=params_dict,
        max_seq_len=max_length + 2,  # + 2 for SOS and EOS tokens
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        num_train_steps=num_train_steps,
        val_every_n_steps=val_every_n_steps,
        lr=lr,
    )


if __name__ == "__main__":
    from jax.config import config

    config.update("jax_debug_nans", True)
    train_charsort(fast_dev_run=False)
