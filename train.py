import jax
import jax.numpy as jnp
from tokenizers import Tokenizer

from model import initialize_transformer_params, transformer_forward_fn
from utils import (
    create_unsorted_sorted_char_pairs,
    load_wmt_2014_pairs,
    train_tokenizer,
)


def categorical_cross_entropy(probabilities, target):
    return -jnp.log(probabilities[target])


def loss_fn(src_token_ids, trg_token_ids, params_dict):
    next_token_probabilities = transformer_forward_fn(
        src_token_ids,
        trg_token_ids,
        **params_dict,
    )
    loss = categorical_cross_entropy(next_token_probabilities[:-1], trg_token_ids[1:])
    return loss


def update_step(lr, grad, params_dict):
    return jax.tree_util.tree_map(lambda w, g: w - g * lr, params_dict, grad)


def data_iterator(pairs, src_tokenizer, trg_tokenizer, batch_size):
    src_texts, trg_texts = zip(*pairs)

    src_token_ids = [enc.ids for enc in src_tokenizer.encode_batch(src_texts)]
    trg_token_ids = [enc.ids for enc in trg_tokenizer.encode_batch(trg_texts)]

    for i in range(0, len(src_token_ids), batch_size):
        src_token_ids_batch = jnp.array(src_token_ids[i : i + batch_size])
        trg_token_ids_batch = jnp.array(trg_token_ids[i : i + batch_size])
        yield src_token_ids_batch, trg_token_ids_batch


def predict_data_iterator(src_texts, src_tokenizer, batch_size):
    src_encodings = src_tokenizer.encode_batch(src_texts)

    for i in range(0, len(src_encodings), batch_size):
        src_token_ids = jnp.array(src_encodings[i : i + batch_size].ids)
        yield src_token_ids


def train(
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
    src_tokenizer: Tokenizer,
    trg_tokenizer: Tokenizer,
    params_dict: dict,
    # data stuff
    train_batch_size: int = 128,
    val_batch_size: int = 256,
    test_batch_size: int = 256,
    # train hyper parameters
    n_epochs: int = 10,
    lr: float = 1e-4,
):
    for epoch in range(n_epochs):
        print(f"\n\n\n--- Epoch {epoch} ---")
        # train step
        for src_token_ids, trg_token_ids in data_iterator(
            train_pairs,
            src_tokenizer,
            trg_tokenizer,
            train_batch_size,
        ):
            train_loss, grad = jax.jit(
                jax.value_and_grad(loss_fn, argnums=2)(
                    src_token_ids,
                    trg_token_ids,
                    params_dict,
                )
            )
            params_dict = update_step(lr, grad, params_dict)
            print(f"train_loss = {train_loss}")

        # validation step
        for src_token_ids, trg_token_ids in data_iterator(
            val_pairs,
            src_tokenizer,
            trg_tokenizer,
            val_batch_size,
        ):
            val_loss = loss_fn(src_token_ids, trg_token_ids, params_dict)
            print(f"val_loss = {val_loss}")

    # test step


def train_wmt2014(
    # dataset
    src_lang: str = "en",
    trg_lang: str = "de",
    # tokenizer
    tokenizer_type: str = "bpe",
    vocab_size: int = 32000,
    # data stuff
    train_batch_size: int = 128,
    val_batch_size: int = 256,
    test_batch_size: int = 256,
    # model stuff
    seed: int = 123,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    n_enc_layers: int = 6,
    n_dec_layers: int = 6,
    # train hyper parameters
    n_epochs: int = 10,
    lr: float = 1e-4,
    # dev
    fast_dev_run: bool = True,
):
    n_train_pairs = 500 if fast_dev_run else None
    n_val_pairs = 100 if fast_dev_run else None
    n_test_pairs = 100 if fast_dev_run else None

    # get data
    train_pairs = load_wmt_2014_pairs(src_lang, trg_lang, "train")[:n_train_pairs]
    val_pairs = load_wmt_2014_pairs(src_lang, trg_lang, "validation")[:n_val_pairs]
    test_pairs = load_wmt_2014_pairs(src_lang, trg_lang, "test")[:n_test_pairs]

    # we used a shared tokenizer between src and trg, but only train it on the
    # train pairs (if val and test has a token that is not learned from train pairs
    # we want it to show up as the UNK_token to better represent predict time accuracy)
    tokenizer = train_tokenizer(
        texts=[text for pair in train_pairs for text in pair],
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
    )

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

    train(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        src_tokenizer=tokenizer,
        trg_tokenizer=tokenizer,
        params_dict=params_dict,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        n_epochs=n_epochs,
        lr=lr,
    )


def train_charsort(
    # data stuff
    train_batch_size: int = 128,
    val_batch_size: int = 256,
    test_batch_size: int = 256,
    # model stuff
    seed: int = 123,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    n_enc_layers: int = 6,
    n_dec_layers: int = 6,
    # train hyper parameters
    n_epochs: int = 10,
    lr: float = 1e-4,
    # dev
    fast_dev_run: bool = True,
):
    n_train_pairs = 500 if fast_dev_run else 100000
    n_val_pairs = 100 if fast_dev_run else 20000
    n_test_pairs = 100 if fast_dev_run else 20000

    # get data
    train_pairs = create_unsorted_sorted_char_pairs(n_train_pairs, 5, 20, 123)
    val_pairs = create_unsorted_sorted_char_pairs(n_val_pairs, 5, 20, 123)
    test_pairs = create_unsorted_sorted_char_pairs(n_test_pairs, 5, 20, 123)

    # we used a shared tokenizer between src and trg, but only train it on the
    # train pairs (if val and test has a token that is not learned from train pairs
    # we want it to show up as the UNK_token to better represent predict time accuracy)
    tokenizer = train_tokenizer(
        texts=[text for pair in train_pairs for text in pair],
        tokenizer_type="charlevel",
        vocab_size=None,
    )
    vocab_size = tokenizer.get_vocab_size()

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

    train(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        src_tokenizer=tokenizer,
        trg_tokenizer=tokenizer,
        params_dict=params_dict,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        n_epochs=n_epochs,
        lr=lr,
    )


if __name__ == "__main__":
    train_charsort(fast_dev_run=True)
