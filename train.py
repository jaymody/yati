from typing import Optional

from utils import load_wmt_2014_pairs, train_tokenizer


def get_train_val_test_pairs(
    src_lang: str,
    trg_lang: str,
    n_train_pairs: Optional[int] = None,
    n_val_pairs: Optional[int] = None,
    n_test_pairs: Optional[int] = None,
):
    train_pairs = load_wmt_2014_pairs(src_lang, trg_lang, "train")[:n_train_pairs]
    val_pairs = load_wmt_2014_pairs(src_lang, trg_lang, "validation")[:n_val_pairs]
    test_pairs = load_wmt_2014_pairs(src_lang, trg_lang, "test")[:n_test_pairs]
    return train_pairs, val_pairs, test_pairs


def train(
    # dataset
    src_lang="en",
    trg_lang="de",
    # tokenizer
    tokenizer_type="bpe",
    vocab_size=32000,
    # data stuff
    train_batch_size=128,
    val_batch_size=256,
    test_batch_size=256,
    num_workers=8,
    # model stuff
    seed=123,
    d_model=512,
    d_ff=2048,
    h=8,
    n_enc_layers=6,
    n_dec_layers=6,
    # dev
    fast_dev_run=True,
):
    # load pairs
    train_pairs, val_pairs, test_pairs = get_train_val_test_pairs(
        src_lang,
        trg_lang,
        n_test_pairs=1000 if fast_dev_run else None,
        n_val_pairs=1000 if fast_dev_run else None,
        n_test_pairs=1000 if fast_dev_run else None,
    )

    # train tokenizer (we use a shared tokenizer between src texts and trg texts, but
    # is only trained on the train pair texts)
    tokenizer = train_tokenizer(
        texts=[text for pair in train_pairs for text in pair],
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
    )

    # TODO: implement the rest of this function


if __name__ == "__main__":
    train()
