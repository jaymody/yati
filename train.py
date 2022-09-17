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
    src_lang: str = "en",
    trg_lang: str = "de",
    # tokenizer
    tokenizer_type: str = "bpe",
    vocab_size: int = 32000,
    # data stuff
    train_batch_size: int = 128,
    val_batch_size: int = 256,
    test_batch_size: int = 256,
    num_workers: int = 8,
    # model stuff
    seed: int = 123,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    n_enc_layers: int = 6,
    n_dec_layers: int = 6,
    # dev
    fast_dev_run: bool = True,
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
