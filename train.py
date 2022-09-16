from torch.utils.data import DataLoader

from data import PAD_index, TranslationCollateFn, WMT2014Dataset


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
    # load dataset
    train_val_test_datasets = WMT2014Dataset.load_train_val_test_datasets(
        src_lang=src_lang,
        trg_lang=trg_lang,
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        percentage_to_keep=0.01 if fast_dev_run else 1.0,
    )
    train_dataset, val_dataset, test_dataset = train_val_test_datasets

    # get tokenizer (shared across src and trg and also across train val and test)
    tokenizer = train_dataset.src_tokenizer

    # dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=TranslationCollateFn(pad_idx=PAD_index),
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=TranslationCollateFn(pad_idx=PAD_index),
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        collate_fn=TranslationCollateFn(pad_idx=PAD_index),
        num_workers=num_workers,
    )

    # TODO: implement train val and test


if __name__ == "__main__":
    train()
