from typing import Optional

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from torch.utils.data import Dataset

PAD_token = "<pad>"
UNK_token = "<unk>"
SOS_token = "<sos>"
EOS_token = "<eos>"

PAD_index = 0
UNK_index = 1
SOS_index = 2
EOS_index = 3

TOKENIZER_TYPES = {
    "bpe": {
        "model": BPE,
        "trainer": BpeTrainer,
    },
    "wordpiece": {
        "model": WordPiece,
        "trainer": WordPieceTrainer,
    },
}


def get_tokenizer(texts: list[str], tokenizer_type: str, vocab_size: int) -> Tokenizer:
    # get model and trainer for the given tokenizer type
    model_class = TOKENIZER_TYPES[tokenizer_type]["model"]
    trainer_class = TOKENIZER_TYPES[tokenizer_type]["trainer"]

    # initialize tokenizer from model
    tokenizer = Tokenizer(model_class(unk_token=UNK_token))

    # preprocessing and normalization
    tokenizer.normalizer = Sequence([NFD(), StripAccents(), Lowercase()])
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # train tokenizer
    trainer = trainer_class(
        vocab_size=vocab_size,
        special_tokens=[PAD_token, UNK_token, SOS_token, EOS_token],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))

    # post processing
    tokenizer.post_processor = TemplateProcessing(
        single=f"{SOS_token} $A {EOS_token}",
        special_tokens=[(SOS_token, SOS_index), (EOS_token, EOS_index)],
    )

    # check that our special tokens have the correct ids that we want
    assert tokenizer.token_to_id(PAD_token) == PAD_index
    assert tokenizer.token_to_id(UNK_token) == UNK_index
    assert tokenizer.token_to_id(SOS_token) == SOS_index
    assert tokenizer.token_to_id(EOS_token) == EOS_index

    return tokenizer


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_texts: list[str],
        trg_texts: list[str],
        src_tokenizer: Tokenizer,
        trg_tokenizer: Tokenizer,
    ) -> None:
        assert len(src_texts) == len(trg_texts)
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_encodings = [self.src_tokenizer.encode(text) for text in src_texts]
        self.trg_encodings = [self.trg_tokenizer.encode(text) for text in trg_texts]

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.src_encodings[idx].ids, self.trg_encodings[idx].ids

    def __len__(self) -> int:
        return len(self.src_encodings)


class TranslationCollateFn(Dataset):
    def __init__(self, pad_idx: int) -> None:
        self.pad_idx = pad_idx

    def __call__(self, batch: list[tuple[str, str]]):
        src_sequences_ids, trg_sequences_ids = zip(*batch)
        # TODO: perform padding here


class WMT2014Dataset:
    @staticmethod
    def load_dataset(
        src_lang: str,
        trg_lang: str,
        tokenizer_type: str,
        vocab_size: int,
        split: str,
        passthrough_tokenizer: Optional[Tokenizer] = None,
        percentage_to_keep: float = 1.0,
    ) -> TranslationDataset:
        # dataset name for load_dataset() must be format "non_en_lang-en"
        assert "en" in {src_lang, trg_lang}
        non_en_lang = trg_lang if src_lang == "en" else src_lang
        name = f"{non_en_lang}-en"

        # load dataset
        dataset = load_dataset("wmt14", name, split=split)

        # get subset based on percentage_to_keep
        num_examples_to_keep = round(percentage_to_keep * dataset.num_rows)
        dataset = dataset.select(range(num_examples_to_keep))

        # get src_texts and trg_texts
        pairs = [(d[src_lang], d[trg_lang]) for d in dataset["translation"]]
        src_texts, trg_texts = zip(*pairs)

        # build tokenizer (shared vocabulary) if passthrough tokenizer is not specified
        tokenizer = (
            passthrough_tokenizer
            if passthrough_tokenizer is not None
            else get_tokenizer(
                texts=src_texts + trg_texts,
                vocab_size=vocab_size,
                tokenizer_type=tokenizer_type,
            )
        )

        return TranslationDataset(
            src_texts,
            trg_texts,
            src_tokenizer=tokenizer,
            trg_tokenizer=tokenizer,
        )

    @staticmethod
    def load_train_val_test_datasets(
        src_lang: str,
        trg_lang: str,
        tokenizer_type: str,
        vocab_size: int,
        percentage_to_keep: float = 1.0,
    ) -> TranslationDataset:
        train_dataset = WMT2014Dataset.load_dataset(
            src_lang,
            trg_lang,
            tokenizer_type,
            vocab_size,
            split="train",
            percentage_to_keep=percentage_to_keep,
        )
        val_dataset = WMT2014Dataset.load_dataset(
            src_lang,
            trg_lang,
            tokenizer_type,
            vocab_size,
            split="validation",
            passthrough_tokenizer=train_dataset.src_tokenizer,
            percentage_to_keep=percentage_to_keep,
        )
        test_dataset = WMT2014Dataset.load_dataset(
            src_lang,
            trg_lang,
            tokenizer_type,
            vocab_size,
            split="test",
            passthrough_tokenizer=train_dataset.src_tokenizer,
            percentage_to_keep=percentage_to_keep,
        )

        return train_dataset, val_dataset, test_dataset
