import random
import string
from typing import Optional

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel, WordPiece
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import Split, WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, WordPieceTrainer

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
        "model_class": BPE,
        "trainer_class": BpeTrainer,
        "pre_tokenizer": WhitespaceSplit(),
        "normalizer": Sequence([NFD(), StripAccents(), Lowercase()]),
    },
    "wordpiece": {
        "model_class": WordPiece,
        "trainer_class": WordPieceTrainer,
        "pre_tokenizer": WhitespaceSplit(),
        "normalizer": Sequence([NFD(), StripAccents(), Lowercase()]),
    },
    "charlevel": {
        "model_class": WordLevel,
        "trainer_class": WordLevelTrainer,
        "pre_tokenizer": Split("", behavior="isolated"),
        "normalizer": None,
    },
}


def random_sequence(
    rng: random.Random, min_length: int, max_length: int, replace: bool, population: str
):
    assert min_length >= 1
    assert max_length >= min_length
    if replace:
        assert max_length <= len(population)

    k = rng.randint(min_length, max_length)
    if replace:
        return rng.choices(population, k=k)
    else:
        return rng.sample(population, k=k)


def train_tokenizer(
    texts: list[str],
    tokenizer_type: str,
    vocab_size: Optional[int] = None,
) -> Tokenizer:
    # get model, trainer, pre_tokenizer, and normalizer based on token type
    model_class = TOKENIZER_TYPES[tokenizer_type]["model_class"]
    trainer_class = TOKENIZER_TYPES[tokenizer_type]["trainer_class"]
    pre_tokenizer = TOKENIZER_TYPES[tokenizer_type]["pre_tokenizer"]
    normalizer = TOKENIZER_TYPES[tokenizer_type]["normalizer"]

    # initialize tokenizer from model
    tokenizer = Tokenizer(model_class(unk_token=UNK_token))

    # preprocessing and normalization
    if pre_tokenizer is not None:
        tokenizer.pre_tokenizer = pre_tokenizer
    if normalizer is not None:
        tokenizer.normalizer = normalizer

    # train tokenizer
    # we need to do this weird trainer_kwargs thing since setting vocab_size = None
    # breaks trainers (either you don't set it at all, or it must be an integer)
    trainer_kwargs = {} if vocab_size is None else {"vocab_size": vocab_size}
    trainer = trainer_class(
        **trainer_kwargs,
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


def load_wmt_2014_pairs(
    src_lang: str,
    trg_lang: str,
    split: str,
):
    # dataset name for load_dataset() must be format "non_en_lang-en"
    assert "en" in {src_lang, trg_lang}
    non_en_lang = trg_lang if src_lang == "en" else src_lang
    name = f"{non_en_lang}-en"

    # load dataset
    dataset = load_dataset("wmt14", name, split=split)

    # build pairs list
    pairs = [(d[src_lang], d[trg_lang]) for d in dataset["translation"]]

    return pairs


def create_unsorted_sorted_char_pairs(
    n_examples: int,
    min_length: int,
    max_length: int,
    seed: int,
    replace: bool = True,
    population: str = string.ascii_lowercase,
):
    """Creates pairs where the src seqs are unsorted and trg seqs are sorted.

    Example
    -------
    src: kjcqblkaz
    trg: abcjkklqz
    """
    rng = random.Random(seed)
    rand_seq = lambda: random_sequence(rng, min_length, max_length, replace, population)

    srcs = [rand_seq() for _ in range(n_examples)]
    trgs = [sorted(src) for src in srcs]  # sort

    srcs = ["".join(src) for src in srcs]
    trgs = ["".join(trg) for trg in trgs]

    pairs = list(zip(srcs, trgs))
    return pairs


def create_same_pairs(
    n_examples: int,
    min_length: int,
    max_length: int,
    seed: int,
    replace: bool = True,
    population: str = string.ascii_lowercase,
):
    """Create pairs where the src seqs and trg seqs are the same.

    Example
    -------
    src: kjcqblkaz
    trg: kjcqblkaz
    """
    rng = random.Random(seed)
    rand_seq = lambda: random_sequence(rng, min_length, max_length, replace, population)

    srcs = [rand_seq() for _ in range(n_examples)]
    trgs = [src for src in srcs]  # copy

    srcs = ["".join(src) for src in srcs]
    trgs = ["".join(trg) for trg in trgs]

    pairs = list(zip(srcs, trgs))
    return pairs
