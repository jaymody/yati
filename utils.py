import random
import string

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

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


def train_tokenizer(
    texts: list[str],
    tokenizer_type: str,
    vocab_size: int,
) -> Tokenizer:
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
    assert min_length >= 1
    assert max_length >= min_length
    if replace:
        assert max_length <= len(population)

    rng = random.Random(seed)

    def random_sequence():
        k = rng.randint(min_length, max_length)
        if replace:
            return rng.choices(population, k=k)
        else:
            return rng.sample(population, k=k)

    srcs = [random_sequence() for _ in range(n_examples)]
    trgs = [sorted(src) for src in srcs]

    srcs = ["".join(src) for src in srcs]
    trgs = ["".join(trg) for trg in trgs]

    pairs = list(zip(srcs, trgs))
    return pairs
