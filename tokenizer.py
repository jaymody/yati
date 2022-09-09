class Tokenizer:
    """Tokenizes text into tokens and sequence of tokens.

    token (str)
        the smallest divisable component in a tokenization model
    seq (list of str)
        a sequence of tokens that encodes some meaning as an ordered (or
        unordered) collection
    text (str)
        a string that can be broken down into a seq
    """

    PAD_tkn = "<pad>"
    SOS_tkn = "<sos>"
    EOS_tkn = "<eos>"
    UNK_tkn = "<unk>"

    PAD_idx = 0
    SOS_idx = 1
    EOS_idx = 2
    UNK_idx = 3

    def __init__(self):
        self.stoc = {}
        self.stoi = {
            self.PAD_tkn: self.PAD_idx,
            self.SOS_tkn: self.SOS_idx,
            self.EOS_tkn: self.EOS_idx,
            self.UNK_tkn: self.UNK_idx,
        }
        self.itos = {v: k for k, v in self.stoi.items()}
        self.size = len(self.stoi)
        self.max_length = 0

    def __len__(self):
        return len(self.stoi)

    def add_token(self, token):
        if token not in self.stoi:
            self.stoi[token] = self.size
            self.stoc[token] = 1
            self.itos[self.size] = token
            self.size += 1
        else:
            self.stoc[token] += 1

    def fit_on_text(self, text):
        seq = self.text_to_seq(text)

        if len(seq) > self.max_length:
            self.max_length = len(seq)

        for token in seq:
            self.add_token(token)

    def fit_on_texts(self, texts):
        from tqdm import tqdm

        for text in tqdm(texts, desc="fitting tokenizer"):
            self.fit_on_text(text)

    def text_to_seq(self, text):
        raise NotImplementedError()

    def seq_to_text(self, seq):
        raise NotImplementedError()


class CharTokenizer(Tokenizer):
    def text_to_seq(self, text):
        return list(text)

    def seq_to_text(self, seq):
        return "".join(seq)
