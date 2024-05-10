from pathlib import Path

import torch
from torch import Tensor


class Dictionary:
    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)


class Corpus:
    def __init__(self, path: Path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path / "train.txt")
        self.valid = self.tokenize(path / "valid.txt")
        self.test = self.tokenize(path / "test.txt")

    def tokenize(self, path: Path) -> Tensor:
        """Tokenizes a text file."""
        assert path.exists()
        # Add words to the dictionary
        with path.open("r", encoding="utf8") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with path.open("r", encoding="utf8") as f:
            idss: list[Tensor] = []
            for line in f:
                words = line.split() + ["<eos>"]
                ids: list[int] = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
        return torch.cat(idss)
