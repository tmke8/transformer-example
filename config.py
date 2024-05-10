from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch


@dataclass
class BaseConfig:
    data: Path = Path("./data/wikitext-2")
    """location of the data corpus"""
    cuda: bool = True
    """use CUDA"""
    seed: int = 1111
    """random seed"""
    mps: bool = False
    """enables macOS GPU training"""

    @cached_property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            if not self.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with cuda=true.")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if not self.mps:
                print("WARNING: You have mps device, to enable macOS GPU run with mps=true.")
        use_mps = self.mps and torch.backends.mps.is_available()
        if self.cuda:
            return torch.device("cuda")
        elif use_mps:
            return torch.device("mps")
        else:
            return torch.device("cpu")


@dataclass
class TrainConfig(BaseConfig):
    emsize: int = 200
    """size of word embeddings"""
    nhid: int = 200
    """number of hidden units per layer"""
    nlayers: int = 2
    """number of layers"""
    lr: float = 20
    """initial learning rate"""
    clip: float = 0.25
    """gradient clipping"""
    epochs: int = 40
    """upper epoch limit"""
    batch_size: int = 20
    """batch size"""
    bptt: int = 35
    """sequence length"""
    dropout: float = 0.2
    """dropout applied to layers (0 = no dropout)"""
    tied: bool = False
    """tie the word embedding and softmax weights"""
    log_interval: int = 200
    """report interval"""
    save: Path = Path("model.pt")
    """path to save the final model"""
    onnx_export: Path | None = None
    """path to export the final model in onnx format"""
    nhead: int = 2
    """the number of heads in the encoder/decoder of the transformer model"""
    dry_run: bool = False
    """verify the code and the model"""


@dataclass
class GenerateConfig(BaseConfig):
    checkpoint: Path = Path("./model.pt")
    """model checkpoint to use"""
    outf: Path = Path("generated.txt")
    """output file for generated text"""
    words: int = 1000
    """number of words to generate"""
    temperature: float = 1.0
    """temperature - higher will increase diversity"""
    log_interval: int = 100
    """reporting interval"""
