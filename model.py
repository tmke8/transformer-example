import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    r"""Inject information about the relative or absolute position of the tokens in the sequence.

    The positional encodings have the same dimension as the embeddings, so that the two can be
    summed. Here, we use sine and cosine functions of different frequencies.

    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)

    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        r"""Inputs of forward function

        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self, ntoken: int, ninp: int, nhead: int, nhid: int, nlayers: int, dropout: float = 0.5
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ninp, nhead=nhead, dim_feedforward=nhid
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers, norm=nn.LayerNorm(ninp))
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.cached_mask: Tensor | None = None
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

        # Initiate parameters in the transformer
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, *, has_mask: bool) -> Tensor:
        if has_mask:
            if self.cached_mask is None or self.cached_mask.size(0) != len(src):
                mask = nn.Transformer.generate_square_subsequent_mask(len(src), device=src.device)
                self.cached_mask = mask
        else:
            self.cached_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.cached_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
