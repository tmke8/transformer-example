import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.onnx
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.modules.loss import _Loss

from config import TrainConfig
from data import Corpus
from model import TransformerModel


@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def main(hydra_config: DictConfig) -> None:
    args = instantiate(hydra_config, _convert_="object")
    assert isinstance(args, TrainConfig)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = Corpus(args.data)

    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)
    model = TransformerModel(
        ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout
    ).to(args.device)

    eval_batch_size = 10

    trainer = Trainer(
        args=args,
        model=model,
        criterion=nn.NLLLoss(),
        ntokens=ntokens,
        curr_lr=args.lr,
        train_data=batchify(corpus.train, args.batch_size, args.device),
        val_data=batchify(corpus.valid, eval_batch_size, args.device),
        test_data=batchify(corpus.test, eval_batch_size, args.device),
    )
    trainer.run()


@dataclass(kw_only=True)
class Trainer:
    args: TrainConfig
    model: TransformerModel
    criterion: _Loss
    ntokens: int
    curr_lr: float
    train_data: Tensor
    val_data: Tensor
    test_data: Tensor
    best_val_loss: float | None = field(init=False, default=None)

    def run(self) -> None:
        args = self.args
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.monotonic()
                self.train(epoch=epoch)
                val_loss = self.evaluate()
                print("-" * 89)
                print(
                    "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                    "valid ppl {:8.2f}".format(
                        epoch, (time.monotonic() - epoch_start_time), val_loss, math.exp(val_loss)
                    )
                )
                print("-" * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not self.best_val_loss or val_loss < self.best_val_loss:
                    with args.save.open("wb") as f:
                        torch.save(self.model, f)
                    self.best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the val dataset.
                    self.curr_lr /= 4.0
        except KeyboardInterrupt:
            print("-" * 89)
            print("Exiting from training early")

        # Load the best saved model.
        with args.save.open("rb") as f:
            model = torch.load(f)

        # Run on test data.
        test_loss = self.evaluate(use_test_data=True)
        print("=" * 89)
        print(
            "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
                test_loss, math.exp(test_loss)
            )
        )
        print("=" * 89)

        if args.onnx_export is not None:
            # Export the model in ONNX format.
            export_onnx(
                model, args.onnx_export, batch_size=1, seq_len=args.bptt, device=args.device
            )

    def train(self, *, epoch: int) -> None:
        args = self.args
        # Turn on training mode which enables dropout.
        self.model.train()
        total_loss = 0.0
        start_time = time.monotonic()
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, args.bptt)):
            data, targets = get_batch(self.train_data, i, bptt=args.bptt)

            self.model.zero_grad()
            output = self.model(data, use_mask=True)
            output = output.view(-1, self.ntokens)
            loss = self.criterion(output, targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)

            # Manual implementation of SGD.
            # TODO: Use `torch.optim` with a proper learning rate scheduler instead.
            for p in self.model.parameters():
                p.data.add_(p.grad, alpha=-self.curr_lr)

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.monotonic() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                    "loss {:5.2f} | ppl {:8.2f}".format(
                        epoch,
                        batch,
                        len(self.train_data) // args.bptt,
                        self.curr_lr,
                        elapsed * 1000 / args.log_interval,
                        cur_loss,
                        math.exp(cur_loss),  # perplexity
                    )
                )
                total_loss = 0
                start_time = time.monotonic()
            if args.dry_run:
                break

    def evaluate(self, *, use_test_data: bool = False) -> float:
        args = self.args
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        data_source = self.test_data if use_test_data else self.val_data
        total_loss = 0.0
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i, bptt=args.bptt)
                output = self.model(data, use_mask=True)
                output = output.view(-1, self.ntokens)
                total_loss += len(data) * self.criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)


def export_onnx(
    model: TransformerModel, path: Path, batch_size: int, seq_len: int, device: torch.device
) -> None:
    print("The model is also exported in ONNX format at {}.".format(path.resolve()))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    torch.onnx.export(model, dummy_input, str(path))


def batchify(data: Tensor, bsz: int, device: torch.device) -> Tensor:
    """Starting from sequential data, batchify arranges the dataset into columns.

    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    batch processing.
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, i: int, bptt: int) -> tuple[Tensor, Tensor]:
    """Subdivide the source data into chunks of length args.bptt.

    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].view(-1)
    return data, target


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=TrainConfig)
    main()
