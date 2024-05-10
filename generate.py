###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

import data
from config import GenerateConfig


@hydra.main(config_path="conf", config_name="generate", version_base="1.3")
def main(hydra_config: DictConfig) -> None:
    args = instantiate(hydra_config, _convert_="object")
    assert isinstance(args, GenerateConfig)

    if args.temperature < 1e-3:
        print("`temperature` has to be greater or equal 1e-3.")
        return

    with open(args.checkpoint, "rb") as f:
        model = torch.load(f, map_location=args.device)
    model.eval()

    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)

    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(args.device)

    with open(args.outf, "w") as outf:
        with torch.no_grad():  # no tracking history
            for i in range(args.words):
                output = model(input, has_mask=False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(args.device)
                input = torch.cat([input, word_tensor], 0)

                word = corpus.dictionary.idx2word[word_idx]

                outf.write(word + ("\n" if i % 20 == 19 else " "))

                if i % args.log_interval == 0:
                    print("| Generated {}/{} words".format(i, args.words))


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=GenerateConfig)
    main()
