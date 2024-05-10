# Word-level Language Modeling using Transformer

This example trains a Transformer on a language modeling task. By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py cuda=true epochs=6 lr=5  # Train a Transformer on Wikitext-2 with CUDA.

python generate.py                      # Generate samples from the default model checkpoint.
```

The model uses the Transformer module (`nn.TransformerEncoder` and `nn.TransformerEncoderLayer`) which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received, training is stopped and the current model is evaluated against the test dataset.

Run

```bash
python main.py --help
```
to see all available arguments.
