# nano-gpt

A character-level GPT built in Rust using the [Burn](https://github.com/tracel-ai/burn) framework. Trains a decoder-only Transformer to generate text in the style of William Shakespeare.

## Architecture

- **Embeddings** — token + positional lookup tables
- **Transformer blocks** — masked multi-head self-attention, feed-forward network, layer norm, residual connections
- **Language model head** — linear projection to vocabulary logits

Current config: `block_size=256, n_embd=256, n_head=8, n_layer=6` (~4.8M parameters)

## Usage

### Train

```
cargo run --bin train                       # default 3000 iterations
cargo run --bin train -- --iters 5000       # custom iteration count
```

Downloads [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) on first run (~1MB). Trains on CPU or GPU via the `wgpu` backend (Metal on macOS, Vulkan/DX12 elsewhere). Saves the trained weights to `artifacts/model.bin` and config to `artifacts/config.json` when done.

### Generate

```
cargo run --bin generate
cargo run --bin generate -- --temperature 0.5   # more focused
cargo run --bin generate -- --temperature 1.2   # more random
```

Loads the saved checkpoint from `artifacts/` and generates text instantly — no retraining needed. Run `train` at least once first. Temperature defaults to `0.8`; lower values produce more predictable text, higher values more creative (and chaotic) output.

## Sample output after 3000 iterations

```
First Citizen:
And I say, the first Coriolanus
With nothing so much to bear the declamOnd.

BONTAGOT:
Well said 'I can say 'sir,' become mock:
I would have no grace soldiers.

CAMILLO:
To sleech you to the chase is anguish on:
We will not stay it of; for 'tis it so.

POLIXENES:
Here's a lover spine for his love.
```

After 3000 iterations (loss 0.98) the model has learned:
- Speaker label format (`NAME:` followed by a newline)
- Real Shakespeare character names — CAMILLO, POLIXENES, CORIOLANUS
- Elizabethan vocabulary and contractions — "'tis", "'sir'"
- Coherent sentence structure and punctuation patterns
