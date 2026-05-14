# nano-gpt

A character-level GPT built in Rust using the [Burn](https://github.com/tracel-ai/burn) framework. Trains a decoder-only Transformer to generate text in the style of William Shakespeare.

## Architecture

- **Embeddings** — token + positional lookup tables
- **Transformer blocks** — masked multi-head self-attention, feed-forward network, layer norm, residual connections
- **Language model head** — linear projection to vocabulary logits

Current config: `block_size=256, n_embd=256, n_head=8, n_layer=6` (~4.8M parameters)

## Training

```
cargo run
```

Downloads [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) on first run (~1MB). Trains on CPU or GPU via the `wgpu` backend (Metal on macOS, Vulkan/DX12 elsewhere).

## Sample output after 1000 iterations

```
First Citizen:
Shall may I have not so with on the stres and supping.

CLAUDIO:
Fie that I who shall.

CLAURENCE:
Desperit him to show him mooth a king.

AUNTIO:
I have she beseem'd:
I would are the consul of at friend subjects
Against they hidst with buck by in the blood.

LUCIO:
A look unto fools, and face be pi
```

After 1000 iterations the model has learned:
- Speaker label format (`NAME:` followed by a newline)
- Real Shakespeare character names — CLAUDIO, LUCIO, CLARENCE
- Elizabethan vocabulary — "beseem'd", "consul", "hidst", "fools"
- Iambic-ish line breaks and punctuation patterns
