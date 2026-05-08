mod data;
mod model;

use burn::prelude::*;
use burn_ndarray::NdArray;

use data::{fetch_tiny_shakespeare, TextDataset, Vocabulary};
use burn::data::dataset::Dataset;
use model::{Embeddings, GptConfig};

type B = NdArray;

fn main() {
    let device = Default::default();

    // --- Data pipeline ---
    let text = fetch_tiny_shakespeare();
    let vocab = Vocabulary::from_text(&text);
    let tokens = vocab.encode(&text);

    println!("Text length  : {} chars", text.len());
    println!("Vocab size   : {} unique chars", vocab.size());

    let block_size = 256;
    let (train, _val) = TextDataset::train_val_split(tokens, block_size, 0.1);

    // --- Embeddings smoke test ---
    let config = GptConfig::new(vocab.size(), block_size, 128, 4, 4);
    let embeddings = Embeddings::<B>::new(&config, &device);

    let sample = train.get(0).unwrap();
    let tokens: Tensor<B, 2, Int> = Tensor::from_data(
        TensorData::new(sample.input, [1, block_size]),
        &device,
    );

    let embedded = embeddings.forward(tokens);
    println!("Embedding output shape: {:?}", embedded.dims());
    // expected: [1, 256, 128]
}
