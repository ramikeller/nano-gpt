mod data;
mod model;

use burn::prelude::*;
use burn_ndarray::NdArray;

use burn::data::dataset::Dataset;
use data::{fetch_tiny_shakespeare, TextDataset, Vocabulary};
use model::{CausalSelfAttention, Embeddings, GptConfig};

type B = NdArray;

fn main() {
    let device = Default::default();

    let text = fetch_tiny_shakespeare();
    let vocab = Vocabulary::from_text(&text);
    let tokens = vocab.encode(&text);

    let block_size = 256;
    let (train, _val) = TextDataset::train_val_split(tokens, block_size, 0.1);

    let config = GptConfig::new(vocab.size(), block_size, 128, 1, 1);
    let embeddings = Embeddings::<B>::new(&config, &device);
    let attention = CausalSelfAttention::<B>::new(&config, &device);

    let sample = train.get(0).unwrap();
    let input_text = vocab.decode(&sample.input);
    let target_text = vocab.decode(&sample.target);
    println!("input  : {:?}", &input_text[..40]);
    println!("target : {:?}", &target_text[..40]);
    println!("indices: {:?}", &sample.input[..8]);

    let tensor_tokens: Tensor<B, 2, Int> = Tensor::from_data(
        TensorData::new(sample.input, [1, block_size]),
        &device,
    );

    let tensor_embedded = embeddings.forward(tensor_tokens);    // [1, 256, 128]
    let tensor_attended = attention.forward(tensor_embedded);   // [1, 256, 128]
    println!("Attention output shape: {:?}", tensor_attended.dims());
    // expected: [1, 256, 128]  — same shape in, same shape out
}
