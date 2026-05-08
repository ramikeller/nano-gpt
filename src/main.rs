mod data;

use data::{fetch_shakespeare, TextDataset, Vocabulary};
use burn::data::dataset::Dataset;

fn main() {
    let text = fetch_shakespeare();
    let vocab = Vocabulary::from_text(&text);
    let tokens = vocab.encode(&text);

    println!("Text length  : {} chars", text.len());
    println!("Vocab size   : {} unique chars", vocab.size());
    println!("Token count  : {}", tokens.len());

    let block_size = 256;
    let (train, val) = TextDataset::train_val_split(tokens, block_size, 0.1);
    println!("Train samples: {}", train.len());
    println!("Val samples  : {}", val.len());

    // Peek at one sample
    let sample = train.get(0).unwrap();
    let input_text = vocab.decode(&sample.input);
    let target_text = vocab.decode(&sample.target);
    println!("\nInput  (first 40 chars): {:?}", &input_text[..40]);
    println!("Target (first 40 chars): {:?}", &target_text[..40]);
}
