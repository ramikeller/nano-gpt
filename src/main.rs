mod data;
mod model;
mod training;

use burn::backend::Autodiff;
use burn_ndarray::NdArray;

use data::{fetch_tiny_shakespeare, Vocabulary};
use model::GptConfig;
use training::{train, TrainConfig};

// Autodiff<NdArray> wraps the CPU backend with gradient tracking for training
type B = Autodiff<NdArray>;

fn main() {
    let device = Default::default();

    let text = fetch_tiny_shakespeare();
    let vocab = Vocabulary::from_text(&text);
    let tokens = vocab.encode(&text);

    let gpt_config = GptConfig::new(vocab.size(), 64, 64, 4, 2);
    //                               vocab  block n_embd heads layers
    //                               65     64    64     4     2

    let train_config = TrainConfig::default();

    train::<B>(&train_config, &gpt_config, &vocab, tokens, &device);
}
