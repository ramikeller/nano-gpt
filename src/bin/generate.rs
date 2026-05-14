use burn::prelude::{Config, Module};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn_wgpu::Wgpu;

use nano_gpt::data::{fetch_tiny_shakespeare, Vocabulary};
use nano_gpt::model::GptConfig;
use nano_gpt::training::generate;

// No Autodiff wrapper needed — inference only
type B = Wgpu;

fn main() {
    let device = Default::default();

    let text = fetch_tiny_shakespeare();
    let vocab = Vocabulary::from_text(&text);

    let gpt_config =
        GptConfig::load("artifacts/config.json").expect(
            "config not found — run `cargo run --bin train` first",
        );

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = gpt_config
        .init::<B>(&device)
        .load_file("artifacts/model", &recorder, &device)
        .expect("weights not found — run `cargo run --bin train` first");

    println!(
        "{}",
        generate(&model, &vocab, "First Citizen:\n", 500, 0.8, gpt_config.block_size, &device)
    );
}
