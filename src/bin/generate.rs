use burn::prelude::{Config, Module};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn_wgpu::Wgpu;

use nano_gpt::data::{fetch_tiny_shakespeare, Vocabulary};
use nano_gpt::model::GptConfig;
use nano_gpt::training::generate;

// No Autodiff wrapper needed — inference only
type B = Wgpu;

fn main() {
    let temperature = parse_temperature();

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
        generate(&model, &vocab, "First Citizen:\n", 500, temperature, gpt_config.block_size, &device)
    );
}

fn parse_temperature() -> f32 {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--temperature" {
            let val = args.get(i + 1).expect("--temperature requires a value");
            return val.parse().expect("--temperature must be a number (e.g. 0.8)");
        }
        i += 1;
    }
    0.8 // default
}
