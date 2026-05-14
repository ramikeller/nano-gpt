use burn::backend::Autodiff;
use burn_wgpu::Wgpu;

use nano_gpt::data::{fetch_tiny_shakespeare, Vocabulary};
use nano_gpt::model::GptConfig;
use nano_gpt::training::{train, TrainConfig};

// Autodiff<Wgpu> runs on Metal (macOS) / Vulkan / DX12 via the wgpu backend
type B = Autodiff<Wgpu>;

fn main() {
    let device = Default::default();

    let text = fetch_tiny_shakespeare();
    let vocab = Vocabulary::from_text(&text);
    let tokens = vocab.encode(&text);

    let gpt_config = GptConfig::new(vocab.size(), 256, 256, 8, 6);
    //                               vocab  block  n_embd heads layers
    //                               65     256    256    8     6

    let train_config = TrainConfig {
        max_iters: 3000,
        batch_size: 64,
        eval_interval: 50,
        ..TrainConfig::default()
    };

    train::<B>(&train_config, &gpt_config, &vocab, tokens, &device);
}
