use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation;
use rand::Rng;

use crate::data::{TextBatcher, TextDataset, Vocabulary};
use crate::model::{Gpt, GptConfig};

// ---------- Config ----------------------------------------------------------

pub struct TrainConfig {
    pub max_iters: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub eval_interval: usize,
    pub generate_length: usize,
    pub temperature: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_iters: 3000,
            batch_size: 16,
            learning_rate: 1e-3,
            eval_interval: 500,
            generate_length: 300,
            temperature: 0.8,
        }
    }
}

// ---------- Training loop ---------------------------------------------------

pub fn train<B: AutodiffBackend>(
    config: &TrainConfig,
    gpt_config: &GptConfig,
    vocab: &Vocabulary,
    tokens: Vec<i64>,
    device: &B::Device,
) {
    let (train_set, _) = TextDataset::train_val_split(tokens, gpt_config.block_size, 0.1);

    let mut model = gpt_config.init::<B>(device);
    let mut optim = AdamWConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let batcher = TextBatcher::<B>::new(device.clone());

    println!(
        "Training {} parameters for {} iterations...\n",
        model.num_params(),
        config.max_iters
    );

    let mut rng = rand::thread_rng();

    for iter in 0..=config.max_iters {
        // Sample a random batch from the training set
        let items: Vec<_> = (0..config.batch_size)
            .map(|_| train_set.get(rng.gen_range(0..train_set.len())).unwrap())
            .collect();
        let batch = batcher.batch(items, device);

        // Forward pass → logits: [batch, seq_len, vocab_size]
        let logits = model.forward(batch.inputs);
        let [batch_size, seq_len, vocab_size] = logits.dims();

        // Flatten to [batch*seq_len, vocab_size] and [batch*seq_len] for cross-entropy
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);

        let loss = loss_fn.forward(logits_flat, targets_flat);

        if iter % config.eval_interval == 0 {
            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            println!("iter {:4} | loss {:.4}", iter, loss_val);
        }

        // Backward pass + parameter update
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(config.learning_rate, model, grads);
    }

    // Generate sample text with the trained model
    println!("\n--- Generated text ---\n");
    let generated = generate(
        &model,
        vocab,
        "First Citizen:\n",
        config.generate_length,
        config.temperature,
        gpt_config.block_size,
        device,
    );
    println!("{}", generated);

    // Save weights so `cargo run --bin generate` can reload them without retraining
    std::fs::create_dir_all("artifacts").expect("failed to create artifacts dir");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file("artifacts/model", &recorder)
        .expect("failed to save checkpoint");
    gpt_config
        .save("artifacts/config.json")
        .expect("failed to save config");
    println!("\nCheckpoint saved to artifacts/");
}

// ---------- Text generation -------------------------------------------------

/// Autoregressively samples `length` new characters given a seed string.
pub fn generate<B: Backend>(
    model: &Gpt<B>,
    vocab: &Vocabulary,
    seed: &str,
    length: usize,
    temperature: f32,
    block_size: usize,
    device: &B::Device,
) -> String {
    let mut context: Vec<i64> = vocab.encode(seed);

    for _ in 0..length {
        // Keep context within the model's maximum sequence length
        let start = context.len().saturating_sub(block_size);
        let input = &context[start..];
        let seq_len = input.len();

        let tokens: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(input.to_vec(), [1, seq_len]),
            device,
        );

        // Forward pass — only the last position's logits are used for generation
        let logits = model.forward(tokens); // [1, seq_len, vocab_size]
        let [_, _, vocab_size] = logits.dims();

        // Extract last position: [1, seq_len, vocab_size] → [vocab_size]
        let last_logits = logits
            .slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size])
            .reshape([vocab_size]);

        // Scale by temperature, apply softmax to get a probability distribution
        let probs = activation::softmax(last_logits.div_scalar(temperature).unsqueeze::<2>(), 1)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        context.push(sample_multinomial(&probs) as i64);
    }

    vocab.decode(&context)
}

// Sample one index from a probability distribution using weighted random selection
fn sample_multinomial(probs: &[f32]) -> usize {
    use rand::distributions::{Distribution, WeightedIndex};
    let dist = WeightedIndex::new(probs).expect("invalid probability distribution");
    dist.sample(&mut rand::thread_rng())
}
