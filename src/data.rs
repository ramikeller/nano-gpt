use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;

const DATA_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
const DATA_PATH: &str = "data/input.txt";

// ---------- Download --------------------------------------------------------

pub fn fetch_tiny_shakespeare() -> String {
    if !Path::new(DATA_PATH).exists() {
        fs::create_dir_all("data").expect("failed to create data/");
        println!("Downloading tinyshakespeare...");
        let status = Command::new("curl")
            .args(["-L", "-o", DATA_PATH, DATA_URL])
            .status()
            .expect("curl not found — install curl or place the file at data/input.txt manually");
        assert!(status.success(), "download failed");
        println!("Done.");
    }
    fs::read_to_string(DATA_PATH).expect("failed to read data/input.txt")
}

// ---------- Vocabulary ------------------------------------------------------

pub struct Vocabulary {
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: Vec<char>,
}

impl Vocabulary {
    pub fn from_text(text: &str) -> Self {
        let chars: Vec<char> = {
            let mut set = std::collections::HashSet::new();
            text.chars().for_each(|c| {
                set.insert(c);
            });
            let mut v: Vec<char> = set.into_iter().collect();
            v.sort();
            v
        };
        let char_to_idx = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        Self {
            char_to_idx,
            idx_to_char: chars,
        }
    }

    pub fn size(&self) -> usize {
        self.idx_to_char.len()
    }

    pub fn encode(&self, text: &str) -> Vec<i64> {
        text.chars().map(|c| self.char_to_idx[&c] as i64).collect()
    }

    pub fn decode(&self, indices: &[i64]) -> String {
        indices
            .iter()
            .map(|&i| self.idx_to_char[i as usize])
            .collect()
    }
}

// ---------- Dataset ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TextSample {
    pub input: Vec<i64>,
    pub target: Vec<i64>,
}

pub struct TextDataset {
    tokens: Vec<i64>,
    block_size: usize,
}

impl TextDataset {
    /// Split the full token sequence into train and validation sets.
    pub fn train_val_split(
        tokens: Vec<i64>,
        block_size: usize,
        val_fraction: f64,
    ) -> (Self, Self) {
        let split = (tokens.len() as f64 * (1.0 - val_fraction)) as usize;
        let val_tokens = tokens[split..].to_vec();
        let train_tokens = tokens[..split].to_vec();
        (
            Self { tokens: train_tokens, block_size },
            Self { tokens: val_tokens, block_size },
        )
    }
}

impl Dataset<TextSample> for TextDataset {
    fn get(&self, index: usize) -> Option<TextSample> {
        let end = index + self.block_size + 1;
        if end > self.tokens.len() {
            return None;
        }
        Some(TextSample {
            input: self.tokens[index..index + self.block_size].to_vec(),
            target: self.tokens[index + 1..end].to_vec(),
        })
    }

    fn len(&self) -> usize {
        self.tokens.len().saturating_sub(self.block_size)
    }
}

// ---------- Batcher ---------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

#[derive(Clone)]
pub struct TextBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TextBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, TextSample, TextBatch<B>> for TextBatcher<B> {
    fn batch(&self, items: Vec<TextSample>, _device: &B::Device) -> TextBatch<B> {
        let batch_size = items.len();
        let block_size = items[0].input.len();

        let flat_inputs: Vec<i64> = items
            .iter()
            .flat_map(|s| s.input.iter().copied())
            .collect();
        let flat_targets: Vec<i64> = items
            .iter()
            .flat_map(|s| s.target.iter().copied())
            .collect();

        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(flat_inputs, [batch_size, block_size]),
            &self.device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(flat_targets, [batch_size, block_size]),
            &self.device,
        );

        TextBatch { inputs, targets }
    }
}
