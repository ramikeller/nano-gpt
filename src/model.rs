use burn::{config::Config, module::Module, nn, prelude::*};

// ---------- Hyperparameters -------------------------------------------------

#[derive(Config, Debug)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    #[config(default = "0.1")]
    pub dropout: f64,
}

// ---------- Embeddings ------------------------------------------------------

#[derive(Module, Debug)]
pub struct Embeddings<B: Backend> {
    token_emb: nn::Embedding<B>,
    pos_emb: nn::Embedding<B>,
}

impl<B: Backend> Embeddings<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        Self {
            token_emb: nn::EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device),
            pos_emb: nn::EmbeddingConfig::new(config.block_size, config.n_embd).init(device),
        }
    }

    /// tokens: [batch, seq_len]  →  [batch, seq_len, n_embd]
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch, seq_len] = tokens.dims();
        let device = tokens.device();

        // [0, 1, ..., seq_len-1] then unsqueeze to [1, seq_len] for broadcasting
        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>();

        let tok = self.token_emb.forward(tokens);  // [batch, seq_len, n_embd]
        let pos = self.pos_emb.forward(positions); // [1,     seq_len, n_embd]

        tok + pos
    }
}
