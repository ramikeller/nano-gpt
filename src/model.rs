use burn::{config::Config, module::Module, nn, prelude::*, tensor::activation};

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

        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>();

        let tok = self.token_emb.forward(tokens);  // [batch, seq_len, n_embd]
        let pos = self.pos_emb.forward(positions); // [1,     seq_len, n_embd]

        tok + pos
    }
}

// ---------- Causal Self-Attention -------------------------------------------

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    q_proj: nn::Linear<B>,
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    out_proj: nn::Linear<B>,
    dropout: nn::Dropout,
    // Primitive fields must be skipped by the Module derive; they are not
    // trainable parameters — just shape metadata needed at runtime.
    #[module(skip)]
    n_head: usize,
    #[module(skip)]
    head_size: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        assert!(
            config.n_embd % config.n_head == 0,
            "n_embd ({}) must be divisible by n_head ({})",
            config.n_embd,
            config.n_head
        );
        Self {
            // No bias on Q/K/V projections — standard GPT convention
            q_proj: nn::LinearConfig::new(config.n_embd, config.n_embd)
                .with_bias(false)
                .init(device),
            k_proj: nn::LinearConfig::new(config.n_embd, config.n_embd)
                .with_bias(false)
                .init(device),
            v_proj: nn::LinearConfig::new(config.n_embd, config.n_embd)
                .with_bias(false)
                .init(device),
            out_proj: nn::LinearConfig::new(config.n_embd, config.n_embd).init(device),
            dropout: nn::DropoutConfig::new(config.dropout).init(),
            n_head: config.n_head,
            head_size: config.n_embd / config.n_head,
        }
    }

    /// tokens: [batch, seq_len, n_embd]  →  [batch, seq_len, n_embd]
    pub fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _n_embd] = tokens.dims();
        let device = tokens.device();

        // Project to queries, keys, values
        let q = self.q_proj.forward(tokens.clone()); // [batch, seq_len, n_embd]
        let k = self.k_proj.forward(tokens.clone()); // [batch, seq_len, n_embd]
        let v = self.v_proj.forward(tokens);         // [batch, seq_len, n_embd]

        // Split embedding dim across heads and move head dim before seq dim:
        // [batch, seq_len, n_embd] → [batch, n_head, seq_len, head_size]
        let q = q.reshape([batch, seq_len, self.n_head, self.head_size]).swap_dims(1, 2);
        let k = k.reshape([batch, seq_len, self.n_head, self.head_size]).swap_dims(1, 2);
        let v = v.reshape([batch, seq_len, self.n_head, self.head_size]).swap_dims(1, 2);

        // Scaled dot-product attention scores: [batch, n_head, seq_len, seq_len]
        let scale = (self.head_size as f32).sqrt();
        let attn = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);

        // Causal mask: fill positions where col > row with -∞ so softmax → 0
        let rows = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .reshape([seq_len, 1]);
        let cols = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .reshape([1, seq_len]);
        let mask = cols.greater(rows).unsqueeze::<4>(); // [1, 1, seq_len, seq_len]
        let attn = attn.mask_fill(mask, f32::NEG_INFINITY);

        let attn = activation::softmax(attn, 3);
        let attn = self.dropout.forward(attn);

        // Weighted sum then merge heads back:
        // [batch, n_head, seq_len, head_size] → [batch, seq_len, n_embd]
        let out = attn
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.n_head * self.head_size]);

        self.out_proj.forward(out)
    }
}

// ---------- Feed-Forward Network --------------------------------------------

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    dropout: nn::Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        Self {
            // Expand to 4× n_embd then project back — standard GPT ratio
            fc1: nn::LinearConfig::new(config.n_embd, 4 * config.n_embd).init(device),
            fc2: nn::LinearConfig::new(4 * config.n_embd, config.n_embd).init(device),
            dropout: nn::DropoutConfig::new(config.dropout).init(),
        }
    }

    /// x: [batch, seq_len, n_embd]  →  [batch, seq_len, n_embd]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = activation::gelu(x);
        let x = self.fc2.forward(x);
        self.dropout.forward(x)
    }
}

// ---------- Transformer Block -----------------------------------------------

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    ln1: nn::LayerNorm<B>,
    attn: CausalSelfAttention<B>,
    ln2: nn::LayerNorm<B>,
    ffn: FeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &GptConfig, device: &B::Device) -> Self {
        Self {
            ln1: nn::LayerNormConfig::new(config.n_embd).init(device),
            attn: CausalSelfAttention::new(config, device),
            ln2: nn::LayerNormConfig::new(config.n_embd).init(device),
            ffn: FeedForward::new(config, device),
        }
    }

    /// x: [batch, seq_len, n_embd]  →  [batch, seq_len, n_embd]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm attention with residual
        let x = x.clone() + self.attn.forward(self.ln1.forward(x));
        // Pre-norm feed-forward with residual
        x.clone() + self.ffn.forward(self.ln2.forward(x))
    }
}
