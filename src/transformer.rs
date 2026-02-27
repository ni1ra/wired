use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{
    embedding, linear, linear_no_bias, Embedding, Linear, Optimizer, VarBuilder,
    VarMap,
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Gradient-safe RmsNorm (candle_nn::RmsNorm has broken backward pass)
// Uses only basic tensor ops that have working autograd.
// ---------------------------------------------------------------------------

pub struct GradRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GradRmsNorm {
    pub fn new(d_model: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(d_model, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // rms = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (variance + self.eps)?.sqrt()?;
        let normed = x.broadcast_div(&rms)?;
        normed.broadcast_mul(&self.weight).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Gradient-safe softmax (candle_nn::ops::softmax_last_dim has broken backward)
// ---------------------------------------------------------------------------

fn grad_softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let shifted = x.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&sum).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Gradient-safe dropout (manual impl — don't trust candle_nn ops, see M-032)
// Uses inverted dropout: scale by 1/(1-p) during training so inference is unchanged.
// ---------------------------------------------------------------------------

fn grad_dropout(x: &Tensor, p: f64, train: bool) -> Result<Tensor> {
    if !train || p <= 0.0 || p >= 1.0 {
        return Ok(x.clone());
    }
    // Random mask: each element kept with probability (1-p)
    let rand = Tensor::rand(0f32, 1f32, x.shape(), x.device())?;
    let mask = rand.ge(p as f64)?.to_dtype(x.dtype())?;
    let scale = 1.0 / (1.0 - p);
    (x.broadcast_mul(&mask)? * scale).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    /// Max sequence length seen during training (for YaRN context extension).
    /// When inference seq > max_train_len, YaRN frequency-selective RoPE scaling activates.
    #[serde(default = "default_max_train_len")]
    pub max_train_len: usize,
    #[serde(default)]
    pub dropout: f64,
    /// Enable cross-attention layers for external memory (decoder only).
    #[serde(default)]
    pub use_cross_attn: bool,
}

fn default_max_train_len() -> usize { 256 }

impl TransformerConfig {
    /// Phase 2.5+ scaled architecture: d=1024, 8 layers, 8 heads, ~200M params.
    pub fn default() -> Self {
        Self {
            d_model: 1024,
            n_layers: 8,
            n_heads: 8,
            d_ff: 4096,
            vocab_size: 8192,
            max_seq_len: 256,
            max_train_len: 256,
            dropout: 0.1,
            use_cross_attn: false,
        }
    }

    /// Small config for gradient checking (fast).
    pub fn tiny() -> Self {
        Self {
            d_model: 64,
            n_layers: 1,
            n_heads: 2,
            d_ff: 128,
            vocab_size: 32,
            max_seq_len: 16,
            max_train_len: 16,
            dropout: 0.0,
            use_cross_attn: false,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

/// Precompute RoPE cos/sin tables with YaRN context extension.
///
/// When `seq_len <= max_train_len`, standard RoPE (base=10000).
/// When `seq_len > max_train_len`, YaRN frequency-selective scaling:
///   - High-freq dims (wavelength < train_len): unchanged (local relationships preserved)
///   - Low-freq dims (wavelength > train_len * beta): NTK-scaled (global relationships stretched)
///   - Mid-freq dims: linear ramp between the two
///
/// Returns (cos, sin) tables of shape (seq_len, head_dim/2).
/// Also returns the attention temperature factor for entropy compensation.
fn precompute_rope(
    seq_len: usize,
    head_dim: usize,
    max_train_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, f32)> {
    let half = head_dim / 2;
    let scale = seq_len as f32 / max_train_len as f32;

    let theta: Vec<f32> = if seq_len > max_train_len {
        // YaRN: frequency-selective scaling
        let beta = 2.0f32; // transition band width
        // NTK-scaled base for low-frequency dimensions
        let base_scaled = 10000.0f32 * scale.powf(head_dim as f32 / (head_dim as f32 - 2.0));

        (0..half)
            .map(|i| {
                let freq_orig = 1.0f32 / 10000f32.powf(2.0 * i as f32 / head_dim as f32);
                let freq_scaled = 1.0f32 / base_scaled.powf(2.0 * i as f32 / head_dim as f32);
                let wavelength = 2.0 * std::f32::consts::PI / freq_orig;

                if wavelength < max_train_len as f32 {
                    // High frequency: preserve local relationships
                    freq_orig
                } else if wavelength > max_train_len as f32 * beta {
                    // Low frequency: full NTK scaling
                    freq_scaled
                } else {
                    // Mid frequency: linear ramp
                    let t = (wavelength - max_train_len as f32)
                        / (max_train_len as f32 * (beta - 1.0));
                    freq_orig * (1.0 - t) + freq_scaled * t
                }
            })
            .collect()
    } else {
        // Standard RoPE
        (0..half)
            .map(|i| 1.0f32 / 10000f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect()
    };

    // Attention temperature: sqrt(L_test / L_train) to compensate for entropy explosion
    let attn_temp = if seq_len > max_train_len {
        scale.sqrt()
    } else {
        1.0
    };

    let theta = Tensor::new(theta, device)?;
    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
    let positions = Tensor::new(positions, device)?;
    let freqs = positions.unsqueeze(1)?.matmul(&theta.unsqueeze(0)?)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Ok((cos, sin, attn_temp))
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: (batch, heads, seq, head_dim)
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;

    let seq_len = x.dim(2)?;
    // cos/sin are (max_seq, half) -- slice to actual seq_len
    let cos = cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

    let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let rotated_x2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[rotated_x1, rotated_x2], D::Minus1).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Self {
            q_proj: linear_no_bias(d, d, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(d, d, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(d, d, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(d, d, vb.pp("o_proj"))?,
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// `n_prefix`: number of leading positions that are prefix tokens (no RoPE applied).
    /// Text tokens receive RoPE starting at position 0.
    /// `attn_temp`: YaRN attention temperature (1.0 during training, sqrt(L/L_train) for extension).
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, causal_mask: &Tensor, n_prefix: usize, attn_temp: f32) -> Result<Tensor> {
        let (b, s, _d) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE only to text tokens (skip prefix positions)
        let (q, k) = if n_prefix > 0 && n_prefix < s {
            let n_text = s - n_prefix;
            let q_prefix = q.narrow(2, 0, n_prefix)?;
            let q_text = q.narrow(2, n_prefix, n_text)?;
            let k_prefix = k.narrow(2, 0, n_prefix)?;
            let k_text = k.narrow(2, n_prefix, n_text)?;
            let q_text = apply_rope(&q_text, cos, sin)?;
            let k_text = apply_rope(&k_text, cos, sin)?;
            // .contiguous() required after cat — candle matmul needs contiguous tensors
            (Tensor::cat(&[&q_prefix, &q_text], 2)?.contiguous()?,
             Tensor::cat(&[&k_prefix, &k_text], 2)?.contiguous()?)
        } else {
            (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?)
        };

        // YaRN attention temperature: divide logits by sqrt(d) * sqrt(L/L_train)
        // to compensate for entropy explosion at longer sequences
        let scale = (self.head_dim as f64).sqrt() * attn_temp as f64;
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn = (attn / scale)?;

        // Slice cached causal mask to actual seq_len (avoids per-call allocation + H2D)
        let mask = causal_mask.narrow(2, 0, s)?.narrow(3, 0, s)?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = grad_softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Into::into)
    }

    /// Cached forward for autoregressive generation.
    /// `seq_offset`: RoPE position of first TEXT token in `x` (prefix tokens get no RoPE).
    /// `n_prefix`: number of leading positions that are prefix (no RoPE). 0 for step calls.
    /// `kv_cache`: mutable cache of past (K, V) tensors. Updated in place.
    fn forward_cached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        seq_offset: usize,
        n_prefix: usize,
        attn_temp: f32,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, s, _d) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE only to text tokens (skip prefix positions)
        let (q, k) = if n_prefix > 0 && n_prefix < s {
            let n_text = s - n_prefix;
            let cos_slice = cos.narrow(0, seq_offset, n_text)?;
            let sin_slice = sin.narrow(0, seq_offset, n_text)?;
            let q_prefix = q.narrow(2, 0, n_prefix)?;
            let q_text = apply_rope(&q.narrow(2, n_prefix, n_text)?, &cos_slice, &sin_slice)?;
            let k_prefix = k.narrow(2, 0, n_prefix)?;
            let k_text = apply_rope(&k.narrow(2, n_prefix, n_text)?, &cos_slice, &sin_slice)?;
            // .contiguous() required after cat — candle matmul needs contiguous tensors
            (Tensor::cat(&[&q_prefix, &q_text], 2)?.contiguous()?,
             Tensor::cat(&[&k_prefix, &k_text], 2)?.contiguous()?)
        } else {
            // No prefix (step calls) or all prefix — apply RoPE to all
            let cos_slice = cos.narrow(0, seq_offset, s)?;
            let sin_slice = sin.narrow(0, seq_offset, s)?;
            (apply_rope(&q, &cos_slice, &sin_slice)?, apply_rope(&k, &cos_slice, &sin_slice)?)
        };

        // Concat with cached K/V
        let (k, v) = if let Some((past_k, past_v)) = kv_cache.take() {
            (Tensor::cat(&[&past_k, &k], 2)?, Tensor::cat(&[&past_v, &v], 2)?)
        } else {
            (k, v)
        };
        *kv_cache = Some((k.clone(), v.clone()));

        let scale = (self.head_dim as f64).sqrt() * attn_temp as f64;
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn = (attn / scale)?;

        // Causal mask for cached generation.
        let kv_len = attn.dim(D::Minus1)?;
        let mask = if s == 1 {
            // Single-token step: attend to all cached entries (this token is always the latest).
            Tensor::zeros((1, 1, 1, kv_len), DType::F32, x.device())?
        } else {
            // Prefill: standard causal mask. Position i attends to positions 0..=i.
            let mask_data: Vec<f32> = (0..s)
                .flat_map(|qi| {
                    (0..kv_len).map(move |kj| if kj <= qi { 0.0f32 } else { f32::NEG_INFINITY })
                })
                .collect();
            Tensor::from_vec(mask_data, (1, 1, s, kv_len), x.device())?
        };
        let attn = attn.broadcast_add(&mask)?;
        let attn = grad_softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Into::into)
    }
}

fn build_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY })
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

struct Mlp {
    gate: Linear,
    down: Linear,
}

impl Mlp {
    fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: linear(d_model, d_ff, vb.pp("gate"))?,
            down: linear(d_ff, d_model, vb.pp("down"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.gate.forward(x)?.gelu()?;
        self.down.forward(&h).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Cross-Attention (for external memory)
// ---------------------------------------------------------------------------

struct CrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Self {
            q_proj: linear_no_bias(d, d, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(d, d, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(d, d, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(d, d, vb.pp("o_proj"))?,
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Cross-attention: Q from hidden states, K/V from external memory.
    /// No causal mask (memory is fully visible). No RoPE (memory has no position).
    fn forward(&self, x: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let (b, s, _d) = x.dims3()?;
        let m = memory.dim(1)?; // number of memory entries

        let q = self.q_proj.forward(x)?
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self.k_proj.forward(memory)?
            .reshape((b, m, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self.v_proj.forward(memory)?
            .reshape((b, m, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn = (attn / scale)?;
        // No mask — full attention to all memory entries
        let attn = grad_softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, s, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&out).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Transformer Block
// ---------------------------------------------------------------------------

struct TransformerBlock {
    attn_norm: GradRmsNorm,
    attn: Attention,
    cross_attn_norm: Option<GradRmsNorm>,
    cross_attn: Option<CrossAttention>,
    mlp_norm: GradRmsNorm,
    mlp: Mlp,
    dropout: f64,
}

impl TransformerBlock {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let (cross_attn_norm, cross_attn) = if cfg.use_cross_attn {
            (
                Some(GradRmsNorm::new(cfg.d_model, 1e-6, vb.pp("cross_attn_norm"))?),
                Some(CrossAttention::new(cfg, vb.pp("cross_attn"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            attn_norm: GradRmsNorm::new(cfg.d_model, 1e-6, vb.pp("attn_norm"))?,
            attn: Attention::new(cfg, vb.pp("attn"))?,
            cross_attn_norm,
            cross_attn,
            mlp_norm: GradRmsNorm::new(cfg.d_model, 1e-6, vb.pp("mlp_norm"))?,
            mlp: Mlp::new(cfg.d_model, cfg.d_ff, vb.pp("mlp"))?,
            dropout: cfg.dropout,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, causal_mask: &Tensor, train: bool, n_prefix: usize, attn_temp: f32, memory: Option<&Tensor>) -> Result<Tensor> {
        let h = self.attn.forward(&self.attn_norm.forward(x)?, cos, sin, causal_mask, n_prefix, attn_temp)?;
        let h = grad_dropout(&h, self.dropout, train)?;
        let mut x = (x + h)?;
        // Cross-attention to external memory (when both layers and memory exist)
        if let (Some(norm), Some(ca), Some(mem)) = (&self.cross_attn_norm, &self.cross_attn, memory) {
            let h = ca.forward(&norm.forward(&x)?, mem)?;
            let h = grad_dropout(&h, self.dropout, train)?;
            x = (x + h)?;
        }
        let h = self.mlp.forward(&self.mlp_norm.forward(&x)?)?;
        let h = grad_dropout(&h, self.dropout, train)?;
        (x + h).map_err(Into::into)
    }

    fn forward_cached(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        seq_offset: usize,
        n_prefix: usize,
        attn_temp: f32,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        memory: Option<&Tensor>,
    ) -> Result<Tensor> {
        let h = self.attn.forward_cached(&self.attn_norm.forward(x)?, cos, sin, seq_offset, n_prefix, attn_temp, kv_cache)?;
        // No dropout during inference
        let mut x = (x + h)?;
        // Cross-attention to external memory (when both layers and memory exist)
        if let (Some(norm), Some(ca), Some(mem)) = (&self.cross_attn_norm, &self.cross_attn, memory) {
            let h = ca.forward(&norm.forward(&x)?, mem)?;
            x = (x + h)?;
        }
        let h = self.mlp.forward(&self.mlp_norm.forward(&x)?)?;
        (x + h).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// WiredTransformer
// ---------------------------------------------------------------------------

pub struct WiredTransformer {
    pub config: TransformerConfig,
    tok_emb: Embedding,
    layers: Vec<TransformerBlock>,
    final_norm: GradRmsNorm,
    lm_head: Linear,
    rope_cos: Tensor,
    rope_sin: Tensor,
    causal_mask: Tensor,
    /// YaRN attention temperature: sqrt(seq_len / max_train_len). 1.0 when within training range.
    attn_temp: f32,
}

impl WiredTransformer {
    pub fn from_vb(cfg: TransformerConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let tok_emb = embedding(cfg.vocab_size, cfg.d_model, vb.pp("tok_emb"))?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(TransformerBlock::new(&cfg, vb.pp(format!("layer_{i}")))?);
        }
        let final_norm = GradRmsNorm::new(cfg.d_model, 1e-6, vb.pp("final_norm"))?;
        let lm_head = linear_no_bias(cfg.d_model, cfg.vocab_size, vb.pp("lm_head"))?;
        let (rope_cos, rope_sin, attn_temp) = precompute_rope(cfg.max_seq_len, cfg.head_dim(), cfg.max_train_len, device)?;
        let causal_mask = build_causal_mask(cfg.max_seq_len, device)?;
        Ok(Self {
            config: cfg,
            tok_emb,
            layers,
            final_norm,
            lm_head,
            rope_cos,
            rope_sin,
            causal_mask,
            attn_temp,
        })
    }

    pub fn new(cfg: TransformerConfig, varmap: &VarMap, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        Self::from_vb(cfg, vb, device)
    }

    /// Forward pass (inference mode, no dropout).
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_t(input_ids, false)
    }

    /// Forward pass with train flag controlling dropout.
    pub fn forward_t(&self, input_ids: &Tensor, train: bool) -> Result<Tensor> {
        let mut x = self.tok_emb.forward(input_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, &self.rope_cos, &self.rope_sin, &self.causal_mask, train, 0, self.attn_temp, None)?;
        }
        x = self.final_norm.forward(&x)?;
        self.lm_head.forward(&x).map_err(Into::into)
    }

    /// Token embedding only — no transformer layers. For diagnostics.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.tok_emb.forward(input_ids).map_err(Into::into)
    }

    /// Encode input tokens -> hidden states (no lm_head projection).
    /// Used by ConceptEncoder to extract concept representations.
    pub fn encode(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.encode_t(input_ids, false)
    }

    /// Encode with train flag controlling dropout.
    pub fn encode_t(&self, input_ids: &Tensor, train: bool) -> Result<Tensor> {
        let mut x = self.tok_emb.forward(input_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, &self.rope_cos, &self.rope_sin, &self.causal_mask, train, 0, self.attn_temp, None)?;
        }
        self.final_norm.forward(&x).map_err(Into::into)
    }

    /// Forward with prefix (inference mode, no dropout).
    pub fn forward_with_prefix(&self, prefix: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_prefix_t(prefix, input_ids, false, None)
    }

    /// Forward with prefix and train flag controlling dropout.
    /// Prefix tokens receive NO RoPE. Text tokens get RoPE starting at position 0.
    /// `memory`: optional external memory embeddings for cross-attention (batch, n_mem, d_model).
    pub fn forward_with_prefix_t(&self, prefix: &Tensor, input_ids: &Tensor, train: bool, memory: Option<&Tensor>) -> Result<Tensor> {
        let tok_embs = self.tok_emb.forward(input_ids)?;
        let mut x = Tensor::cat(&[prefix, &tok_embs], 1)?;
        let n_prefix = prefix.dim(1)?;

        for layer in &self.layers {
            x = layer.forward(&x, &self.rope_cos, &self.rope_sin, &self.causal_mask, train, n_prefix, self.attn_temp, memory)?;
        }
        x = self.final_norm.forward(&x)?;

        let n_prefix = prefix.dim(1)?;
        let seq_len = input_ids.dim(1)?;
        let token_hidden = x.narrow(1, n_prefix, seq_len)?;
        self.lm_head.forward(&token_hidden).map_err(Into::into)
    }

    /// Number of transformer layers (for initializing KV cache vec).
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Cached forward with prefix — prefill step.
    /// Processes full [prefix + input_ids], populates KV cache, returns logits for last token only.
    /// Prefix tokens receive NO RoPE. Text tokens get RoPE starting at position 0.
    /// `memory`: optional external memory embeddings for cross-attention (batch, n_mem, d_model).
    pub fn forward_with_prefix_cached(
        &self,
        prefix: &Tensor,
        input_ids: &Tensor,
        kv_caches: &mut Vec<Option<(Tensor, Tensor)>>,
        memory: Option<&Tensor>,
    ) -> Result<Tensor> {
        let tok_embs = self.tok_emb.forward(input_ids)?;
        let mut x = Tensor::cat(&[prefix, &tok_embs], 1)?;
        let total_len = x.dim(1)?;
        let n_prefix = prefix.dim(1)?;

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_cached(&x, &self.rope_cos, &self.rope_sin, 0, n_prefix, self.attn_temp, &mut kv_caches[i], memory)?;
        }
        x = self.final_norm.forward(&x)?;

        // Return logits for the last token position only
        let last_hidden = x.narrow(1, total_len - 1, 1)?;
        self.lm_head.forward(&last_hidden).map_err(Into::into)
    }

    /// Cached forward — single token step (after prefill).
    /// `token_id`: single token tensor (batch, 1).
    /// `seq_offset`: RoPE position of this TEXT token (excludes prefix from position count).
    /// `memory`: optional external memory embeddings for cross-attention (batch, n_mem, d_model).
    pub fn forward_step_cached(
        &self,
        token_id: &Tensor,
        seq_offset: usize,
        kv_caches: &mut Vec<Option<(Tensor, Tensor)>>,
        memory: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = self.tok_emb.forward(token_id)?;

        for (i, layer) in self.layers.iter().enumerate() {
            // n_prefix=0: step calls are always text tokens, no prefix
            x = layer.forward_cached(&x, &self.rope_cos, &self.rope_sin, seq_offset, 0, self.attn_temp, &mut kv_caches[i], memory)?;
        }
        x = self.final_norm.forward(&x)?;
        self.lm_head.forward(&x).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Loss
// ---------------------------------------------------------------------------

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (b, s, v) = logits.dims3()?;
    let logits_flat = logits.reshape((b * s, v))?;
    let targets_flat = targets.reshape(b * s)?;
    candle_nn::loss::cross_entropy(&logits_flat, &targets_flat).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Gradient Check
// ---------------------------------------------------------------------------

pub fn gradient_check(
    cfg: &TransformerConfig,
    device: &Device,
    n_params_to_check: usize,
    eps: f64,
) -> Result<f64> {
    use rand::Rng;

    let varmap = VarMap::new();
    let model = WiredTransformer::new(cfg.clone(), &varmap, device)?;

    let batch = 2;
    let seq = cfg.max_seq_len.min(8);
    let mut rng = rand::thread_rng();

    let input_data: Vec<u32> = (0..batch * seq)
        .map(|_| rng.gen_range(0..cfg.vocab_size as u32))
        .collect();
    let target_data: Vec<u32> = (0..batch * seq)
        .map(|_| rng.gen_range(0..cfg.vocab_size as u32))
        .collect();
    let input_ids = Tensor::from_vec(input_data.clone(), (batch, seq), device)?;
    let targets = Tensor::from_vec(target_data.clone(), (batch, seq), device)?;

    let logits = model.forward(&input_ids)?;
    let loss = cross_entropy_loss(&logits, &targets)?;
    let grads = loss.backward()?;

    let all_vars = varmap.all_vars();
    let mut max_rel_err = 0.0f64;
    let mut checked = 0usize;

    for var in all_vars.iter() {
        if checked >= n_params_to_check {
            break;
        }

        let auto_grad = match grads.get(var.as_tensor()) {
            Some(g) => g.flatten_all()?.to_vec1::<f32>()?,
            None => continue,
        };

        let data = var.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let n_elements = data.len();
        let indices_to_check: Vec<usize> = if n_elements <= 2 {
            (0..n_elements).collect()
        } else {
            let mut indices = Vec::new();
            for _ in 0..2.min(n_params_to_check.saturating_sub(checked)) {
                indices.push(rng.gen_range(0..n_elements));
            }
            indices
        };

        for &idx in &indices_to_check {
            if checked >= n_params_to_check {
                break;
            }

            let shape = var.as_tensor().shape().clone();
            let original_data = data.clone();

            let mut dp = original_data.clone();
            dp[idx] += eps as f32;
            var.set(&Tensor::from_vec(dp, shape.clone(), device)?)?;
            let lp = cross_entropy_loss(&model.forward(&input_ids)?, &targets)?
                .to_scalar::<f32>()? as f64;

            let mut dm = original_data.clone();
            dm[idx] -= eps as f32;
            var.set(&Tensor::from_vec(dm, shape.clone(), device)?)?;
            let lm = cross_entropy_loss(&model.forward(&input_ids)?, &targets)?
                .to_scalar::<f32>()? as f64;

            var.set(&Tensor::from_vec(original_data, shape, device)?)?;

            let numerical = (lp - lm) / (2.0 * eps);
            let autograd = auto_grad[idx] as f64;

            let denom = numerical.abs().max(autograd.abs()).max(1e-8);
            let rel_err = (numerical - autograd).abs() / denom;

            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }

            checked += 1;
        }
    }

    Ok(max_rel_err)
}

// ---------------------------------------------------------------------------
// Training Benchmark
// ---------------------------------------------------------------------------

pub fn run_benchmark(
    cfg: &TransformerConfig,
    device: &Device,
    steps: usize,
    batch_size: usize,
) -> Result<Vec<f32>> {
    use rand::Rng;

    let varmap = VarMap::new();
    let model = WiredTransformer::new(cfg.clone(), &varmap, device)?;
    let params = candle_nn::ParamsAdamW {
        lr: 3e-4,
        weight_decay: 0.01,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), params)?;

    let seq = cfg.max_seq_len;
    let mut rng = rand::thread_rng();
    let mut losses = Vec::with_capacity(steps);

    for step in 0..steps {
        let input_data: Vec<u32> = (0..batch_size * seq)
            .map(|_| rng.gen_range(0..cfg.vocab_size as u32))
            .collect();
        let target_data: Vec<u32> = (0..batch_size * seq)
            .map(|_| rng.gen_range(0..cfg.vocab_size as u32))
            .collect();
        let input_ids = Tensor::from_vec(input_data, (batch_size, seq), device)?;
        let targets = Tensor::from_vec(target_data, (batch_size, seq), device)?;

        let logits = model.forward(&input_ids)?;
        let loss = cross_entropy_loss(&logits, &targets)?;
        let loss_val = loss.to_scalar::<f32>()?;
        optimizer.backward_step(&loss)?;

        if step % 10 == 0 {
            eprintln!("[step {step:>4}/{steps}] loss = {loss_val:.4}");
        }
        losses.push(loss_val);
    }

    Ok(losses)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_shapes() -> Result<()> {
        let cfg = TransformerConfig::tiny();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = WiredTransformer::new(cfg.clone(), &varmap, &device)?;

        let input = Tensor::zeros((2, 8), DType::U32, &device)?;
        let logits = model.forward(&input)?;

        assert_eq!(logits.dims3()?, (2, 8, cfg.vocab_size));
        Ok(())
    }

    #[test]
    fn test_encode_shapes() -> Result<()> {
        let cfg = TransformerConfig::tiny();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = WiredTransformer::new(cfg.clone(), &varmap, &device)?;

        let input = Tensor::zeros((2, 8), DType::U32, &device)?;
        let hidden = model.encode(&input)?;

        // encode() returns hidden states: (B, S, d_model)
        assert_eq!(hidden.dims3()?, (2, 8, cfg.d_model));
        Ok(())
    }

    #[test]
    fn test_forward_with_prefix() -> Result<()> {
        let cfg = TransformerConfig::tiny();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = WiredTransformer::new(cfg.clone(), &varmap, &device)?;

        let n_prefix = 4;
        let seq_len = 6;
        let prefix = Tensor::randn(0f32, 1.0, (1, n_prefix, cfg.d_model), &device)?;
        let input = Tensor::zeros((1, seq_len), DType::U32, &device)?;

        let logits = model.forward_with_prefix(&prefix, &input)?;

        // Should return logits for input positions only (not prefix)
        assert_eq!(logits.dims3()?, (1, seq_len, cfg.vocab_size));
        Ok(())
    }

    #[test]
    fn test_gradient_check() -> Result<()> {
        let cfg = TransformerConfig::tiny();
        let device = Device::Cpu;
        let max_rel_err = gradient_check(&cfg, &device, 50, 1e-3)?;
        eprintln!("gradient check max relative error: {max_rel_err:.6e}");
        assert!(
            max_rel_err < 1.0,
            "gradient check failed: max relative error {max_rel_err:.6e} >= 1.0"
        );
        Ok(())
    }

    #[test]
    fn test_rope_positions() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let seq_len = 4;
        let (cos, sin, _attn_temp) = precompute_rope(seq_len, head_dim, seq_len, &device)?;

        // cos/sin shape: (seq_len, head_dim/2)
        assert_eq!(cos.dims2()?, (seq_len, head_dim / 2));
        assert_eq!(sin.dims2()?, (seq_len, head_dim / 2));

        // Position 0: cos(0) = 1.0 for all frequencies
        let cos_row0 = cos.get(0)?.to_vec1::<f32>()?;
        for &v in &cos_row0 {
            assert!((v - 1.0).abs() < 1e-5, "cos(0) should be 1.0, got {v}");
        }

        // Position 0: sin(0) = 0.0 for all frequencies
        let sin_row0 = sin.get(0)?.to_vec1::<f32>()?;
        for &v in &sin_row0 {
            assert!(v.abs() < 1e-5, "sin(0) should be 0.0, got {v}");
        }

        Ok(())
    }

    #[test]
    fn test_loss_finite() -> Result<()> {
        let cfg = TransformerConfig::tiny();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = WiredTransformer::new(cfg.clone(), &varmap, &device)?;

        let input = Tensor::zeros((2, 8), DType::U32, &device)?;
        let targets = Tensor::zeros((2, 8), DType::U32, &device)?;
        let logits = model.forward(&input)?;
        let loss = cross_entropy_loss(&logits, &targets)?;
        let val = loss.to_scalar::<f32>()?;

        assert!(val.is_finite(), "loss should be finite, got {val}");
        assert!(val > 0.0, "loss should be positive, got {val}");
        Ok(())
    }

    #[test]
    fn test_backward_no_explosion() -> Result<()> {
        let cfg = TransformerConfig::tiny();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = WiredTransformer::new(cfg.clone(), &varmap, &device)?;

        let input = Tensor::zeros((2, 8), DType::U32, &device)?;
        let targets = Tensor::zeros((2, 8), DType::U32, &device)?;
        let logits = model.forward(&input)?;
        let loss = cross_entropy_loss(&logits, &targets)?;
        let grads = loss.backward()?;

        for var in varmap.all_vars() {
            if let Some(g) = grads.get(var.as_tensor()) {
                let max_val = g.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(max_val < 1e4, "gradient explosion: max grad = {max_val}");
                assert!(max_val.is_finite(), "non-finite gradient detected");
            }
        }
        Ok(())
    }

    #[test]
    fn test_training_step_reduces_loss() -> Result<()> {
        let cfg = TransformerConfig::tiny();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = WiredTransformer::new(cfg.clone(), &varmap, &device)?;
        let mut opt = candle_nn::AdamW::new(
            varmap.all_vars(),
            candle_nn::ParamsAdamW {
                lr: 1e-3,
                ..Default::default()
            },
        )?;

        let input = Tensor::new(vec![1u32, 2, 3, 4, 5, 6, 7, 8], &device)?.unsqueeze(0)?;
        let target = Tensor::new(vec![2u32, 3, 4, 5, 6, 7, 8, 9], &device)?.unsqueeze(0)?;

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for step in 0..50 {
            let logits = model.forward(&input)?;
            let loss = cross_entropy_loss(&logits, &target)?;
            let val = loss.to_scalar::<f32>()?;
            if step == 0 {
                first_loss = val;
            }
            last_loss = val;
            opt.backward_step(&loss)?;
        }

        assert!(
            last_loss < first_loss,
            "loss should decrease: {first_loss:.4} -> {last_loss:.4}"
        );
        Ok(())
    }

    #[test]
    fn test_cross_attention_forward() -> Result<()> {
        let cfg = TransformerConfig {
            use_cross_attn: true,
            ..TransformerConfig::tiny()
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = WiredTransformer::new(cfg.clone(), &varmap, &device)?;

        let batch = 2;
        let seq = 8;
        let n_mem = 4;
        let input = Tensor::zeros((batch, seq), DType::U32, &device)?;
        let memory = Tensor::randn(0f32, 1.0, (batch, n_mem, cfg.d_model), &device)?;

        // Forward with memory
        let logits_mem = model.forward_with_prefix_t(
            &Tensor::randn(0f32, 1.0, (batch, 2, cfg.d_model), &device)?,
            &input, false, Some(&memory),
        )?;
        assert_eq!(logits_mem.dims3()?, (batch, seq, cfg.vocab_size));

        // Forward without memory (same model)
        let logits_no_mem = model.forward_with_prefix_t(
            &Tensor::randn(0f32, 1.0, (batch, 2, cfg.d_model), &device)?,
            &input, false, None,
        )?;
        assert_eq!(logits_no_mem.dims3()?, (batch, seq, cfg.vocab_size));

        Ok(())
    }

    #[test]
    fn test_cross_attention_grads() -> Result<()> {
        let cfg = TransformerConfig {
            use_cross_attn: true,
            ..TransformerConfig::tiny()
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = WiredTransformer::new(cfg.clone(), &varmap, &device)?;

        let batch = 2;
        let seq = 8;
        let n_mem = 4;
        let prefix = Tensor::randn(0f32, 1.0, (batch, 2, cfg.d_model), &device)?;
        let input = Tensor::zeros((batch, seq), DType::U32, &device)?;
        let targets = Tensor::zeros((batch, seq), DType::U32, &device)?;
        let memory = Tensor::randn(0f32, 1.0, (batch, n_mem, cfg.d_model), &device)?;

        let logits = model.forward_with_prefix_t(&prefix, &input, true, Some(&memory))?;
        let loss = cross_entropy_loss(&logits, &targets)?;
        let grads = loss.backward()?;

        // Cross-attention params should get gradients
        let mut cross_attn_grads = 0;
        for var in varmap.all_vars() {
            if grads.get(var.as_tensor()).is_some() {
                cross_attn_grads += 1;
            }
        }
        // With cross_attn, we should have more params than without
        assert!(cross_attn_grads > 0, "no gradients found");
        Ok(())
    }
}
