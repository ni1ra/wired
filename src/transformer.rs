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
}

impl TransformerConfig {
    /// Phase 0-1 default: d=512, 4 layers, 8 heads.
    pub fn default() -> Self {
        Self {
            d_model: 512,
            n_layers: 4,
            n_heads: 8,
            d_ff: 2048,
            vocab_size: 373,
            max_seq_len: 256,
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
        }
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

fn precompute_rope(
    seq_len: usize,
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let theta: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / 10000f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta, device)?;
    let positions: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
    let positions = Tensor::new(positions, device)?;
    let freqs = positions.unsqueeze(1)?.matmul(&theta.unsqueeze(0)?)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Ok((cos, sin))
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

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
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

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn = (attn / scale)?;

        let mask = build_causal_mask(s, x.device())?;
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
// Transformer Block
// ---------------------------------------------------------------------------

struct TransformerBlock {
    attn_norm: GradRmsNorm,
    attn: Attention,
    mlp_norm: GradRmsNorm,
    mlp: Mlp,
}

impl TransformerBlock {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: GradRmsNorm::new(cfg.d_model, 1e-6, vb.pp("attn_norm"))?,
            attn: Attention::new(cfg, vb.pp("attn"))?,
            mlp_norm: GradRmsNorm::new(cfg.d_model, 1e-6, vb.pp("mlp_norm"))?,
            mlp: Mlp::new(cfg.d_model, cfg.d_ff, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.attn_norm.forward(x)?, cos, sin)?;
        let x = (x + h)?;
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
        let (rope_cos, rope_sin) = precompute_rope(cfg.max_seq_len, cfg.head_dim(), device)?;
        Ok(Self {
            config: cfg,
            tok_emb,
            layers,
            final_norm,
            lm_head,
            rope_cos,
            rope_sin,
        })
    }

    pub fn new(cfg: TransformerConfig, varmap: &VarMap, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        Self::from_vb(cfg, vb, device)
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut x = self.tok_emb.forward(input_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, &self.rope_cos, &self.rope_sin)?;
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
        let mut x = self.tok_emb.forward(input_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, &self.rope_cos, &self.rope_sin)?;
        }
        self.final_norm.forward(&x).map_err(Into::into)
    }

    /// Forward with continuous prefix embeddings prepended to the token sequence.
    /// Returns logits for input_ids positions only (prefix positions excluded).
    pub fn forward_with_prefix(&self, prefix: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let tok_embs = self.tok_emb.forward(input_ids)?;
        let mut x = Tensor::cat(&[prefix, &tok_embs], 1)?;

        for layer in &self.layers {
            x = layer.forward(&x, &self.rope_cos, &self.rope_sin)?;
        }
        x = self.final_norm.forward(&x)?;

        let n_prefix = prefix.dim(1)?;
        let seq_len = input_ids.dim(1)?;
        let token_hidden = x.narrow(1, n_prefix, seq_len)?;
        self.lm_head.forward(&token_hidden).map_err(Into::into)
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
            max_rel_err < 5e-1,
            "gradient check failed: max relative error {max_rel_err:.6e} >= 5e-1"
        );
        Ok(())
    }

    #[test]
    fn test_rope_positions() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 8;
        let seq_len = 4;
        let (cos, sin) = precompute_rope(seq_len, head_dim, &device)?;

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
}
