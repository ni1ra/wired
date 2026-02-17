// Unified brain: concept encoder + projector + decoder + policy heads + memory
// Merged from V4 brain_regions.rs (973 LOC) + brain_policy.rs (641 LOC) + talk.rs components -- T-005, T-007
//
// Architecture:
//   ConceptEncoder:  goal_bytes -> transformer -> concept_vec (bottleneck)
//   ConceptProjector: concept_vec -> N prefix embeddings for decoder
//   MemoryBank:      episodic store with cosine similarity retrieval
//   LanguageDecoder: [concept_prefix; memory_prefix; BOS; response] -> next byte
//   PolicyHeads:     byte_embed -> transformer(encode) -> global_avg_pool -> 5 heads
//
// "Language is just an abstraction of concepts" -- Lain
// "It must be based on memory too, not just concepts" -- Lain

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
#[cfg(test)]
use candle_core::Var;
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};
use crate::training::{CosineScheduler, EarlyStopping, EarlyStopAction, Trainer, TrainingConfig, weighted_cross_entropy};
use crate::transformer::{TransformerConfig, WiredTransformer};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Byte-level tokenizer for natural language (from talk.rs)
// ---------------------------------------------------------------------------

pub const TOK_PAD: u32 = 256;
pub const TOK_BOS: u32 = 257;
pub const TOK_EOS: u32 = 258;
pub const TALK_VOCAB_SIZE: usize = 259; // 256 bytes + PAD + BOS + EOS

pub struct TalkTokenizer;

impl TalkTokenizer {
    pub fn vocab_size(&self) -> usize { TALK_VOCAB_SIZE }

    pub fn encode(&self, s: &str) -> Vec<u32> {
        let mut ids = vec![TOK_BOS];
        ids.extend(s.bytes().map(|b| b as u32));
        ids.push(TOK_EOS);
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter()
            .filter(|&&id| id < 256)
            .map(|&id| id as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn pad_or_truncate(&self, ids: &[u32], len: usize) -> Vec<u32> {
        if ids.len() >= len {
            ids[..len].to_vec()
        } else {
            let mut out = vec![TOK_PAD; len];
            let offset = len - ids.len();
            out[offset..].copy_from_slice(ids);
            out
        }
    }
}

// ---------------------------------------------------------------------------
// JARVIS Dialogue Corpus (loaded from data/brain_corpus.json at compile time)
// ---------------------------------------------------------------------------

static CORPUS: OnceLock<Vec<(String, String)>> = OnceLock::new();

/// Load the merged JARVIS dialogue corpus. Parsed once from embedded JSON,
/// cached in a static OnceLock. Returns ~3,100 deduplicated (user, assistant) pairs
/// from V3 expanded corpus, V3 gold corpus, and original V5 hardcoded dialogues.
pub fn load_corpus() -> &'static [(String, String)] {
    CORPUS.get_or_init(|| {
        #[derive(serde::Deserialize)]
        struct Entry { user: String, assistant: String }
        let entries: Vec<Entry> = serde_json::from_str(
            include_str!("../data/brain_corpus.json")
        ).expect("brain_corpus.json parse failed");
        entries.into_iter().map(|e| (e.user, e.assistant)).collect()
    })
}

/// Backward-compat wrapper for callers that need `(&str, &str)` pairs.
pub fn corpus_as_str_pairs() -> Vec<(&'static str, &'static str)> {
    load_corpus().iter().map(|(u, a)| (u.as_str(), a.as_str())).collect()
}
/// Temperature sampling over logit distribution.
pub fn sample_with_temperature(logits: &[f32], temperature: f64) -> u32 {
    let temp = temperature as f32;
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = logits.iter()
        .map(|&l| ((l - max_logit) / temp).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    let r: f32 = rand::random::<f32>();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

// ---------------------------------------------------------------------------
// Brain Config (regions)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BrainConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_concept_tokens: usize,
    pub memory_k: usize,
    pub encoder_seq_len: usize,
    pub decoder_seq_len: usize,
    pub memory_capacity: usize,
    pub sft_steps: usize,
    pub sft_lr: f64,
    pub sft_batch_size: usize,
    pub temperature: f64,
    pub da_steps: usize,
    pub da_lr: f64,
    pub da_batch_size: usize,
    pub max_noise_rate: f64,
    pub early_stop_sft: f32,     // SFT early stop threshold (0.0 = disabled)
    pub early_stop_da: f32,      // DA early stop threshold
    pub early_stop_patience: usize,
}

impl BrainConfig {
    pub fn default_brain() -> Self {
        Self {
            d_model: 512,
            encoder_layers: 1,  // v12: shallow encoder to prevent depth-induced collapse (sim=0.96 at 4L init!)
            decoder_layers: 4,
            n_heads: 8,
            d_ff: 2048,
            n_concept_tokens: 16,
            memory_k: 8,
            encoder_seq_len: 128,
            decoder_seq_len: 256,
            memory_capacity: 1024,
            sft_steps: 50000,   // v16: 13x more data needs more steps
            sft_lr: 3e-4,
            sft_batch_size: 48,
            temperature: 0.8,
            da_steps: 8192,
            da_lr: 1e-4,
            da_batch_size: 16,
            max_noise_rate: 0.10,
            early_stop_sft: 0.0,  // v16: disabled — val loss + patience handles stopping
            early_stop_da: 5e-3,
            early_stop_patience: 10, // v16: more patience for smoother convergence signal
        }
    }

    /// Default config with DA disabled.
    /// Use when DA hyperparams are not yet tuned (DA at lr=1e-4 corrupts
    /// first-byte accuracy and breaks autoregressive coherence — v14 finding).
    pub fn default_no_da() -> Self {
        let mut cfg = Self::default_brain();
        cfg.da_steps = 0;
        cfg
    }

    pub fn test_brain() -> Self {
        Self {
            d_model: 64,
            encoder_layers: 1,
            decoder_layers: 2,
            n_heads: 4,
            d_ff: 128,
            n_concept_tokens: 2,
            memory_k: 2,
            encoder_seq_len: 64,
            decoder_seq_len: 128,
            memory_capacity: 32,
            sft_steps: 30,
            sft_lr: 1e-3,
            sft_batch_size: 16, // v16: minibatch for expanded corpus (was 0=full batch)
            temperature: 1.0,
            da_steps: 0,  // v16: DA disabled in test (known to produce NaN with expanded corpus)
            da_lr: 3e-4,
            da_batch_size: 4,
            max_noise_rate: 0.02,
            early_stop_sft: 0.0, // disabled for test
            early_stop_da: 0.0,
            early_stop_patience: 3,
        }
    }

    /// Phase 2 config: d=1024, 8+8 layers, ~200M params.
    /// Designed for RTX 5070 Ti (16GB VRAM).
    pub fn phase2() -> Self {
        Self {
            d_model: 1024,
            encoder_layers: 8,
            decoder_layers: 8,
            n_heads: 8,
            d_ff: 4096,
            n_concept_tokens: 16,
            memory_k: 8,
            encoder_seq_len: 128,
            decoder_seq_len: 256,
            memory_capacity: 1024,
            sft_steps: 50000,
            sft_lr: 3e-4,
            sft_batch_size: 8,
            temperature: 0.1,
            da_steps: 16384,
            da_lr: 1e-4,
            da_batch_size: 4,
            max_noise_rate: 0.10,
            early_stop_sft: 1e-4,
            early_stop_da: 5e-3,
            early_stop_patience: 5,
        }
    }

    fn total_decoder_seq(&self) -> usize {
        self.n_concept_tokens + self.memory_k + self.decoder_seq_len
    }
}

// ---------------------------------------------------------------------------
// Policy Config + Constants
// ---------------------------------------------------------------------------

pub const NUM_INTENTS: usize = 16;
pub const NUM_ACTIONS: usize = 16;
pub const PLAN_STEPS: usize = 6;
pub const NUM_PATTERNS: usize = 6;
pub const NUM_FILES: usize = 10;
pub const NUM_PICKS: usize = 129;
const BYTE_VOCAB: usize = 256;

pub const ACT_END: usize = 0;
pub const ACT_TALK: usize = 1;
pub const ACT_CARGO_TEST: usize = 2;
pub const ACT_DOCS_LINT: usize = 3;
pub const ACT_RG: usize = 4;
pub const ACT_PROVE_ALGEBRA: usize = 5;
pub const ACT_PATCH_DRY_RUN: usize = 6;
pub const ACT_WIRED_EVAL: usize = 7;
pub const ACT_WIRED_TRAIN_TEST: usize = 8;
pub const ACT_MEMORY_ADD: usize = 9;
pub const ACT_MEMORY_SEARCH: usize = 10;
pub const ACT_CARGO_CHECK: usize = 11;
pub const ACT_REPO_LIST: usize = 12;
pub const ACT_REPO_READ: usize = 13;
pub const ACT_FIX_TESTS: usize = 14;
pub const ACT_LEAN_SUITE: usize = 15;

const INTENT_RUN_TESTS: usize = 0;
const INTENT_HELLO: usize = 1;
const INTENT_REPO_SEARCH: usize = 2;
const INTENT_PATCH_DRY_RUN: usize = 3;
const INTENT_PROVE_ALGEBRA: usize = 4;
const INTENT_FIX_TESTS: usize = 5;
const INTENT_DOCS_LINT: usize = 6;
const INTENT_COMPOSITE: usize = 9;
const INTENT_MEMORY_ADD: usize = 10;
const INTENT_MEMORY_SEARCH: usize = 11;
const INTENT_CARGO_CHECK: usize = 12;
const INTENT_REPO_LIST: usize = 13;
const INTENT_REPO_READ: usize = 14;
const INTENT_LEAN_SUITE: usize = 15;

const LOSS_W_INTENT: f64 = 2.0;
const LOSS_W_ACT: f64 = 2.0;
const LOSS_W_PAT: f64 = 2.0;
const LOSS_W_FILE: f64 = 2.0;
const LOSS_W_PICK: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct PolicyConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub seq_len: usize,
    pub steps: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub full_curriculum: bool,
    pub early_stop_loss: f32,    // 0.0 = disabled
    pub early_stop_patience: usize,
}

impl PolicyConfig {
    pub fn test() -> Self {
        Self {
            d_model: 64,
            n_layers: 1,
            n_heads: 4,
            d_ff: 128,
            seq_len: 64,
            steps: 6144,
            lr: 2e-3,
            batch_size: 1,
            full_curriculum: false,
            early_stop_loss: 0.0,
            early_stop_patience: 3,
        }
    }

    pub fn full() -> Self {
        Self {
            d_model: 256,
            n_layers: 3,
            n_heads: 8,
            d_ff: 512,
            seq_len: 128,
            steps: 16384,
            lr: 7e-4,
            batch_size: 8,
            full_curriculum: true,
            early_stop_loss: 1e-3,
            early_stop_patience: 5,
        }
    }

    /// Phase 2 policy: scaled to match Phase 2 brain dimensions.
    pub fn phase2() -> Self {
        Self {
            d_model: 512,
            n_layers: 4,
            n_heads: 8,
            d_ff: 2048,
            seq_len: 128,
            steps: 32768,
            lr: 5e-4,
            batch_size: 8,
            full_curriculum: true,
            early_stop_loss: 1e-3,
            early_stop_patience: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Memory Bank -- Episodic Memory
// ---------------------------------------------------------------------------

struct MemoryEntry {
    concept_vec: Vec<f32>,
    _response: String,
}

pub struct MemoryBank {
    entries: Vec<MemoryEntry>,
    capacity: usize,
}

impl MemoryBank {
    fn new(capacity: usize) -> Self {
        Self { entries: Vec::new(), capacity }
    }

    fn store(&mut self, concept_vec: Vec<f32>, response: String) {
        if self.entries.len() >= self.capacity {
            self.entries.remove(0); // FIFO eviction
        }
        self.entries.push(MemoryEntry { concept_vec, _response: response });
    }

    fn retrieve_vecs(&self, query: &[f32], k: usize) -> Vec<&[f32]> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }

        let q_norm = l2_norm(query);
        if q_norm < 1e-8 { return vec![]; }

        let mut scored: Vec<(usize, f32)> = self.entries.iter().enumerate()
            .map(|(i, e)| {
                let e_norm = l2_norm(&e.concept_vec);
                let sim = if e_norm > 1e-8 {
                    dot(&e.concept_vec, query) / (q_norm * e_norm)
                } else {
                    0.0
                };
                (i, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.iter()
            .take(k)
            .map(|(i, _)| self.entries[*i].concept_vec.as_slice())
            .collect()
    }

    pub fn len(&self) -> usize { self.entries.len() }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

// ---------------------------------------------------------------------------
// Simple RNG (deterministic)
// ---------------------------------------------------------------------------

struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// Unified Brain (regions + policy)
// ---------------------------------------------------------------------------

pub struct Brain {
    // --- Regions ---
    concept_encoder: WiredTransformer,
    concept_projector: Linear,
    memory_projector: Linear,
    language_decoder: WiredTransformer,
    memory_bank: MemoryBank,
    pub config: BrainConfig,
    // --- Policy ---
    policy_backbone: WiredTransformer,
    head_intent: Linear,
    head_actions: Linear,
    head_patterns: Linear,
    head_files: Linear,
    head_picks: Linear,
}

pub struct PolicyOutput {
    pub intent_logits: Tensor,
    pub act_logits: Tensor,
    pub pat_logits: Tensor,
    pub file_logits: Tensor,
    pub pick_logits: Tensor,
}

impl Brain {
    pub fn new(
        config: BrainConfig,
        policy_cfg: &PolicyConfig,
        varmap: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);

        // --- Concept encoder ---
        let enc_cfg = TransformerConfig {
            d_model: config.d_model,
            n_layers: config.encoder_layers,
            n_heads: config.n_heads,
            d_ff: config.d_ff,
            vocab_size: TALK_VOCAB_SIZE,
            max_seq_len: config.encoder_seq_len,
        };
        let concept_encoder = WiredTransformer::from_vb(enc_cfg, vb.pp("enc"), device)?;

        // --- Concept projector ---
        let proj_out = config.n_concept_tokens * config.d_model;
        let concept_projector = linear(config.d_model, proj_out, vb.pp("concept_proj"))?;

        // --- Memory projector ---
        let memory_projector = linear(config.d_model, config.d_model, vb.pp("memory_proj"))?;

        // --- Language decoder ---
        let dec_cfg = TransformerConfig {
            d_model: config.d_model,
            n_layers: config.decoder_layers,
            n_heads: config.n_heads,
            d_ff: config.d_ff,
            vocab_size: TALK_VOCAB_SIZE,
            max_seq_len: config.total_decoder_seq(),
        };
        let language_decoder = WiredTransformer::from_vb(dec_cfg, vb.pp("dec"), device)?;

        let memory_bank = MemoryBank::new(config.memory_capacity);

        // --- Policy backbone + heads ---
        let pol_tcfg = TransformerConfig {
            d_model: policy_cfg.d_model,
            n_layers: policy_cfg.n_layers,
            n_heads: policy_cfg.n_heads,
            d_ff: policy_cfg.d_ff,
            vocab_size: BYTE_VOCAB,
            max_seq_len: policy_cfg.seq_len,
        };
        let policy_backbone = WiredTransformer::from_vb(pol_tcfg, vb.pp("pol_backbone"), device)?;

        let pd = policy_cfg.d_model;
        let head_intent = linear(pd, NUM_INTENTS, vb.pp("head_intent"))?;
        let head_actions = linear(pd, NUM_ACTIONS * PLAN_STEPS, vb.pp("head_actions"))?;
        let head_patterns = linear(pd, NUM_PATTERNS * PLAN_STEPS, vb.pp("head_patterns"))?;
        let head_files = linear(pd, NUM_FILES * PLAN_STEPS, vb.pp("head_files"))?;
        let head_picks = linear(pd, NUM_PICKS * PLAN_STEPS, vb.pp("head_picks"))?;

        Ok(Self {
            concept_encoder, concept_projector, memory_projector,
            language_decoder, memory_bank, config,
            policy_backbone, head_intent, head_actions,
            head_patterns, head_files, head_picks,
        })
    }

    // --- Regions methods ---

    /// Encode goal tokens into a concept vector via mean pooling over non-PAD positions.
    /// v14: replaces last-token pooling which produced identical embeddings because
    /// every input ended with EOS at the same position.
    pub fn encode_concept(&self, goal_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.concept_encoder.encode(goal_ids)?; // (batch, seq, d_model)
        // Build mask: 1.0 for non-PAD tokens, 0.0 for PAD (TOK_PAD = 256)
        let pad_val = Tensor::new(&[TOK_PAD], goal_ids.device())?;
        let is_pad = goal_ids.broadcast_eq(&pad_val)?; // (batch, seq) bool
        let mask = is_pad.to_dtype(DType::F32)?.neg()?.affine(1.0, 1.0)?; // 0->1, 1->0
        let mask_3d = mask.unsqueeze(2)?; // (batch, seq, 1)
        let masked_hidden = hidden.broadcast_mul(&mask_3d)?;
        let sum = masked_hidden.sum(1)?; // (batch, d_model)
        let count = mask.sum(1)?.unsqueeze(1)?.clamp(1.0, f64::MAX)?; // (batch, 1) min 1
        let mean = sum.broadcast_div(&count)?;
        Ok(mean.contiguous()?)
    }

    fn project_concepts(&self, concept_vec: &Tensor) -> Result<Tensor> {
        let batch = concept_vec.dim(0)?;
        let projected = self.concept_projector.forward(concept_vec)?;
        Ok(projected.reshape((batch, self.config.n_concept_tokens, self.config.d_model))?)
    }

    fn build_prefix(
        &self,
        concept_vec: &Tensor,
        memory_vecs: Option<&Tensor>,
    ) -> Result<Tensor> {
        let concept_prefix = self.project_concepts(concept_vec)?;

        match memory_vecs {
            Some(mem) => {
                let memory_prefix = self.memory_projector.forward(mem)?;
                Tensor::cat(&[&concept_prefix, &memory_prefix], 1)
                    .map_err(Into::into)
            }
            None => Ok(concept_prefix),
        }
    }

    pub fn forward(
        &self,
        goal_ids: &Tensor,
        response_ids: &Tensor,
    ) -> Result<Tensor> {
        let concept_vec = self.encode_concept(goal_ids)?;
        self.forward_from_concept(&concept_vec, response_ids)
    }

    /// Forward pass using pre-computed concept vectors (avoids double encoder call).
    pub fn forward_from_concept(
        &self,
        concept_vec: &Tensor,
        response_ids: &Tensor,
    ) -> Result<Tensor> {
        let prefix = self.build_prefix(concept_vec, None)?;
        self.language_decoder.forward_with_prefix(&prefix, response_ids)
    }

    pub fn forward_with_memory(
        &self,
        goal_ids: &Tensor,
        response_ids: &Tensor,
        device: &Device,
    ) -> Result<Tensor> {
        let concept_vec = self.encode_concept(goal_ids)?;

        let cv = concept_vec.squeeze(0)?.to_vec1::<f32>()?;
        let memories = self.memory_bank.retrieve_vecs(&cv, self.config.memory_k);

        let prefix = if memories.is_empty() {
            self.build_prefix(&concept_vec, None)?
        } else {
            let mem_data: Vec<f32> = memories.iter()
                .flat_map(|m| m.iter().copied())
                .collect();
            let n_mems = memories.len();
            let mem_tensor = Tensor::from_vec(
                mem_data, (1, n_mems, self.config.d_model), device,
            )?;
            self.build_prefix(&concept_vec, Some(&mem_tensor))?
        };

        self.language_decoder.forward_with_prefix(&prefix, response_ids)
    }

    pub fn store_memory(&mut self, concept_vec: Vec<f32>, response: String) {
        self.memory_bank.store(concept_vec, response);
    }

    pub fn memory_count(&self) -> usize { self.memory_bank.len() }

    // --- Policy methods ---

    pub fn classify(&self, input_ids: &Tensor) -> Result<PolicyOutput> {
        let hidden = self.policy_backbone.encode(input_ids)?;
        let pooled = hidden.mean(1)?;
        let b = pooled.dim(0)?;

        let intent_logits = self.head_intent.forward(&pooled)?;

        let act_flat = self.head_actions.forward(&pooled)?;
        let act_logits = act_flat.reshape((b, PLAN_STEPS, NUM_ACTIONS))?;

        let pat_flat = self.head_patterns.forward(&pooled)?;
        let pat_logits = pat_flat.reshape((b, PLAN_STEPS, NUM_PATTERNS))?;

        let file_flat = self.head_files.forward(&pooled)?;
        let file_logits = file_flat.reshape((b, PLAN_STEPS, NUM_FILES))?;

        let pick_flat = self.head_picks.forward(&pooled)?;
        let pick_logits = pick_flat.reshape((b, PLAN_STEPS, NUM_PICKS))?;

        Ok(PolicyOutput { intent_logits, act_logits, pat_logits, file_logits, pick_logits })
    }
}

// ---------------------------------------------------------------------------
// Right-padding utility (CRITICAL for RoPE position consistency)
// ---------------------------------------------------------------------------

fn right_pad(ids: &[u32], len: usize) -> Vec<u32> {
    if ids.len() >= len {
        ids[..len].to_vec()
    } else {
        let mut out = ids.to_vec();
        out.resize(len, TOK_PAD);
        out
    }
}

// ---------------------------------------------------------------------------
// Brain Training Data Preparation
// ---------------------------------------------------------------------------

fn prepare_brain_data(
    tok: &TalkTokenizer,
    enc_seq: usize,
    dec_seq: usize,
) -> Vec<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>)> {
    let corpus = load_corpus();
    let mut data = Vec::new();

    for (prompt, response) in corpus.iter() {
        let goal_ids = tok.encode(prompt);
        let goal_padded = tok.pad_or_truncate(&goal_ids, enc_seq);

        let resp_ids = tok.encode(response);
        if resp_ids.len() < 3 { continue; }

        let resp_input: Vec<u32> = resp_ids[..resp_ids.len() - 1].to_vec();
        let resp_target: Vec<u32> = resp_ids[1..].to_vec();

        let input_padded = right_pad(&resp_input, dec_seq);
        let target_padded = right_pad(&resp_target, dec_seq);

        let real_len = resp_input.len().min(dec_seq);
        let mut weights = vec![0.0f32; dec_seq];
        for w in weights.iter_mut().take(real_len) {
            *w = 1.0;
        }

        data.push((goal_padded, input_padded, target_padded, weights));
    }

    data
}

// ---------------------------------------------------------------------------
// Train/Val Split for Generalization
// ---------------------------------------------------------------------------

type BrainDatum = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>);

/// Split prepared data into train and validation sets using Fisher-Yates shuffle.
/// Returns (train_data, val_data). val_fraction=0.10 means 10% held out for validation.
fn train_val_split(
    data: Vec<BrainDatum>,
    val_fraction: f64,
    seed: u64,
) -> (Vec<BrainDatum>, Vec<BrainDatum>) {
    let mut rng = SimpleRng::new(seed);
    let mut indices: Vec<usize> = (0..data.len()).collect();

    // Fisher-Yates shuffle
    for i in (1..indices.len()).rev() {
        let j = rng.next_usize(i + 1);
        indices.swap(i, j);
    }

    let val_count = (data.len() as f64 * val_fraction).round() as usize;
    let val_count = val_count.max(1).min(data.len() - 1); // at least 1 val, at least 1 train

    // Move data out by index (avoid cloning)
    let mut data_vec: Vec<Option<BrainDatum>> = data.into_iter().map(Some).collect();
    let mut val_data = Vec::with_capacity(val_count);
    let mut train_data = Vec::with_capacity(indices.len() - val_count);

    for (rank, &idx) in indices.iter().enumerate() {
        let datum = data_vec[idx].take().unwrap();
        if rank < val_count {
            val_data.push(datum);
        } else {
            train_data.push(datum);
        }
    }

    (train_data, val_data)
}

/// Compute validation loss on held-out data (no gradients, no noise).
/// Samples up to `max_examples` from val_data to limit memory/compute.
fn compute_val_loss(
    brain: &Brain,
    val_data: &[BrainDatum],
    enc_seq: usize,
    dec_seq: usize,
    max_examples: usize,
    device: &Device,
) -> Result<f32> {
    let n = val_data.len().min(max_examples);
    if n == 0 { return Ok(0.0); }

    let mut total_loss = 0.0f64;
    let mut count = 0usize;

    // Process in small batches to avoid OOM
    let batch_size = 32usize.min(n);
    let mut offset = 0;

    while offset < n {
        let end = (offset + batch_size).min(n);
        let bs = end - offset;

        let mut bg = Vec::with_capacity(bs * enc_seq);
        let mut bi = Vec::with_capacity(bs * dec_seq);
        let mut bt = Vec::with_capacity(bs * dec_seq);
        let mut bw = Vec::with_capacity(bs * dec_seq);

        for datum in &val_data[offset..end] {
            bg.extend_from_slice(&datum.0);
            bi.extend_from_slice(&datum.1); // no noise for validation
            bt.extend_from_slice(&datum.2);
            bw.extend_from_slice(&datum.3);
        }

        let goal_t = Tensor::from_vec(bg, (bs, enc_seq), device)?;
        let input_t = Tensor::from_vec(bi, (bs, dec_seq), device)?;
        let target_t = Tensor::from_vec(bt, (bs, dec_seq), device)?;
        let weight_t = Tensor::from_vec(bw, (bs, dec_seq), device)?;

        let logits = brain.forward(&goal_t, &input_t)?;
        let loss = weighted_cross_entropy(&logits, &target_t, 0.0, Some(&weight_t))?;
        total_loss += loss.to_scalar::<f32>()? as f64;
        count += 1;

        offset = end;
    }

    Ok((total_loss / count as f64) as f32)
}

// ---------------------------------------------------------------------------
// Phase 1: Brain SFT Training with Token Noise Injection
// ---------------------------------------------------------------------------

pub fn train_brain_talk(
    config: &BrainConfig,
    policy_cfg: &PolicyConfig,
    device: &Device,
) -> Result<(Brain, VarMap, Vec<f32>)> {
    let tok = TalkTokenizer;
    let varmap = VarMap::new();
    let mut brain = Brain::new(config.clone(), policy_cfg, &varmap, device)?;

    let all_data = prepare_brain_data(&tok, config.encoder_seq_len, config.decoder_seq_len);
    let n_total = all_data.len();
    let enc_seq = config.encoder_seq_len;
    let dec_seq = config.decoder_seq_len;

    // Train/val split: 10% held out for validation-based early stopping
    // For test configs (tiny data), skip split to avoid breaking existing tests
    let use_val = n_total >= 20;
    let (data, val_data) = if use_val {
        let (train, val) = train_val_split(all_data, 0.10, 42);
        eprintln!("[brain] Loaded {} dialogues: {} train, {} val ({:.1}% val)",
            n_total, train.len(), val.len(), val.len() as f64 / n_total as f64 * 100.0);
        (train, val)
    } else {
        eprintln!("[brain] {} dialogues (no val split for small datasets)", n_total);
        (all_data, Vec::new())
    };

    let n_dialogues = data.len();

    // Determine batch size: 0 means full dataset (backward compat for test configs)
    let batch_size = if config.sft_batch_size == 0 || config.sft_batch_size >= n_dialogues {
        n_dialogues
    } else {
        config.sft_batch_size
    };
    let use_minibatch = batch_size < n_dialogues;

    let mut rng = SimpleRng::new(42 ^ 0xB2A1_CAFE);
    let mut losses = Vec::new();

    // Early stopping on validation loss (patience-based, no fixed threshold)
    // When val data exists: threshold=0.0 disables threshold, patience controls stopping
    // When no val data: fall back to training loss with configured threshold
    // Skip checkpoint saving for tiny runs (tests) to avoid clobbering production checkpoints
    let save_ckpt = config.sft_steps >= 100;
    let mut early_stop = if use_val {
        EarlyStopping::new(
            0.0, // no fixed threshold — purely patience-based on val loss
            config.early_stop_patience,
            if save_ckpt { Some("brain_best_sft.safetensors".to_string()) } else { None },
        ).with_stale_stop()
    } else {
        EarlyStopping::new(
            config.early_stop_sft,
            config.early_stop_patience,
            if save_ckpt && config.early_stop_sft > 0.0 { Some("brain_best_sft.safetensors".to_string()) } else { None },
        )
    };

    // =====================================================================
    // v14: Direct encoder training with gradient fixes
    // =====================================================================
    // v2-v12 failed because candle_nn::RmsNorm and softmax_last_dim had broken
    // backward passes (M-032). v13 bypassed the encoder with a learnable codebook.
    // v14 trains the encoder directly — GradRmsNorm + grad_softmax_last_dim fix
    // the gradient chain, and mean pooling in encode_concept fixes the EOS
    // dominance bug (all inputs ended with same token at same position).
    let sft_config = TrainingConfig {
        lr: config.sft_lr,
        min_lr: config.sft_lr * 0.01,
        weight_decay: 0.01,
        warmup_fraction: 0.1,
        total_steps: config.sft_steps,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        label_smoothing: 0.0,
    };

    let mut trainer = Trainer::new(varmap.clone(), sft_config)?;

    eprintln!("[brain] v16 SFT: {} train dialogues, batch={}, {} steps, lr={:.2e}, noise={:.1}%, encoder=DIRECT",
        n_dialogues, batch_size, config.sft_steps, config.sft_lr,
        config.max_noise_rate * 100.0);

    // Pre-flatten data for full-batch mode (test configs)
    let (full_goal_batch, full_raw_input, full_target_batch, full_weight_batch) = if !use_minibatch {
        let mut gd = Vec::with_capacity(n_dialogues * enc_seq);
        let mut id = Vec::with_capacity(n_dialogues * dec_seq);
        let mut td = Vec::with_capacity(n_dialogues * dec_seq);
        let mut wd = Vec::with_capacity(n_dialogues * dec_seq);
        for (g, i, t, w) in &data {
            gd.extend_from_slice(g);
            id.extend_from_slice(i);
            td.extend_from_slice(t);
            wd.extend_from_slice(w);
        }
        let gb = Tensor::from_vec(gd, (n_dialogues, enc_seq), device)?;
        let tb = Tensor::from_vec(td, (n_dialogues, dec_seq), device)?;
        let wb = Tensor::from_vec(wd, (n_dialogues, dec_seq), device)?;
        (Some(gb), Some(id), Some(tb), Some(wb))
    } else {
        (None, None, None, None)
    };

    let total_steps = config.sft_steps;

    for step in 0..total_steps {
        // Constant denoising with brief warmup
        let warmup = 1000usize.min(total_steps / 10);
        let noise_rate = if warmup > 0 && step < warmup {
            config.max_noise_rate * (step as f64 / warmup as f64)
        } else {
            config.max_noise_rate
        };

        let (goal_t, input_t, target_t, weight_t) = if use_minibatch {
            let mut bg = Vec::with_capacity(batch_size * enc_seq);
            let mut bi = Vec::with_capacity(batch_size * dec_seq);
            let mut bt = Vec::with_capacity(batch_size * dec_seq);
            let mut bw = Vec::with_capacity(batch_size * dec_seq);

            for _ in 0..batch_size {
                let idx = rng.next_usize(n_dialogues);
                let (ref goal, ref inp, ref tgt, ref wt) = data[idx];
                bg.extend_from_slice(goal);
                for &tok_id in inp.iter() {
                    if tok_id != TOK_PAD && tok_id != TOK_BOS && rng.next_f64() < noise_rate {
                        bi.push(TOK_PAD);
                    } else {
                        bi.push(tok_id);
                    }
                }
                bt.extend_from_slice(tgt);
                bw.extend_from_slice(wt);
            }

            (
                Tensor::from_vec(bg, (batch_size, enc_seq), device)?,
                Tensor::from_vec(bi, (batch_size, dec_seq), device)?,
                Tensor::from_vec(bt, (batch_size, dec_seq), device)?,
                Tensor::from_vec(bw, (batch_size, dec_seq), device)?,
            )
        } else {
            let raw = full_raw_input.as_ref().unwrap();
            let noisy: Vec<u32> = raw.iter().map(|&tok_id| {
                if tok_id != TOK_PAD && tok_id != TOK_BOS && rng.next_f64() < noise_rate {
                    TOK_PAD
                } else {
                    tok_id
                }
            }).collect();
            (
                full_goal_batch.as_ref().unwrap().clone(),
                Tensor::from_vec(noisy, (n_dialogues, dec_seq), device)?,
                full_target_batch.as_ref().unwrap().clone(),
                full_weight_batch.as_ref().unwrap().clone(),
            )
        };

        // v14: Direct encoder forward — gradients flow through encoder via
        // GradRmsNorm + grad_softmax_last_dim (fixed candle-nn backward bugs).
        // encode_concept now uses mean pooling over non-PAD tokens.
        let logits = brain.forward(&goal_t, &input_t)?;
        let loss = weighted_cross_entropy(&logits, &target_t, 0.0, Some(&weight_t))?;

        let loss_val = loss.to_scalar::<f32>()?;
        losses.push(loss_val);

        trainer.backward_step(&loss)?;

        let log_interval = if total_steps > 5000 { 1000 } else { 500 };
        if step % log_interval == 0 || step == total_steps - 1 {
            let window = losses.len().min(10);
            let avg: f32 = losses.iter().rev().take(window).sum::<f32>() / window as f32;
            let rss_kb = std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|s| s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok()))
                .unwrap_or(0);

            // Compute validation loss if val data available
            let (stop_loss, val_str) = if use_val && !val_data.is_empty() {
                let vl = compute_val_loss(&brain, &val_data, enc_seq, dec_seq, 100, device)?;
                (vl, format!(" val_loss={vl:.4}"))
            } else {
                (avg, String::new())
            };

            eprintln!("[brain] step {step}/{total_steps} train_loss={avg:.4}{val_str} lr={:.2e} noise={:.3} rss={:.0}MB",
                trainer.current_lr(), noise_rate, rss_kb as f64 / 1024.0);

            // Early stopping on validation loss (or training loss if no val data)
            if step > 0 {
                if let EarlyStopAction::Stop = early_stop.check(stop_loss, step, &varmap) {
                    eprintln!("[brain] SFT early stopped at step {step}/{total_steps} (best val_loss={:.4})",
                        early_stop.best_loss());
                    break;
                }
            }
        }

        // Mid-training encoder diversity diagnostic
        if (step == total_steps / 4 || step == total_steps / 2) && total_steps >= 4000 {
            eprintln!("[brain] === Mid-SFT diagnostic (step {step}) ===");
            diagnose_encoder(&brain, config, device)?;
        }
    }

    trainer.print_timer("brain-sft-v14");

    // Post-SFT encoder diagnostics
    if total_steps >= 1000 {
        diagnose_encoder(&brain, config, device)?;
    }

    // Save SFT-only checkpoint before DA (DA can corrupt generation quality)
    if config.da_steps > 0 {
        crate::training::save_checkpoint(&varmap, "brain_sft_only.safetensors")?;
        eprintln!("[CHECKPOINT] Saved SFT-only checkpoint (pre-DA fallback)");
    }

    // Phase 2: Dialogue-aligned finetuning
    if config.da_steps > 0 {
        let da_losses = brain_finetune_dialogue_aligned(&brain, &varmap, config, device)?;
        losses.extend(da_losses);
    }

    // Populate memory bank (cap to memory_capacity to avoid wasted compute)
    let corpus = load_corpus();
    let mem_sample = corpus.len().min(config.memory_capacity);
    for (prompt, response) in &corpus[..mem_sample] {
        let goal_ids = tok.encode(prompt);
        let goal_padded = tok.pad_or_truncate(&goal_ids, config.encoder_seq_len);
        let goal_tensor = Tensor::from_vec(
            goal_padded, (1, config.encoder_seq_len), device,
        )?;
        let cv = brain.encode_concept(&goal_tensor)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        brain.store_memory(cv, response.to_string());
    }

    eprintln!("[brain] Memory bank populated: {} entries", brain.memory_count());

    Ok((brain, varmap, losses))
}

// ---------------------------------------------------------------------------
// T-014: Concept Tokenizer Bootstrap (uses trained encoder)
// ---------------------------------------------------------------------------

/// Bootstrap the ConceptTokenizer from a trained brain's encoder.
/// Discovers merge rules ranked by semantic consistency in concept space.
pub fn bootstrap_concept_tokenizer(
    brain: &Brain,
    max_merges: usize,
    min_frequency: usize,
    device: &Device,
) -> Result<crate::tokenizer::ConceptTokenizer> {
    let tok = TalkTokenizer;
    let enc_seq = brain.config.encoder_seq_len;
    let corpus = corpus_as_str_pairs();

    eprintln!("[tokenizer] Bootstrapping concept tokenizer from trained encoder...");
    eprintln!("[tokenizer] Corpus: {} pairs, max_merges={}, min_freq={}",
        corpus.len(), max_merges, min_frequency);

    // Build concept_fn closure: byte slice -> concept vector via trained encoder
    let concept_fn = |bytes: &[u8]| -> Vec<f32> {
        let text = String::from_utf8_lossy(bytes);
        let ids = tok.encode(&text);
        let padded = tok.pad_or_truncate(&ids, enc_seq);
        let goal_tensor = Tensor::from_vec(padded, (1, enc_seq), device)
            .expect("tensor creation failed");
        let cv = brain.encode_concept(&goal_tensor)
            .expect("encode_concept failed");
        cv.squeeze(0)
            .expect("squeeze failed")
            .to_vec1::<f32>()
            .expect("to_vec1 failed")
    };

    let merges = crate::tokenizer::ConceptTokenizer::discover_merges(
        &corpus.iter().map(|(p, r)| (*p, *r)).collect::<Vec<_>>(),
        concept_fn,
        max_merges,
        min_frequency,
    );

    eprintln!("[tokenizer] Discovered {} merge rules", merges.len());
    if !merges.is_empty() {
        eprintln!("[tokenizer] Top 5 merges:");
        for (i, rule) in merges.iter().take(5).enumerate() {
            let pattern_str = String::from_utf8_lossy(&rule.pattern);
            eprintln!("  {}: \"{}\" (id={}, score={:.2})",
                i + 1, pattern_str, rule.token_id, rule.score);
        }
    }

    let concept_tok = crate::tokenizer::ConceptTokenizer::from_merges(merges);
    eprintln!("[tokenizer] ConceptTokenizer vocab: {} tokens (259 base + {} merges)",
        concept_tok.vocab_size(), concept_tok.num_merges());

    Ok(concept_tok)
}

// ---------------------------------------------------------------------------
// Post-SFT Diagnostics: concept diversity + greedy generation
// ---------------------------------------------------------------------------

#[allow(dead_code)] // Kept for future encoder distillation diagnostics
fn diagnose_after_sft(
    brain: &Brain,
    config: &BrainConfig,
    device: &Device,
) -> Result<()> {
    let tok = TalkTokenizer;
    let corpus = load_corpus();
    let enc_seq = config.encoder_seq_len;

    eprintln!("\n[DIAG] === Post-SFT Diagnostics ===");

    // 1. Concept vector diversity: compute pairwise cosine similarities
    let sample_size = corpus.len().min(50);
    let mut concept_vecs: Vec<Vec<f32>> = Vec::with_capacity(sample_size);
    for (prompt, _) in corpus.iter().take(sample_size) {
        let goal_ids = tok.encode(prompt);
        let goal_padded = tok.pad_or_truncate(&goal_ids, enc_seq);
        let goal_tensor = Tensor::from_vec(goal_padded, (1, enc_seq), device)?;
        let cv = brain.encode_concept(&goal_tensor)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        concept_vecs.push(cv);
    }

    // Compute mean pairwise cosine similarity
    let mut total_sim = 0.0f64;
    let mut count = 0u64;
    for i in 0..concept_vecs.len() {
        for j in (i+1)..concept_vecs.len() {
            let dot: f64 = concept_vecs[i].iter().zip(concept_vecs[j].iter())
                .map(|(&a, &b)| a as f64 * b as f64).sum();
            let norm_i: f64 = concept_vecs[i].iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            let norm_j: f64 = concept_vecs[j].iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            let sim = if norm_i > 1e-8 && norm_j > 1e-8 { dot / (norm_i * norm_j) } else { 0.0 };
            total_sim += sim;
            count += 1;
        }
    }
    let avg_sim = if count > 0 { total_sim / count as f64 } else { 0.0 };
    let avg_norm: f64 = concept_vecs.iter()
        .map(|v| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt())
        .sum::<f64>() / concept_vecs.len() as f64;

    eprintln!("[DIAG] Concept vectors ({} prompts): avg_cosine_sim={:.4}, avg_norm={:.2}",
        sample_size, avg_sim, avg_norm);
    eprintln!("[DIAG] (ideal: avg_cosine_sim < 0.5, meaning vectors are discriminative)");

    // 2. Greedy generation test (temp=0) for a few prompts
    let test_prompts = ["hello", "what can you do", "search for bugs", "write a test", "good morning"];
    eprintln!("[DIAG] Greedy generation (temp=0, no rep penalty):");
    for prompt in &test_prompts {
        let goal_ids = tok.encode(prompt);
        let goal_padded = tok.pad_or_truncate(&goal_ids, enc_seq);
        let goal_tensor = Tensor::from_vec(goal_padded.clone(), (1, enc_seq), device)?;
        let concept_vec = brain.encode_concept(&goal_tensor)?;
        let prefix = brain.build_prefix(&concept_vec, None)?;

        let max_gen = 80;
        let mut generated: Vec<u32> = vec![TOK_BOS];
        let dec_seq = config.decoder_seq_len;

        for _ in 0..max_gen {
            let padded = right_pad(&generated, dec_seq);
            let input = Tensor::from_vec(padded, (1, dec_seq), device)?;
            let logits = brain.language_decoder.forward_with_prefix(&prefix, &input)?;
            let read_pos = generated.len() - 1;
            let next_logits = logits.i((0, read_pos))?.to_vec1::<f32>()?;

            // Greedy: pick highest logit, but mask PAD
            let mut best_id = TOK_EOS;
            let mut best_val = f32::NEG_INFINITY;
            for (id, &val) in next_logits.iter().enumerate() {
                if id == TOK_PAD as usize { continue; }
                if val > best_val { best_val = val; best_id = id as u32; }
            }

            if best_id == TOK_EOS { break; }
            generated.push(best_id);
        }

        let response = tok.decode(&generated[1..]);
        // Truncate at a char boundary to avoid panicking on multi-byte UTF-8
        let truncated: &str = if response.len() > 60 {
            match response.char_indices().nth(60) {
                Some((idx, _)) => &response[..idx],
                None => &response,
            }
        } else {
            &response
        };
        eprintln!("[DIAG]   \"{}\" -> \"{}\"", prompt, truncated);
    }

    eprintln!("[DIAG] === End Post-SFT Diagnostics ===\n");
    Ok(())
}

// v13: Codebook-specific diagnostic — measures codebook vector diversity and
// greedy generation quality using codebook vectors (NOT encoder).
/// v14: Diagnose encoder diversity by encoding corpus prompts and measuring
/// pairwise cosine similarity of the resulting concept vectors.
/// Also runs greedy generation for sample prompts.
fn diagnose_encoder(
    brain: &Brain,
    config: &BrainConfig,
    device: &Device,
) -> Result<()> {
    let tok = TalkTokenizer;
    let corpus = load_corpus();
    let enc_seq = config.encoder_seq_len;

    eprintln!("\n[DIAG] === Encoder Diagnostics (v14) ===");

    // 1. Encode a sample of prompts and measure concept_vec diversity
    let sample_size = corpus.len().min(50);
    let mut concept_vecs: Vec<Vec<f32>> = Vec::with_capacity(sample_size);
    for i in 0..sample_size {
        let (prompt, _) = &corpus[i];
        let ids = tok.encode(prompt);
        let padded = tok.pad_or_truncate(&ids, enc_seq);
        let goal_t = Tensor::from_vec(padded, (1, enc_seq), device)?;
        let cv = brain.encode_concept(&goal_t)?.squeeze(0)?.to_vec1::<f32>()?;
        concept_vecs.push(cv);
    }

    let mut total_sim = 0.0f64;
    let mut count = 0u64;
    for i in 0..concept_vecs.len() {
        for j in (i+1)..concept_vecs.len() {
            let dot: f64 = concept_vecs[i].iter().zip(concept_vecs[j].iter())
                .map(|(&a, &b)| a as f64 * b as f64).sum();
            let norm_i: f64 = concept_vecs[i].iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            let norm_j: f64 = concept_vecs[j].iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            let sim = if norm_i > 1e-8 && norm_j > 1e-8 { dot / (norm_i * norm_j) } else { 0.0 };
            total_sim += sim;
            count += 1;
        }
    }
    let avg_sim = if count > 0 { total_sim / count as f64 } else { 0.0 };
    let avg_norm: f64 = concept_vecs.iter()
        .map(|v| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt())
        .sum::<f64>() / concept_vecs.len() as f64;

    eprintln!("[DIAG] Encoder concept_vecs ({} prompts): avg_cosine_sim={:.4}, avg_norm={:.4}",
        sample_size, avg_sim, avg_norm);
    eprintln!("[DIAG] (want low sim = discriminative; high sim = collapse)");

    // 2. Greedy generation using encoder-produced concept vectors
    let test_indices = [0usize, 1, 10, 50, 100];
    eprintln!("[DIAG] Greedy generation (temp=0, encoder concept_vecs):");
    for &idx in &test_indices {
        if idx >= corpus.len() { continue; }
        let (prompt, _) = &corpus[idx];

        let ids = tok.encode(prompt);
        let padded = tok.pad_or_truncate(&ids, enc_seq);
        let goal_t = Tensor::from_vec(padded, (1, enc_seq), device)?;
        let concept_vec = brain.encode_concept(&goal_t)?;
        let prefix = brain.build_prefix(&concept_vec, None)?;

        let max_gen = 80;
        let mut generated: Vec<u32> = vec![TOK_BOS];
        let dec_seq = config.decoder_seq_len;

        for _ in 0..max_gen {
            let padded = right_pad(&generated, dec_seq);
            let input = Tensor::from_vec(padded, (1, dec_seq), device)?;
            let logits = brain.language_decoder.forward_with_prefix(&prefix, &input)?;
            let read_pos = generated.len() - 1;
            let next_logits = logits.i((0, read_pos))?.to_vec1::<f32>()?;

            let mut best_id = TOK_EOS;
            let mut best_val = f32::NEG_INFINITY;
            for (id, &val) in next_logits.iter().enumerate() {
                if id == TOK_PAD as usize { continue; }
                if val > best_val { best_val = val; best_id = id as u32; }
            }

            if best_id == TOK_EOS { break; }
            generated.push(best_id);
        }

        let response = tok.decode(&generated[1..]);
        let truncated: &str = if response.len() > 60 {
            match response.char_indices().nth(60) {
                Some((idx_c, _)) => &response[..idx_c],
                None => &response,
            }
        } else {
            &response
        };
        let prompt_short: &str = if prompt.len() > 30 {
            match prompt.char_indices().nth(30) {
                Some((idx_c, _)) => &prompt[..idx_c],
                None => prompt,
            }
        } else {
            prompt
        };
        eprintln!("[DIAG]   [{}] \"{}\" -> \"{}\"", idx, prompt_short, truncated);
    }

    eprintln!("[DIAG] === End Encoder Diagnostics ===\n");
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2: Brain Dialogue-Aligned Finetuning
// ---------------------------------------------------------------------------

fn brain_finetune_dialogue_aligned(
    brain: &Brain,
    varmap: &VarMap,
    config: &BrainConfig,
    device: &Device,
) -> Result<Vec<f32>> {
    let tok = TalkTokenizer;
    let corpus = load_corpus();
    let enc_seq = config.encoder_seq_len;
    let dec_seq = config.decoder_seq_len;
    let batch_size = config.da_batch_size;

    let dialogues: Vec<(Vec<u32>, Vec<u32>)> = corpus.iter().map(|(prompt, response)| {
        let goal_ids = tok.encode(prompt);
        let goal_padded = tok.pad_or_truncate(&goal_ids, enc_seq);
        let resp_bytes: Vec<u32> = response.bytes().map(|b| b as u32).collect();
        (goal_padded, resp_bytes)
    }).collect();

    let total_resp_tokens: usize = dialogues.iter().map(|(_, r)| r.len().max(1)).sum();

    let da_config = TrainingConfig {
        lr: config.da_lr,
        min_lr: config.da_lr * 0.01,
        weight_decay: 0.01,
        warmup_fraction: 0.1,
        total_steps: config.da_steps,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        label_smoothing: 0.0,
    };

    let mut trainer = Trainer::new(varmap.clone(), da_config)?;
    let mut rng = SimpleRng::new(0xDA10_B2A1 ^ 42);

    let mut early_stop = EarlyStopping::new(
        config.early_stop_da,
        config.early_stop_patience,
        if config.early_stop_da > 0.0 { Some("brain_best_da.safetensors".to_string()) } else { None },
    );

    eprintln!("[brain] Phase 2 DA: {} dialogues, batch={}, {} steps, lr={:.2e}",
        dialogues.len(), batch_size, config.da_steps, config.da_lr);

    let mut losses = Vec::new();

    for step in 0..config.da_steps {
        let mut batch_goals: Vec<u32> = Vec::with_capacity(batch_size * enc_seq);
        let mut batch_input: Vec<u32> = Vec::with_capacity(batch_size * dec_seq);
        let mut batch_target: Vec<u32> = Vec::with_capacity(batch_size * dec_seq);
        let mut batch_weight: Vec<f32> = Vec::with_capacity(batch_size * dec_seq);

        for _ in 0..batch_size {
            let r = rng.next_usize(total_resp_tokens);
            let mut cum = 0usize;
            let mut di = 0usize;
            for (i, (_, resp)) in dialogues.iter().enumerate() {
                cum += resp.len().max(1);
                if r < cum { di = i; break; }
            }

            let (ref goal_padded, ref resp_bytes) = dialogues[di];
            let resp_len = resp_bytes.len().max(1);

            let ri = rng.next_usize(resp_len);
            let target_byte = resp_bytes[ri.min(resp_bytes.len() - 1)];

            let mut dec_input = vec![TOK_PAD; dec_seq];
            dec_input[0] = TOK_BOS;
            for (j, &b) in resp_bytes[..ri].iter().enumerate() {
                if j + 1 < dec_seq { dec_input[j + 1] = b; }
            }

            let mut tgt = vec![TOK_PAD; dec_seq];
            if ri < dec_seq { tgt[ri] = target_byte; }

            let mut wt = vec![0.0f32; dec_seq];
            if ri < dec_seq { wt[ri] = 1.0; }

            batch_goals.extend_from_slice(goal_padded);
            batch_input.extend_from_slice(&dec_input);
            batch_target.extend_from_slice(&tgt);
            batch_weight.extend_from_slice(&wt);
        }

        let goal_tensor = Tensor::from_vec(batch_goals, (batch_size, enc_seq), device)?;
        let input_tensor = Tensor::from_vec(batch_input, (batch_size, dec_seq), device)?;
        let target_tensor = Tensor::from_vec(batch_target, (batch_size, dec_seq), device)?;
        let weight_tensor = Tensor::from_vec(batch_weight, (batch_size, dec_seq), device)?;

        let logits = brain.forward(&goal_tensor, &input_tensor)?;
        let loss = weighted_cross_entropy(&logits, &target_tensor, 0.0, Some(&weight_tensor))?;
        let loss_val = loss.to_scalar::<f32>()?;
        losses.push(loss_val);

        trainer.backward_step(&loss)?;

        if step % 200 == 0 || step == config.da_steps - 1 {
            let window = losses.len().min(10);
            let avg: f32 = losses.iter().rev().take(window).sum::<f32>() / window as f32;
            let rss_kb = std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|s| s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok()))
                .unwrap_or(0);
            eprintln!("[brain-DA] step {step}/{} avg_loss={avg:.4} lr={:.2e} rss={:.0}MB",
                config.da_steps, trainer.current_lr(), rss_kb as f64 / 1024.0);

            // Early stopping + best checkpoint check
            if step > 0 {
                if let EarlyStopAction::Stop = early_stop.check(avg, step, varmap) {
                    eprintln!("[brain-DA] Early stopped at step {step}/{}", config.da_steps);
                    break;
                }
            }
        }
    }

    trainer.print_timer("brain-da");
    Ok(losses)
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty <= 1.0 { return; }
    let mut counts = std::collections::HashMap::<u32, u32>::new();
    for &id in generated {
        *counts.entry(id).or_insert(0) += 1;
    }
    for (&id, &count) in &counts {
        if (id as usize) < logits.len() {
            let p = penalty.powi(count as i32);
            if logits[id as usize] > 0.0 {
                logits[id as usize] /= p;
            } else {
                logits[id as usize] *= p;
            }
        }
    }
}

fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() { return; }
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let threshold = indexed[k - 1].1;
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

pub fn brain_generate(
    brain: &Brain,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    use_memory: bool,
    device: &Device,
) -> Result<String> {
    let tok = TalkTokenizer;
    let enc_seq = brain.config.encoder_seq_len;
    let dec_seq = brain.config.decoder_seq_len;

    let goal_ids = tok.encode(prompt);
    let goal_padded = tok.pad_or_truncate(&goal_ids, enc_seq);
    let goal_tensor = Tensor::from_vec(goal_padded, (1, enc_seq), device)?;

    let concept_vec = brain.encode_concept(&goal_tensor)?;
    let prefix = if use_memory {
        let cv = concept_vec.squeeze(0)?.to_vec1::<f32>()?;
        let memories = brain.memory_bank.retrieve_vecs(&cv, brain.config.memory_k);
        if memories.is_empty() {
            brain.build_prefix(&concept_vec, None)?
        } else {
            let mem_data: Vec<f32> = memories.iter()
                .flat_map(|m| m.iter().copied())
                .collect();
            let n_mems = memories.len();
            let mem_tensor = Tensor::from_vec(
                mem_data, (1, n_mems, brain.config.d_model), device,
            )?;
            brain.build_prefix(&concept_vec, Some(&mem_tensor))?
        }
    } else {
        brain.build_prefix(&concept_vec, None)?
    };

    let mut generated: Vec<u32> = vec![TOK_BOS];
    let rep_penalty = 1.3_f32;
    let top_k = 40_usize;

    for _ in 0..max_tokens {
        let padded = right_pad(&generated, dec_seq);
        let input = Tensor::from_vec(padded, (1, dec_seq), device)?;

        let logits = brain.language_decoder.forward_with_prefix(&prefix, &input)?;

        let read_pos = generated.len() - 1;
        let mut next_logits = logits.i((0, read_pos))?.to_vec1::<f32>()?;

        apply_repetition_penalty(&mut next_logits, &generated, rep_penalty);
        apply_top_k(&mut next_logits, top_k);

        if (TOK_PAD as usize) < next_logits.len() {
            next_logits[TOK_PAD as usize] = f32::NEG_INFINITY;
        }

        let next_id = if temperature < 0.01 {
            next_logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(TOK_EOS)
        } else {
            sample_with_temperature(&next_logits, temperature)
        };

        if next_id == TOK_EOS { break; }
        generated.push(next_id);
    }

    let response_ids = &generated[1..];
    Ok(tok.decode(response_ids))
}

pub fn brain_diagnostic_decode(
    brain: &Brain,
    prompt: &str,
    max_tokens: usize,
    device: &Device,
) -> Result<Vec<String>> {
    let tok = TalkTokenizer;
    let enc_seq = brain.config.encoder_seq_len;
    let dec_seq = brain.config.decoder_seq_len;

    let goal_ids = tok.encode(prompt);
    let goal_padded = tok.pad_or_truncate(&goal_ids, enc_seq);
    let goal_tensor = Tensor::from_vec(goal_padded, (1, enc_seq), device)?;

    let concept_vec = brain.encode_concept(&goal_tensor)?;
    let prefix = brain.build_prefix(&concept_vec, None)?;

    let mut generated: Vec<u32> = vec![TOK_BOS];
    let mut diag_lines = Vec::new();

    for pos in 0..max_tokens {
        let padded = right_pad(&generated, dec_seq);
        let input = Tensor::from_vec(padded, (1, dec_seq), device)?;

        let logits = brain.language_decoder.forward_with_prefix(&prefix, &input)?;
        let read_pos = generated.len() - 1;
        let raw_logits = logits.i((0, read_pos))?.to_vec1::<f32>()?;

        let max_l = raw_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw_logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let greedy_id = indexed[0].0 as u32;
        let greedy_prob = indexed[0].1;

        let byte_char = if greedy_id < 256 {
            let c = greedy_id as u8 as char;
            if c.is_ascii_graphic() || c == ' ' { format!("'{c}'") } else { format!("0x{greedy_id:02x}") }
        } else if greedy_id == TOK_EOS {
            "EOS".to_string()
        } else {
            format!("#{greedy_id}")
        };

        let alts: Vec<String> = indexed[1..6.min(indexed.len())].iter().map(|(id, p)| {
            let c = if *id < 256 {
                let ch = *id as u8 as char;
                if ch.is_ascii_graphic() || ch == ' ' { format!("'{ch}'") } else { format!("0x{id:02x}") }
            } else if *id as u32 == TOK_EOS {
                "EOS".into()
            } else {
                format!("#{id}")
            };
            format!("{c}({p:.3})")
        }).collect();

        diag_lines.push(format!(
            "  pos={pos:3} tok={byte_char:6} p={greedy_prob:.4} alts=[{}]",
            alts.join(", ")
        ));

        if greedy_id == TOK_EOS as u32 { break; }
        generated.push(greedy_id);
    }

    Ok(diag_lines)
}

pub fn eval_brain_talk(
    brain: &Brain,
    temperature: f64,
    device: &Device,
) -> Result<Vec<(String, String)>> {
    let prompts = [
        "hello",
        "who are you",
        "how do you check the build",
        "what is a transformer",
        "tell me a joke",
        "explain how you handle a new codebase",
        "what do you think about rust",
        "help",
    ];

    let mut results = Vec::new();
    for prompt in &prompts {
        let response = brain_generate(brain, prompt, 200, temperature, false, device)?;
        results.push((prompt.to_string(), response));
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Policy Curriculum (64 deterministic tasks)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PolicyTask {
    pub goal: String,
    pub intent: usize,
    pub actions: [usize; PLAN_STEPS],
    pub patterns: [usize; PLAN_STEPS],
    pub files: [usize; PLAN_STEPS],
    pub picks: [usize; PLAN_STEPS],
}

fn ptask(
    goal: &str, intent: usize,
    actions: [usize; PLAN_STEPS],
    patterns: [usize; PLAN_STEPS],
    files: [usize; PLAN_STEPS],
    picks: [usize; PLAN_STEPS],
) -> PolicyTask {
    PolicyTask {
        goal: goal.to_string(), intent, actions, patterns, files, picks,
    }
}

fn core_tasks() -> Vec<PolicyTask> {
    vec![
        ptask("patch docs/ALL_TASKS.md --dry-run", INTENT_PATCH_DRY_RUN,
            [ACT_PATCH_DRY_RUN, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("verify lean suite", INTENT_LEAN_SUITE,
            [ACT_LEAN_SUITE, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("docs lint", INTENT_DOCS_LINT,
            [ACT_DOCS_LINT, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("prove x*(y+z) == x*y + x*z", INTENT_PROVE_ALGEBRA,
            [ACT_PROVE_ALGEBRA, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("lean suite", INTENT_LEAN_SUITE,
            [ACT_LEAN_SUITE, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("search jarviscmd", INTENT_REPO_SEARCH,
            [ACT_RG, 0,0,0,0,0], [1,0,0,0,0,0], [0;6], [0;6]),
        ptask("list files", INTENT_REPO_LIST,
            [ACT_REPO_LIST, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("check lean suite", INTENT_LEAN_SUITE,
            [ACT_LEAN_SUITE, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("cargo check", INTENT_CARGO_CHECK,
            [ACT_CARGO_CHECK, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("validate the workspace", INTENT_RUN_TESTS,
            [ACT_CARGO_TEST, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("fix tests", INTENT_FIX_TESTS,
            [ACT_FIX_TESTS, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("search jarviscmd and then read it and then docs lint", INTENT_COMPOSITE,
            [ACT_RG, ACT_REPO_READ, ACT_DOCS_LINT, 0,0,0], [1,0,0,0,0,0], [0;6], [0;6]),
        ptask("search gpupolicy and then open it and then docs lint", INTENT_COMPOSITE,
            [ACT_RG, ACT_REPO_READ, ACT_DOCS_LINT, 0,0,0], [4,0,0,0,0,0], [0;6], [0;6]),
        ptask("store preference: favorite color is blue", INTENT_MEMORY_ADD,
            [ACT_MEMORY_ADD, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("retrieve favorite color", INTENT_MEMORY_SEARCH,
            [ACT_MEMORY_SEARCH, 0,0,0,0,0], [0;6], [0;6], [0;6]),
        ptask("preflight", INTENT_COMPOSITE,
            [ACT_CARGO_TEST, ACT_DOCS_LINT, 0,0,0,0], [0;6], [0;6], [0;6]),
    ]
}

fn expansion_tasks() -> Vec<PolicyTask> {
    let mut tasks = Vec::new();

    let pats = [
        (1, "jarviscmd"), (2, "policyrunctx"), (3, "runid"),
        (4, "gpupolicy"), (5, "patch_apply"),
    ];
    for &(pid, pat) in &pats {
        for verb in ["search", "find", "look for"] {
            tasks.push(ptask(
                &format!("{verb} {pat}"), INTENT_REPO_SEARCH,
                [ACT_RG, 0,0,0,0,0], [pid,0,0,0,0,0], [0;6], [0;6],
            ));
        }
    }

    tasks.push(ptask("run tests", INTENT_RUN_TESTS,
        [ACT_CARGO_TEST, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("check the build", INTENT_CARGO_CHECK,
        [ACT_CARGO_CHECK, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("repair the test suite", INTENT_FIX_TESTS,
        [ACT_FIX_TESTS, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("verify the workspace", INTENT_RUN_TESTS,
        [ACT_CARGO_TEST, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("sync docs", INTENT_DOCS_LINT,
        [ACT_DOCS_LINT, 0,0,0,0,0], [0;6], [0;6], [0;6]));

    tasks.push(ptask("remember preference: dark mode enabled", INTENT_MEMORY_ADD,
        [ACT_MEMORY_ADD, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("remember fact: gpu policy", INTENT_MEMORY_ADD,
        [ACT_MEMORY_ADD, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("remember procedure: sync docs", INTENT_MEMORY_ADD,
        [ACT_MEMORY_ADD, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("recall dark mode setting", INTENT_MEMORY_SEARCH,
        [ACT_MEMORY_SEARCH, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("recall gpu policy", INTENT_MEMORY_SEARCH,
        [ACT_MEMORY_SEARCH, 0,0,0,0,0], [0;6], [0;6], [0;6]));

    tasks.push(ptask("prove a*(b+c) == a*b + a*c", INTENT_PROVE_ALGEBRA,
        [ACT_PROVE_ALGEBRA, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("prove (x+y)*z == x*z + y*z", INTENT_PROVE_ALGEBRA,
        [ACT_PROVE_ALGEBRA, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("verify lean proofs", INTENT_LEAN_SUITE,
        [ACT_LEAN_SUITE, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("run lean verification", INTENT_LEAN_SUITE,
        [ACT_LEAN_SUITE, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("check formal proofs", INTENT_LEAN_SUITE,
        [ACT_LEAN_SUITE, 0,0,0,0,0], [0;6], [0;6], [0;6]));

    tasks.push(ptask("search runid and then open it", INTENT_COMPOSITE,
        [ACT_RG, ACT_REPO_READ, 0,0,0,0], [3,0,0,0,0,0], [0;6], [0;6]));
    tasks.push(ptask("search policyrunctx and then open it", INTENT_COMPOSITE,
        [ACT_RG, ACT_REPO_READ, 0,0,0,0], [2,0,0,0,0,0], [0;6], [0;6]));
    tasks.push(ptask("find patch_apply and then open it", INTENT_COMPOSITE,
        [ACT_RG, ACT_REPO_READ, 0,0,0,0], [5,0,0,0,0,0], [0;6], [0;6]));
    tasks.push(ptask("search jarviscmd and then open it", INTENT_COMPOSITE,
        [ACT_RG, ACT_REPO_READ, 0,0,0,0], [1,0,0,0,0,0], [0;6], [0;6]));
    tasks.push(ptask("cargo test and then docs lint", INTENT_COMPOSITE,
        [ACT_CARGO_TEST, ACT_DOCS_LINT, 0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("fix tests and then cargo check", INTENT_COMPOSITE,
        [ACT_FIX_TESTS, ACT_CARGO_CHECK, 0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("remember preference: theme is dark and then recall theme", INTENT_COMPOSITE,
        [ACT_MEMORY_ADD, ACT_MEMORY_SEARCH, 0,0,0,0], [0;6], [0;6], [0;6]));

    tasks.push(ptask("hello", INTENT_HELLO,
        [ACT_TALK, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("hi", INTENT_HELLO,
        [ACT_TALK, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("hey", INTENT_HELLO,
        [ACT_TALK, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("greetings", INTENT_HELLO,
        [ACT_TALK, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("open the policy code", INTENT_REPO_READ,
        [ACT_REPO_READ, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("list all files", INTENT_REPO_LIST,
        [ACT_REPO_LIST, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("run the test suite", INTENT_RUN_TESTS,
        [ACT_CARGO_TEST, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("execute cargo test", INTENT_RUN_TESTS,
        [ACT_CARGO_TEST, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("lint documentation", INTENT_DOCS_LINT,
        [ACT_DOCS_LINT, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("patch README.md --dry-run", INTENT_PATCH_DRY_RUN,
        [ACT_PATCH_DRY_RUN, 0,0,0,0,0], [0;6], [0;6], [0;6]));
    tasks.push(ptask("patch docs/PAPER.md --dry-run", INTENT_PATCH_DRY_RUN,
        [ACT_PATCH_DRY_RUN, 0,0,0,0,0], [0;6], [0;6], [0;6]));

    tasks
}

pub fn curriculum(full: bool) -> Vec<PolicyTask> {
    let mut tasks = core_tasks();
    if full {
        tasks.extend(expansion_tasks());
    }
    tasks
}

// ---------------------------------------------------------------------------
// Policy Training
// ---------------------------------------------------------------------------

fn encode_goal(goal: &str, seq_len: usize, device: &Device) -> Result<Tensor> {
    let bytes: Vec<u32> = goal.bytes().map(|b| b as u32).collect();
    let mut padded = vec![0u32; seq_len];
    let len = bytes.len().min(seq_len);
    padded[..len].copy_from_slice(&bytes[..len]);
    Ok(Tensor::new(padded, device)?.unsqueeze(0)?)
}

fn as_3d(t: &Tensor) -> Result<Tensor> {
    let dims = t.dims();
    if dims.len() == 2 {
        Ok(t.unsqueeze(1)?)
    } else {
        Ok(t.clone())
    }
}

fn policy_loss(
    output: &PolicyOutput,
    task: &PolicyTask,
    device: &Device,
) -> Result<Tensor> {
    let intent_target = Tensor::new(&[task.intent as u32], device)?;
    let intent_loss = weighted_cross_entropy(
        &as_3d(&output.intent_logits)?, &intent_target, 0.0, None,
    )?;
    let mut total = (intent_loss * LOSS_W_INTENT)?;

    for s in 0..PLAN_STEPS {
        let act_logits_s = output.act_logits.i((.., s..s+1, ..))?;
        let act_target = Tensor::new(&[task.actions[s] as u32], device)?;
        let act_loss = weighted_cross_entropy(
            &act_logits_s, &act_target, 0.0, None,
        )?;
        total = (total + act_loss * LOSS_W_ACT)?;
    }

    for s in 0..PLAN_STEPS {
        if task.actions[s] == ACT_RG {
            let pat_logits_s = output.pat_logits.i((.., s..s+1, ..))?;
            let pat_target = Tensor::new(&[task.patterns[s] as u32], device)?;
            let pat_loss = weighted_cross_entropy(
                &pat_logits_s, &pat_target, 0.0, None,
            )?;
            total = (total + pat_loss * LOSS_W_PAT)?;
        }
        if task.actions[s] == ACT_REPO_READ {
            let file_logits_s = output.file_logits.i((.., s..s+1, ..))?;
            let file_target = Tensor::new(&[task.files[s] as u32], device)?;
            let file_loss = weighted_cross_entropy(
                &file_logits_s, &file_target, 0.0, None,
            )?;
            total = (total + file_loss * LOSS_W_FILE)?;

            if task.files[s] == 0 {
                let pick_logits_s = output.pick_logits.i((.., s..s+1, ..))?;
                let pick_target = Tensor::new(&[task.picks[s] as u32], device)?;
                let pick_loss = weighted_cross_entropy(
                    &pick_logits_s, &pick_target, 0.0, None,
                )?;
                total = (total + pick_loss * LOSS_W_PICK)?;
            }
        }
    }

    Ok(total)
}

pub fn train_and_bench_cpu(cfg: &PolicyConfig) -> Result<(usize, usize)> {
    train_and_bench(cfg, &Device::Cpu)
}

pub fn train_and_bench(cfg: &PolicyConfig, device: &Device) -> Result<(usize, usize)> {
    let tasks = curriculum(cfg.full_curriculum);
    let n_tasks = tasks.len();

    eprintln!("[BRAIN POLICY] {} tasks, {} steps, d={}, layers={}, lr={:.1e}",
        n_tasks, cfg.steps, cfg.d_model, cfg.n_layers, cfg.lr);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let tcfg = TransformerConfig {
        d_model: cfg.d_model,
        n_layers: cfg.n_layers,
        n_heads: cfg.n_heads,
        d_ff: cfg.d_ff,
        vocab_size: BYTE_VOCAB,
        max_seq_len: cfg.seq_len,
    };
    let backbone = WiredTransformer::from_vb(tcfg, vb.pp("backbone"), device)?;

    let d = cfg.d_model;
    let head_intent = linear(d, NUM_INTENTS, vb.pp("head_intent"))?;
    let head_actions = linear(d, NUM_ACTIONS * PLAN_STEPS, vb.pp("head_actions"))?;
    let head_patterns = linear(d, NUM_PATTERNS * PLAN_STEPS, vb.pp("head_patterns"))?;
    let head_files = linear(d, NUM_FILES * PLAN_STEPS, vb.pp("head_files"))?;
    let head_picks = linear(d, NUM_PICKS * PLAN_STEPS, vb.pp("head_picks"))?;

    let total_params: usize = varmap.all_vars().iter()
        .map(|v| v.as_tensor().elem_count()).sum();
    eprintln!("[BRAIN POLICY] params: {} ({:.1}K)", total_params, total_params as f64 / 1e3);

    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW { lr: cfg.lr, weight_decay: 0.01, ..Default::default() },
    )?;
    let mut scheduler = CosineScheduler::new(cfg.lr, cfg.lr * 0.01, (cfg.steps / 10).max(1), cfg.steps);

    let mut early_stop = EarlyStopping::new(
        cfg.early_stop_loss,
        cfg.early_stop_patience,
        if cfg.early_stop_loss > 0.0 { Some("policy_best.safetensors".to_string()) } else { None },
    );
    let mut recent_losses: Vec<f32> = Vec::new();

    for step in 0..cfg.steps {
        let lr = scheduler.step();
        optimizer.set_learning_rate(lr);

        let task_idx = step % n_tasks;
        let t = &tasks[task_idx];

        let input = encode_goal(&t.goal, cfg.seq_len, device)?;

        // Inline forward for standalone policy training
        let hidden = backbone.encode(&input)?;
        let pooled = hidden.mean(1)?;
        let b = pooled.dim(0)?;

        let intent_logits = head_intent.forward(&pooled)?;
        let act_logits = head_actions.forward(&pooled)?.reshape((b, PLAN_STEPS, NUM_ACTIONS))?;
        let pat_logits = head_patterns.forward(&pooled)?.reshape((b, PLAN_STEPS, NUM_PATTERNS))?;
        let file_logits = head_files.forward(&pooled)?.reshape((b, PLAN_STEPS, NUM_FILES))?;
        let pick_logits = head_picks.forward(&pooled)?.reshape((b, PLAN_STEPS, NUM_PICKS))?;

        let output = PolicyOutput { intent_logits, act_logits, pat_logits, file_logits, pick_logits };
        let loss = policy_loss(&output, t, device)?;
        let lv = loss.to_scalar::<f32>()?;
        recent_losses.push(lv);
        optimizer.backward_step(&loss)?;

        if step % 500 == 0 || step == cfg.steps - 1 {
            let window = recent_losses.len().min(10);
            let avg: f32 = recent_losses.iter().rev().take(window).sum::<f32>() / window as f32;
            eprintln!("[step {step:>5}/{steps}] loss={avg:.4} lr={lr:.2e}", steps = cfg.steps);

            // Early stopping + best checkpoint check
            if step > 0 {
                if let EarlyStopAction::Stop = early_stop.check(avg, step, &varmap) {
                    eprintln!("[BRAIN POLICY] Early stopped at step {step}/{}", cfg.steps);
                    break;
                }
            }
        }
    }

    // Benchmark using inline forward
    let total = tasks.len();
    let mut correct = 0;

    for t in &tasks {
        let input = encode_goal(&t.goal, cfg.seq_len, device)?;
        let hidden = backbone.encode(&input)?;
        let pooled = hidden.mean(1)?;
        let b = pooled.dim(0)?;

        let intent_logits = head_intent.forward(&pooled)?;
        let act_logits = head_actions.forward(&pooled)?.reshape((b, PLAN_STEPS, NUM_ACTIONS))?;

        let intent_pred = intent_logits.squeeze(0)?
            .argmax(0)?
            .to_scalar::<u32>()? as usize;

        let mut act_pred = [0usize; PLAN_STEPS];
        for s in 0..PLAN_STEPS {
            act_pred[s] = act_logits.i((0, s, ..))?
                .argmax(0)?
                .to_scalar::<u32>()? as usize;
        }

        if intent_pred == t.intent && act_pred == t.actions {
            correct += 1;
        }
    }

    eprintln!("[BRAIN POLICY BENCH] {correct}/{total}");
    Ok((correct, total))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_policy_cfg() -> PolicyConfig {
        PolicyConfig::test()
    }

    // --- Talk tokenizer tests ---

    #[test]
    fn test_talk_tokenizer_roundtrip() {
        let tok = TalkTokenizer;
        let text = "Hello, world! I'm JARVIS.";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_talk_tokenizer_vocab_size() {
        let tok = TalkTokenizer;
        assert_eq!(tok.vocab_size(), 259);
    }

    #[test]
    fn test_corpus_has_dialogues() {
        let corpus = load_corpus();
        assert!(corpus.len() >= 40, "Need at least 40 dialogues, got {}", corpus.len());
    }

    #[test]
    fn test_sample_with_temperature() {
        let logits = vec![0.0f32; 259];
        let id = sample_with_temperature(&logits, 1.0);
        assert!(id < 259);

        let mut logits2 = vec![0.0f32; 259];
        logits2[42] = 100.0;
        let id2 = sample_with_temperature(&logits2, 0.01);
        assert_eq!(id2, 42);
    }

    // --- Brain regions tests ---

    #[test]
    fn test_brain_construction() -> Result<()> {
        let device = Device::Cpu;
        let config = BrainConfig::test_brain();
        let pcfg = test_policy_cfg();
        let varmap = VarMap::new();
        let brain = Brain::new(config, &pcfg, &varmap, &device)?;

        let total: usize = varmap.all_vars().iter()
            .map(|v| v.as_tensor().elem_count())
            .sum();
        assert!(total > 0, "Should have parameters");
        assert_eq!(brain.memory_count(), 0);
        Ok(())
    }

    #[test]
    fn test_concept_encoding() -> Result<()> {
        let device = Device::Cpu;
        let config = BrainConfig::test_brain();
        let pcfg = test_policy_cfg();
        let varmap = VarMap::new();
        let brain = Brain::new(config.clone(), &pcfg, &varmap, &device)?;

        let tok = TalkTokenizer;
        let ids = tok.encode("hello");
        let padded = tok.pad_or_truncate(&ids, config.encoder_seq_len);
        let input = Tensor::from_vec(padded, (1, config.encoder_seq_len), &device)?;

        let cv = brain.encode_concept(&input)?;
        assert_eq!(cv.dims(), &[1, config.d_model]);
        Ok(())
    }

    #[test]
    fn test_prefix_building() -> Result<()> {
        let device = Device::Cpu;
        let config = BrainConfig::test_brain();
        let pcfg = test_policy_cfg();
        let varmap = VarMap::new();
        let brain = Brain::new(config.clone(), &pcfg, &varmap, &device)?;

        let cv = Tensor::zeros((2, config.d_model), DType::F32, &device)?;
        let prefix = brain.build_prefix(&cv, None)?;
        assert_eq!(prefix.dims(), &[2, config.n_concept_tokens, config.d_model]);

        let mem = Tensor::zeros((2, 3, config.d_model), DType::F32, &device)?;
        let prefix = brain.build_prefix(&cv, Some(&mem))?;
        assert_eq!(prefix.dims(), &[2, config.n_concept_tokens + 3, config.d_model]);
        Ok(())
    }

    #[test]
    fn test_forward_no_memory() -> Result<()> {
        let device = Device::Cpu;
        let config = BrainConfig::test_brain();
        let pcfg = test_policy_cfg();
        let varmap = VarMap::new();
        let brain = Brain::new(config.clone(), &pcfg, &varmap, &device)?;

        let batch = 2;
        let goal_ids = Tensor::zeros((batch, config.encoder_seq_len), DType::U32, &device)?;
        let resp_ids = Tensor::zeros((batch, config.decoder_seq_len), DType::U32, &device)?;

        let logits = brain.forward(&goal_ids, &resp_ids)?;
        assert_eq!(logits.dims(), &[batch, config.decoder_seq_len, TALK_VOCAB_SIZE]);
        Ok(())
    }

    #[test]
    fn test_memory_store_retrieve() {
        let mut bank = MemoryBank::new(10);
        bank.store(vec![1.0, 0.0, 0.0], "hello".into());
        bank.store(vec![0.0, 1.0, 0.0], "world".into());
        bank.store(vec![0.9, 0.1, 0.0], "similar".into());

        let results = bank.retrieve_vecs(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert!((results[0][0] - 1.0).abs() < 0.01 || (results[0][0] - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_training_reduces_loss() -> Result<()> {
        let device = Device::Cpu;
        let config = BrainConfig::test_brain();
        let pcfg = test_policy_cfg();
        let (brain, _varmap, losses) = train_brain_talk(&config, &pcfg, &device)?;

        assert!(!losses.is_empty());
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(last < first, "Loss should decrease: first={first} last={last}");
        assert!(brain.memory_count() > 0, "Memory should be populated");
        Ok(())
    }

    #[test]
    fn test_generation_produces_text() -> Result<()> {
        let device = Device::Cpu;
        let config = BrainConfig::test_brain();
        let pcfg = test_policy_cfg();
        let varmap = VarMap::new();
        let brain = Brain::new(config, &pcfg, &varmap, &device)?;

        let response = brain_generate(&brain, "hello", 20, 1.0, false, &device)?;
        let _ = response;
        Ok(())
    }

    #[test]
    fn test_backward_flows_through_encoder() -> Result<()> {
        let device = Device::Cpu;
        let config = BrainConfig::test_brain();
        let pcfg = test_policy_cfg();
        let varmap = VarMap::new();
        let brain = Brain::new(config.clone(), &pcfg, &varmap, &device)?;

        let goal = Tensor::zeros((1, config.encoder_seq_len), DType::U32, &device)?;
        let resp = Tensor::zeros((1, config.decoder_seq_len), DType::U32, &device)?;
        let target = Tensor::zeros((1, config.decoder_seq_len), DType::U32, &device)?;

        let logits = brain.forward(&goal, &resp)?;
        let loss = weighted_cross_entropy(&logits, &target, 0.0, None)?;
        let grads = loss.backward()?;

        let mut encoder_has_grad = false;
        for var in varmap.all_vars() {
            if let Some(_g) = grads.get(var.as_tensor()) {
                encoder_has_grad = true;
                break;
            }
        }
        assert!(encoder_has_grad, "Gradients should flow through encoder");
        Ok(())
    }

    // --- Policy tests ---

    #[test]
    fn test_policy_constants() {
        assert_eq!(NUM_INTENTS, 16);
        assert_eq!(NUM_ACTIONS, 16);
        assert_eq!(PLAN_STEPS, 6);
        assert_eq!(NUM_PATTERNS, 6);
        assert_eq!(NUM_FILES, 10);
    }

    #[test]
    fn test_core_tasks_count() {
        assert_eq!(core_tasks().len(), 16);
    }

    #[test]
    fn test_full_curriculum_count() {
        assert_eq!(curriculum(true).len(), 64);
    }

    #[test]
    fn test_derive_labels_hello() {
        let tasks = curriculum(true);
        let hello = tasks.iter().find(|t| t.goal == "hello").unwrap();
        assert_eq!(hello.intent, INTENT_HELLO);
        assert_eq!(hello.actions, [ACT_TALK, 0,0,0,0,0]);
    }

    #[test]
    fn test_derive_labels_composite() {
        let t = &core_tasks()[11];
        assert_eq!(t.intent, INTENT_COMPOSITE);
        assert_eq!(t.actions, [ACT_RG, ACT_REPO_READ, ACT_DOCS_LINT, 0,0,0]);
        assert_eq!(t.patterns[0], 1);
    }

    #[test]
    fn test_policy_construction() -> Result<()> {
        let cfg = test_policy_cfg();
        let config = BrainConfig::test_brain();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let brain = Brain::new(config, &cfg, &varmap, &device)?;
        let total: usize = varmap.all_vars().iter()
            .map(|v| v.as_tensor().elem_count()).sum();
        assert!(total > 0, "model should have parameters");

        // Test classify
        let input = encode_goal("hello", cfg.seq_len, &device)?;
        let output = brain.classify(&input)?;
        assert_eq!(output.intent_logits.dims(), &[1, NUM_INTENTS]);
        assert_eq!(output.act_logits.dims(), &[1, PLAN_STEPS, NUM_ACTIONS]);
        Ok(())
    }

    #[test]
    fn test_policy_train_loss_decreases() -> Result<()> {
        let cfg = PolicyConfig { steps: 200, ..test_policy_cfg() };
        let device = Device::Cpu;
        let tasks = core_tasks();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let tcfg = TransformerConfig {
            d_model: cfg.d_model,
            n_layers: cfg.n_layers,
            n_heads: cfg.n_heads,
            d_ff: cfg.d_ff,
            vocab_size: BYTE_VOCAB,
            max_seq_len: cfg.seq_len,
        };
        let backbone = WiredTransformer::from_vb(tcfg, vb.pp("backbone"), &device)?;

        let d = cfg.d_model;
        let hi = linear(d, NUM_INTENTS, vb.pp("head_intent"))?;
        let ha = linear(d, NUM_ACTIONS * PLAN_STEPS, vb.pp("head_actions"))?;
        let hp = linear(d, NUM_PATTERNS * PLAN_STEPS, vb.pp("head_patterns"))?;
        let hf = linear(d, NUM_FILES * PLAN_STEPS, vb.pp("head_files"))?;
        let hk = linear(d, NUM_PICKS * PLAN_STEPS, vb.pp("head_picks"))?;

        let mut opt = candle_nn::AdamW::new(
            varmap.all_vars(),
            candle_nn::ParamsAdamW { lr: cfg.lr, ..Default::default() },
        )?;

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for step in 0..cfg.steps {
            let t = &tasks[step % tasks.len()];
            let input = encode_goal(&t.goal, cfg.seq_len, &device)?;

            let hidden = backbone.encode(&input)?;
            let pooled = hidden.mean(1)?;
            let b = pooled.dim(0)?;

            let intent_logits = hi.forward(&pooled)?;
            let act_logits = ha.forward(&pooled)?.reshape((b, PLAN_STEPS, NUM_ACTIONS))?;
            let pat_logits = hp.forward(&pooled)?.reshape((b, PLAN_STEPS, NUM_PATTERNS))?;
            let file_logits = hf.forward(&pooled)?.reshape((b, PLAN_STEPS, NUM_FILES))?;
            let pick_logits = hk.forward(&pooled)?.reshape((b, PLAN_STEPS, NUM_PICKS))?;

            let output = PolicyOutput { intent_logits, act_logits, pat_logits, file_logits, pick_logits };
            let loss = policy_loss(&output, t, &device)?;
            let lv = loss.to_scalar::<f32>()?;
            if step == 0 { first_loss = lv; }
            last_loss = lv;
            opt.backward_step(&loss)?;
        }

        assert!(last_loss < first_loss,
            "loss should decrease: {first_loss:.4} -> {last_loss:.4}");
        Ok(())
    }

    #[test]
    fn test_policy_bench_passes() -> Result<()> {
        let cfg = test_policy_cfg();
        let device = Device::Cpu;
        let (correct, total) = train_and_bench(&cfg, &device)?;
        assert_eq!(total, 16);
        assert_eq!(correct, total,
            "brain policy bench should pass {total}/{total}, got {correct}/{total}");
        Ok(())
    }

    /// DEEP-DEBUG DIAGNOSTIC: Traces data flow through every stage of the
    /// encoder→projector→decoder pipeline. Finds exactly where information is lost.
    #[test]
    fn test_diagnostic_dataflow_trace() -> Result<()> {
        let device = Device::Cpu;
        let config = BrainConfig::default_brain();
        let pcfg = test_policy_cfg();
        let varmap = VarMap::new();
        let brain = Brain::new(config.clone(), &pcfg, &varmap, &device)?;
        let tok = TalkTokenizer;
        let corpus = load_corpus();

        let test_prompts: Vec<&str> = vec![
            "hello", "search for bugs", "write a test", "good morning",
            "what can you do", "run the pipeline", "deploy to production",
            "explain transformers", "fix the memory leak", "compile with cuda",
        ];
        let n = test_prompts.len();

        eprintln!("\n======================================================================");
        eprintln!("DEEP-DEBUG: Data Flow Trace Through Encoder->Projector->Decoder");
        eprintln!("======================================================================");

        // -- STAGE 1: Input Diversity --
        eprintln!("\n-- STAGE 1: Raw Input Tokens --");
        let mut goal_tensors: Vec<Tensor> = Vec::new();
        for prompt in &test_prompts {
            let ids = tok.encode(prompt);
            let padded = tok.pad_or_truncate(&ids, config.encoder_seq_len);
            let n_content = ids.len();
            let n_pad = config.encoder_seq_len - n_content;
            eprintln!("  {:30} -> {} content, {} pad, last_tok={}",
                prompt, n_content, n_pad, padded[config.encoder_seq_len - 1]);
            let t = Tensor::from_vec(padded, (1, config.encoder_seq_len), &device)?;
            goal_tensors.push(t);
        }

        let mut input_diffs = 0usize;
        let mut input_pairs = 0usize;
        for i in 0..n {
            for j in (i+1)..n {
                let a = goal_tensors[i].to_vec2::<u32>()?;
                let b = goal_tensors[j].to_vec2::<u32>()?;
                let diff: usize = a[0].iter().zip(b[0].iter()).filter(|(x, y)| x != y).count();
                input_diffs += diff;
                input_pairs += 1;
            }
        }
        eprintln!("  Avg pairwise hamming distance: {:.1} / {} positions",
            input_diffs as f64 / input_pairs as f64, config.encoder_seq_len);

        // -- STAGE 2: After Token Embedding (last position only) --
        eprintln!("\n-- STAGE 2: After Token Embedding (last position) --");
        let mut emb_vecs: Vec<Vec<f32>> = Vec::new();
        for (i, t) in goal_tensors.iter().enumerate() {
            let emb = brain.concept_encoder.embed_tokens(t)?;
            let last_pos = config.encoder_seq_len - 1;
            let last_emb = emb.i((0, last_pos))?.to_vec1::<f32>()?;
            let norm: f64 = last_emb.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            eprintln!("  {:30} -> last_emb norm={:.4}", test_prompts[i], norm);
            emb_vecs.push(last_emb);
        }
        let emb_sim = diag_avg_cosine(&emb_vecs);
        eprintln!("  Avg pairwise cosine sim (embeddings): {:.6}", emb_sim);

        // Also: avg cosine sim of ALL position embeddings (full sequence)
        eprintln!("\n-- STAGE 2b: Embedding diversity across ALL positions --");
        let mut all_pos_vecs: Vec<Vec<f32>> = Vec::new();
        for t in &goal_tensors {
            let emb = brain.concept_encoder.embed_tokens(t)?;
            let flat = emb.flatten_all()?.to_vec1::<f32>()?;
            all_pos_vecs.push(flat);
        }
        let all_pos_sim = diag_avg_cosine(&all_pos_vecs);
        eprintln!("  Avg cosine sim (full {} embeddings flattened): {:.6}",
            config.encoder_seq_len, all_pos_sim);

        // -- STAGE 3: After Encoder (concept_vec) --
        eprintln!("\n-- STAGE 3: After Encoder -> concept_vec --");
        let mut concept_vecs: Vec<Vec<f32>> = Vec::new();
        for (i, t) in goal_tensors.iter().enumerate() {
            let cv = brain.encode_concept(t)?;
            let v = cv.squeeze(0)?.to_vec1::<f32>()?;
            let norm: f64 = v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            eprintln!("  {:30} -> concept_vec norm={:.4}", test_prompts[i], norm);
            concept_vecs.push(v);
        }
        let cv_sim = diag_avg_cosine(&concept_vecs);
        eprintln!("  Avg pairwise cosine sim (concept_vecs): {:.6}", cv_sim);
        eprintln!("  >>> THIS IS THE KEY METRIC. 0.9591 = collapse. <<<");

        let mut pair_sims: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in (i+1)..n {
                let sim = diag_cosine(&concept_vecs[i], &concept_vecs[j]);
                pair_sims.push((i, j, sim));
            }
        }
        pair_sims.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        eprintln!("  Most different pair: ({}, {}) sim={:.6}",
            test_prompts[pair_sims[0].0], test_prompts[pair_sims[0].1], pair_sims[0].2);
        let last_idx = pair_sims.len() - 1;
        eprintln!("  Most similar pair:   ({}, {}) sim={:.6}",
            test_prompts[pair_sims[last_idx].0], test_prompts[pair_sims[last_idx].1], pair_sims[last_idx].2);

        // -- STAGE 4: After Projector (prefix tokens) --
        eprintln!("\n-- STAGE 4: After Projector -> prefix tokens --");
        let mut prefix_vecs: Vec<Vec<f32>> = Vec::new();
        for (i, t) in goal_tensors.iter().enumerate() {
            let cv = brain.encode_concept(t)?;
            let prefix = brain.project_concepts(&cv)?;
            let flat = prefix.flatten_all()?.to_vec1::<f32>()?;
            let norm: f64 = flat.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            eprintln!("  {:30} -> prefix norm={:.4} ({}x{}d)",
                test_prompts[i], norm, config.n_concept_tokens, config.d_model);
            prefix_vecs.push(flat);
        }
        let prefix_sim = diag_avg_cosine(&prefix_vecs);
        eprintln!("  Avg pairwise cosine sim (prefix): {:.6}", prefix_sim);

        // -- STAGE 5: Decoder Sensitivity --
        eprintln!("\n-- STAGE 5: Decoder Sensitivity --");
        eprintln!("  (Same input tokens, different concept_vecs -> how much do logits differ?)");
        let test_input = {
            let resp = "Hello, sir.";
            let mut ids = vec![TOK_BOS];
            ids.extend(resp.bytes().map(|b| b as u32));
            ids.resize(config.decoder_seq_len, TOK_PAD as u32);
            Tensor::from_vec(ids, (1, config.decoder_seq_len), &device)?
        };

        let mut logit_vecs: Vec<Vec<f32>> = Vec::new();
        for t in &goal_tensors {
            let cv = brain.encode_concept(t)?;
            let logits = brain.forward_from_concept(&cv, &test_input)?;
            let pos1_logits = logits.i((0, 1))?.to_vec1::<f32>()?;
            logit_vecs.push(pos1_logits);
        }
        let logit_sim = diag_avg_cosine(&logit_vecs);
        eprintln!("  Avg pairwise cosine sim of logits@pos1: {:.6}", logit_sim);
        eprintln!("  (1.0 = decoder ignores prefix, <0.99 = decoder uses prefix)");

        let argmaxes: Vec<u32> = logit_vecs.iter().map(|v| {
            v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32).unwrap_or(0)
        }).collect();
        let all_same = argmaxes.iter().all(|&x| x == argmaxes[0]);
        eprintln!("  Argmax predictions: {:?} (all_same={})", argmaxes, all_same);

        // -- STAGE 6: Gradient Flow --
        eprintln!("\n-- STAGE 6: Gradient Flow --");
        let batch_goal = Tensor::cat(&goal_tensors[..3], 0)?;
        let batch_input = {
            let mut all_ids = Vec::new();
            for (_, resp) in corpus.iter().take(3) {
                let mut ids = vec![TOK_BOS];
                ids.extend(resp.bytes().map(|b| b as u32));
                ids.resize(config.decoder_seq_len, TOK_PAD as u32);
                all_ids.extend_from_slice(&ids);
            }
            Tensor::from_vec(all_ids, (3, config.decoder_seq_len), &device)?
        };
        let batch_target = {
            let mut all_ids = Vec::new();
            for (_, resp) in corpus.iter().take(3) {
                let mut ids: Vec<u32> = resp.bytes().map(|b| b as u32).collect();
                ids.push(TOK_EOS);
                ids.resize(config.decoder_seq_len, TOK_PAD as u32);
                all_ids.extend_from_slice(&ids);
            }
            Tensor::from_vec(all_ids, (3, config.decoder_seq_len), &device)?
        };

        let logits = brain.forward(&batch_goal, &batch_input)?;
        let loss = weighted_cross_entropy(&logits, &batch_target, 0.0, None)?;
        let loss_val = loss.to_scalar::<f32>()?;
        eprintln!("  Forward loss (3 samples): {:.4}", loss_val);

        let grads = loss.backward()?;

        let data_lock = varmap.data().lock().unwrap();
        let mut enc_grad_sq = 0.0f64;
        let mut dec_grad_sq = 0.0f64;
        let mut proj_grad_sq = 0.0f64;
        let mut enc_params = 0usize;
        let mut dec_params = 0usize;
        let mut proj_params = 0usize;

        for (name, var) in data_lock.iter() {
            let n_p = var.as_tensor().elem_count();
            if let Some(g) = grads.get(var.as_tensor()) {
                let g_sq_sum = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                let norm = g_sq_sum.sqrt();
                if name.starts_with("enc.") {
                    enc_grad_sq += g_sq_sum;
                    enc_params += n_p;
                } else if name.starts_with("dec.") {
                    dec_grad_sq += g_sq_sum;
                    dec_params += n_p;
                } else if name.starts_with("proj") {
                    proj_grad_sq += g_sq_sum;
                    proj_params += n_p;
                }
                eprintln!("  {:40} params={:>8} grad_norm={:.6e}", name, n_p, norm);
            } else {
                eprintln!("  {:40} params={:>8} NO GRADIENT", name, n_p);
            }
        }
        drop(data_lock);

        eprintln!("\n  Summary:");
        eprintln!("  Encoder  total_grad_norm={:.6e} ({} params)", enc_grad_sq.sqrt(), enc_params);
        eprintln!("  Projector total_grad_norm={:.6e} ({} params)", proj_grad_sq.sqrt(), proj_params);
        eprintln!("  Decoder  total_grad_norm={:.6e} ({} params)", dec_grad_sq.sqrt(), dec_params);
        if enc_params > 0 && dec_params > 0 {
            let enc_per = enc_grad_sq.sqrt() / enc_params as f64;
            let dec_per = dec_grad_sq.sqrt() / dec_params as f64;
            eprintln!("  Per-param ratio (dec/enc): {:.2}x", dec_per / enc_per.max(1e-30));
        }

        eprintln!("\n======================================================================");
        eprintln!("DIAGNOSTIC COMPLETE -- Look for the stage where diversity drops.");
        eprintln!("  Stage 1 (inputs): should have hamming > 5");
        eprintln!("  Stage 2 (embeddings): last-pos sim depends on last byte overlap");
        eprintln!("  Stage 2b (full emb): should be < 0.95 (different content)");
        eprintln!("  Stage 3 (concept_vec): THE BOTTLENECK. If sim > 0.9 here, collapse.");
        eprintln!("  Stage 4 (prefix): amplified or dampened from stage 3?");
        eprintln!("  Stage 5 (logits): does decoder even notice different prefixes?");
        eprintln!("  Stage 6 (gradients): enc grad << dec grad = encoder starved");
        eprintln!("======================================================================\n");

        Ok(())
    }

    /// Minimal gradient flow test: isolate where gradients break
    #[test]
    fn test_diagnostic_gradient_isolation() -> Result<()> {
        let device = Device::Cpu;
        let config = BrainConfig::default_brain();
        let pcfg = test_policy_cfg();
        let varmap = VarMap::new();
        let brain = Brain::new(config.clone(), &pcfg, &varmap, &device)?;
        let tok = TalkTokenizer;

        let ids = tok.encode("hello");
        let padded = tok.pad_or_truncate(&ids, config.encoder_seq_len);
        let goal_t = Tensor::from_vec(padded, (1, config.encoder_seq_len), &device)?;

        eprintln!("\n== GRADIENT ISOLATION TEST ==");

        // Test A: Direct loss on concept_vec → encoder should get gradient
        eprintln!("\n-- Test A: MSE loss on concept_vec --");
        let cv = brain.encode_concept(&goal_t)?;
        let target = Tensor::ones_like(&cv)?;
        let loss_a = (&cv - &target)?.sqr()?.mean_all()?;
        eprintln!("  loss_a = {:.4}", loss_a.to_scalar::<f32>()?);
        let grads_a = loss_a.backward()?;
        let data_lock = varmap.data().lock().unwrap();
        let mut enc_has_grad = false;
        for (name, var) in data_lock.iter() {
            if name.starts_with("enc.") {
                if let Some(g) = grads_a.get(var.as_tensor()) {
                    let norm = g.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                    eprintln!("  {:40} grad_norm={:.6e}", name, norm);
                    if norm > 0.0 { enc_has_grad = true; }
                }
            }
        }
        eprintln!("  Encoder has gradient from direct MSE: {}", enc_has_grad);
        drop(data_lock);

        // Test B: Loss on projected prefix → projector + encoder should get gradient
        eprintln!("\n-- Test B: MSE loss on projected prefix --");
        let cv2 = brain.encode_concept(&goal_t)?;
        let prefix = brain.project_concepts(&cv2)?;
        let target_p = Tensor::ones_like(&prefix)?;
        let loss_b = (&prefix - &target_p)?.sqr()?.mean_all()?;
        eprintln!("  loss_b = {:.4}", loss_b.to_scalar::<f32>()?);
        let grads_b = loss_b.backward()?;
        let data_lock = varmap.data().lock().unwrap();
        let mut proj_has_grad = false;
        for (name, var) in data_lock.iter() {
            if name.starts_with("concept_proj") {
                if let Some(g) = grads_b.get(var.as_tensor()) {
                    let norm = g.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                    eprintln!("  {:40} grad_norm={:.6e}", name, norm);
                    if norm > 0.0 { proj_has_grad = true; }
                }
            }
            if name.starts_with("enc.") {
                if let Some(g) = grads_b.get(var.as_tensor()) {
                    let norm = g.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                    eprintln!("  {:40} grad_norm={:.6e}", name, norm);
                }
            }
        }
        eprintln!("  Projector has gradient: {}", proj_has_grad);
        drop(data_lock);

        // Test C: Full pipeline → where does it break?
        eprintln!("\n-- Test C: Full pipeline loss --");
        let cv3 = brain.encode_concept(&goal_t)?;
        let test_input = {
            let mut ids = vec![TOK_BOS];
            ids.extend("Hi".bytes().map(|b| b as u32));
            ids.resize(config.decoder_seq_len, TOK_PAD as u32);
            Tensor::from_vec(ids, (1, config.decoder_seq_len), &device)?
        };
        let logits = brain.forward_from_concept(&cv3, &test_input)?;
        let target_ids = {
            let mut ids: Vec<u32> = "Hi".bytes().map(|b| b as u32).collect();
            ids.push(TOK_EOS);
            ids.resize(config.decoder_seq_len, TOK_PAD as u32);
            Tensor::from_vec(ids, (1, config.decoder_seq_len), &device)?
        };
        let loss_c = weighted_cross_entropy(&logits, &target_ids, 0.0, None)?;
        eprintln!("  loss_c = {:.4}", loss_c.to_scalar::<f32>()?);
        let grads_c = loss_c.backward()?;
        let data_lock = varmap.data().lock().unwrap();
        let mut c_enc = 0;
        let mut c_dec = 0;
        let mut c_proj = 0;
        for (name, var) in data_lock.iter() {
            if grads_c.get(var.as_tensor()).is_some() {
                let g = grads_c.get(var.as_tensor()).unwrap();
                let norm = g.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                if name.starts_with("enc.") { c_enc += 1; }
                else if name.starts_with("dec.") { c_dec += 1; }
                else if name.starts_with("concept_proj") || name.starts_with("proj") { c_proj += 1; }
                if norm > 0.001 {
                    eprintln!("  {:40} grad_norm={:.6e}", name, norm);
                }
            }
        }
        eprintln!("  Params with gradient: enc={}, proj={}, dec={}", c_enc, c_proj, c_dec);
        drop(data_lock);

        // Test D: Isolate which op in encode_concept breaks gradients
        eprintln!("\n-- Test D: Isolating the break in encode_concept --");
        let hidden = brain.concept_encoder.encode(&goal_t)?;
        let seq_len = hidden.dim(1)?;
        eprintln!("  hidden shape: {:?}", hidden.dims());

        // Step 1: loss on full hidden → does encoder get grad?
        let loss_d1 = hidden.sqr()?.mean_all()?;
        let grads_d1 = loss_d1.backward()?;
        let data_lock = varmap.data().lock().unwrap();
        let enc_grad_d1 = data_lock.iter()
            .filter(|(n, _)| n.starts_with("enc."))
            .any(|(_, v)| grads_d1.get(v.as_tensor()).is_some());
        eprintln!("  Loss on full hidden: encoder has grad = {}", enc_grad_d1);
        drop(data_lock);

        // Step 2: narrow → loss
        let narrowed = hidden.narrow(1, seq_len - 1, 1)?;
        let loss_d2 = narrowed.sqr()?.mean_all()?;
        let grads_d2 = loss_d2.backward()?;
        let data_lock = varmap.data().lock().unwrap();
        let enc_grad_d2 = data_lock.iter()
            .filter(|(n, _)| n.starts_with("enc."))
            .any(|(_, v)| grads_d2.get(v.as_tensor()).is_some());
        eprintln!("  Loss on narrow(last): encoder has grad = {}", enc_grad_d2);
        drop(data_lock);

        // Step 3: narrow + squeeze → loss
        let squeezed = narrowed.squeeze(1)?;
        let loss_d3 = squeezed.sqr()?.mean_all()?;
        let grads_d3 = loss_d3.backward()?;
        let data_lock = varmap.data().lock().unwrap();
        let enc_grad_d3 = data_lock.iter()
            .filter(|(n, _)| n.starts_with("enc."))
            .any(|(_, v)| grads_d3.get(v.as_tensor()).is_some());
        eprintln!("  Loss on narrow+squeeze: encoder has grad = {}", enc_grad_d3);
        drop(data_lock);

        // Step 4: narrow + squeeze + contiguous → loss
        let contig = squeezed.contiguous()?;
        let loss_d4 = contig.sqr()?.mean_all()?;
        let grads_d4 = loss_d4.backward()?;
        let data_lock = varmap.data().lock().unwrap();
        let enc_grad_d4 = data_lock.iter()
            .filter(|(n, _)| n.starts_with("enc."))
            .any(|(_, v)| grads_d4.get(v.as_tensor()).is_some());
        eprintln!("  Loss on narrow+squeeze+contiguous: encoder has grad = {}", enc_grad_d4);
        drop(data_lock);

        eprintln!("\n== END GRADIENT ISOLATION ==\n");
        Ok(())
    }

    /// Test individual candle-rs operations for gradient flow.
    /// This is the deepest layer of diagnostic — tests ops, not architecture.
    #[test]
    fn test_candle_op_gradient_flow() -> Result<()> {
        use candle_nn::embedding;

        let device = Device::Cpu;

        eprintln!("\n== CANDLE OP GRADIENT FLOW TEST ==\n");

        // Test 0: Does candle_nn::ops::softmax_last_dim preserve gradients?
        eprintln!("-- Test 0: softmax_last_dim gradient check --");
        let v0 = Var::from_tensor(&Tensor::randn(0f32, 1f32, (2, 4), &device)?)?;
        let sm0 = candle_nn::ops::softmax_last_dim(v0.as_tensor())?;
        let loss0 = sm0.sqr()?.mean_all()?;
        let grads0 = loss0.backward()?;
        let sm_has_grad = grads0.get(v0.as_tensor()).is_some();
        eprintln!("  Var through softmax_last_dim has gradient: {}", sm_has_grad);

        // Test 0b: Does manual softmax preserve gradients?
        eprintln!("\n-- Test 0b: manual softmax gradient check --");
        let v0b = Var::from_tensor(&Tensor::randn(0f32, 1f32, (2, 4), &device)?)?;
        let max0b = v0b.as_tensor().max_keepdim(candle_core::D::Minus1)?;
        let shifted = v0b.as_tensor().broadcast_sub(&max0b)?;
        let exp0b = shifted.exp()?;
        let sum0b = exp0b.sum_keepdim(candle_core::D::Minus1)?;
        let sm0b = exp0b.broadcast_div(&sum0b)?;
        let loss0b = sm0b.sqr()?.mean_all()?;
        let grads0b = loss0b.backward()?;
        let msm_has_grad = grads0b.get(v0b.as_tensor()).is_some();
        eprintln!("  Var through manual softmax has gradient: {}", msm_has_grad);

        // Test 1: Does a plain Var get gradients?
        eprintln!("\n-- Test 1: Plain Var → sqr → mean → backward --");
        let v1 = Var::from_tensor(&Tensor::randn(0f32, 1f32, (4, 8), &device)?)?;
        let loss1 = v1.as_tensor().sqr()?.mean_all()?;
        let grads1 = loss1.backward()?;
        let has_grad = grads1.get(v1.as_tensor()).is_some();
        eprintln!("  Var has gradient: {}", has_grad);

        // Test 2: Var → linear (via VarMap) → loss → backward
        eprintln!("\n-- Test 2: VarMap Linear → loss → backward --");
        let vm2 = VarMap::new();
        let vb2 = VarBuilder::from_varmap(&vm2, DType::F32, &device);
        let lin = linear(8, 4, vb2.pp("test_lin"))?;
        let input2 = Tensor::randn(0f32, 1f32, (2, 8), &device)?;
        let out2 = lin.forward(&input2)?;
        let loss2 = out2.sqr()?.mean_all()?;
        let grads2 = loss2.backward()?;
        let data2 = vm2.data().lock().unwrap();
        for (name, var) in data2.iter() {
            let has = grads2.get(var.as_tensor()).is_some();
            eprintln!("  {} has gradient: {}", name, has);
        }
        drop(data2);

        // Test 3: VarMap Embedding → loss → backward
        eprintln!("\n-- Test 3: VarMap Embedding → loss → backward --");
        let vm3 = VarMap::new();
        let vb3 = VarBuilder::from_varmap(&vm3, DType::F32, &device);
        let emb = embedding(32, 8, vb3.pp("test_emb"))?;
        let ids3 = Tensor::from_vec(vec![1u32, 5, 10], (1, 3), &device)?;
        let out3 = emb.forward(&ids3)?;
        let loss3 = out3.sqr()?.mean_all()?;
        let grads3 = loss3.backward()?;
        let data3 = vm3.data().lock().unwrap();
        for (name, var) in data3.iter() {
            let has = grads3.get(var.as_tensor()).is_some();
            eprintln!("  {} has gradient: {}", name, has);
        }
        drop(data3);

        // Test 4: Embedding → contiguous → loss → backward
        eprintln!("\n-- Test 4: Embedding → contiguous → loss → backward --");
        let vm4 = VarMap::new();
        let vb4 = VarBuilder::from_varmap(&vm4, DType::F32, &device);
        let emb4 = embedding(32, 8, vb4.pp("test_emb4"))?;
        let ids4 = Tensor::from_vec(vec![1u32, 5, 10], (1, 3), &device)?;
        let out4 = emb4.forward(&ids4)?.contiguous()?;
        let loss4 = out4.sqr()?.mean_all()?;
        let grads4 = loss4.backward()?;
        let data4 = vm4.data().lock().unwrap();
        for (name, var) in data4.iter() {
            let has = grads4.get(var.as_tensor()).is_some();
            eprintln!("  {} (through contiguous) has gradient: {}", name, has);
        }
        drop(data4);

        // Test 5: Embedding → transpose → contiguous → loss → backward
        eprintln!("\n-- Test 5: Embedding → transpose → contiguous → loss → backward --");
        let vm5 = VarMap::new();
        let vb5 = VarBuilder::from_varmap(&vm5, DType::F32, &device);
        let emb5 = embedding(32, 8, vb5.pp("test_emb5"))?;
        let ids5 = Tensor::from_vec(vec![1u32, 5, 10], (1, 3), &device)?;
        let out5 = emb5.forward(&ids5)?;
        let tr5 = out5.transpose(1, 2)?;
        let ct5 = tr5.contiguous()?;
        let loss5 = ct5.sqr()?.mean_all()?;
        let grads5 = loss5.backward()?;
        let data5 = vm5.data().lock().unwrap();
        for (name, var) in data5.iter() {
            let has = grads5.get(var.as_tensor()).is_some();
            eprintln!("  {} (through transpose+contiguous) has gradient: {}", name, has);
        }
        drop(data5);

        // Test 6: Embedding → RmsNorm → loss → backward
        eprintln!("\n-- Test 6: Embedding → RmsNorm → loss → backward --");
        let vm6 = VarMap::new();
        let vb6 = VarBuilder::from_varmap(&vm6, DType::F32, &device);
        let emb6 = embedding(32, 8, vb6.pp("emb6"))?;
        let norm6 = crate::transformer::GradRmsNorm::new(8, 1e-6, vb6.pp("norm6"))?;
        let ids6 = Tensor::from_vec(vec![1u32, 5, 10], (1, 3), &device)?;
        let out6 = norm6.forward(&emb6.forward(&ids6)?)?;
        let loss6 = out6.sqr()?.mean_all()?;
        let grads6 = loss6.backward()?;
        let data6 = vm6.data().lock().unwrap();
        for (name, var) in data6.iter() {
            let has = grads6.get(var.as_tensor()).is_some();
            eprintln!("  {} has gradient: {}", name, has);
        }
        drop(data6);

        // Test 7: Embedding → Linear → loss → backward
        eprintln!("\n-- Test 7: Embedding → Linear → loss → backward --");
        let vm7 = VarMap::new();
        let vb7 = VarBuilder::from_varmap(&vm7, DType::F32, &device);
        let emb7 = embedding(32, 8, vb7.pp("emb7"))?;
        let lin7 = linear(8, 4, vb7.pp("lin7"))?;
        let ids7 = Tensor::from_vec(vec![1u32, 5, 10], (1, 3), &device)?;
        let out7 = lin7.forward(&emb7.forward(&ids7)?)?;
        let loss7 = out7.sqr()?.mean_all()?;
        let grads7 = loss7.backward()?;
        let data7 = vm7.data().lock().unwrap();
        for (name, var) in data7.iter() {
            let has = grads7.get(var.as_tensor()).is_some();
            eprintln!("  {} has gradient: {}", name, has);
        }
        drop(data7);

        // Test 8: Full mini transformer encode → loss → backward
        eprintln!("\n-- Test 8: Mini WiredTransformer encode → loss → backward --");
        let vm8 = VarMap::new();
        let tcfg = TransformerConfig::tiny(); // d=64, 1 layer, 2 heads
        let t8 = WiredTransformer::new(tcfg, &vm8, &device)?;
        let ids8 = Tensor::from_vec(vec![1u32, 5, 10, 15], (1, 4), &device)?;
        let hidden8 = t8.encode(&ids8)?;
        let loss8 = hidden8.sqr()?.mean_all()?;
        let grads8 = loss8.backward()?;
        let data8 = vm8.data().lock().unwrap();
        let mut any_grad = false;
        for (name, var) in data8.iter() {
            let has = grads8.get(var.as_tensor()).is_some();
            if has { any_grad = true; }
            eprintln!("  {} has gradient: {}", name, has);
        }
        eprintln!("  Any param has gradient: {}", any_grad);
        drop(data8);

        eprintln!("\n== END CANDLE OP TEST ==\n");
        Ok(())
    }

    fn diag_cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
        let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        if na > 1e-8 && nb > 1e-8 { dot / (na * nb) } else { 0.0 }
    }

    fn diag_avg_cosine(vecs: &[Vec<f32>]) -> f64 {
        let mut total = 0.0f64;
        let mut count = 0u64;
        for i in 0..vecs.len() {
            for j in (i+1)..vecs.len() {
                total += diag_cosine(&vecs[i], &vecs[j]);
                count += 1;
            }
        }
        if count > 0 { total / count as f64 } else { 0.0 }
    }
}
