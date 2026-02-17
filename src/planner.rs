// FSM constrained decoder: 17-state finite state machine for plan generation
// Port from V4 plan_lm.rs (718 LOC) -- T-004

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor, D};
use candle_nn::VarMap;
use crate::tokenizer::{PlanTokenizer, TOK_STEP};
use crate::transformer::{TransformerConfig, WiredTransformer};
use crate::training::{EarlyStopping, EarlyStopAction, Trainer, TrainingConfig, weighted_cross_entropy};
use crate::eval::{plan_bench_goals, reference_plan_tokens, score_plan_bench};

// ---------------------------------------------------------------------------
// Plan-LM Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PlanLmConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub batch_size: usize,
    pub sft_steps: usize,
    pub sft_lr: f64,
    pub scheduled_sampling_steps: usize,
    pub ss_lr: f64,
    pub step_token_weight: f64,
    pub action_token_weight: f64,
    pub early_stop_loss: f32,    // 0.0 = disabled
    pub early_stop_patience: usize,
}

impl PlanLmConfig {
    pub fn default_plan() -> Self {
        Self {
            d_model: 512,
            n_layers: 4,
            n_heads: 8,
            d_ff: 2048,
            max_seq_len: 256,
            batch_size: 21,
            sft_steps: 4000,
            sft_lr: 3e-4,
            scheduled_sampling_steps: 1000,
            ss_lr: 1e-4,
            step_token_weight: 0.1,
            action_token_weight: 1.0,
            early_stop_loss: 1e-4,
            early_stop_patience: 3,
        }
    }

    pub fn test_plan() -> Self {
        Self {
            d_model: 64,
            n_layers: 2,
            n_heads: 4,
            d_ff: 128,
            max_seq_len: 128,
            batch_size: 4,
            sft_steps: 200,
            sft_lr: 1e-3,
            scheduled_sampling_steps: 100,
            ss_lr: 5e-4,
            step_token_weight: 0.1,
            action_token_weight: 1.0,
            early_stop_loss: 0.0,
            early_stop_patience: 3,
        }
    }

    /// Phase 2: scaled planner (d=1024, 8 layers).
    pub fn phase2() -> Self {
        Self {
            d_model: 1024,
            n_layers: 8,
            n_heads: 8,
            d_ff: 4096,
            max_seq_len: 512,
            batch_size: 21,
            sft_steps: 8000,
            sft_lr: 2e-4,
            scheduled_sampling_steps: 2000,
            ss_lr: 5e-5,
            step_token_weight: 0.1,
            action_token_weight: 1.0,
            early_stop_loss: 1e-4,
            early_stop_patience: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Constrained Decoder FSM (17 states)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextReturn { Action, BeforeRhs, ProveRhs }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FsmState {
    Start,
    AfterStep,
    AfterActionNoArgs,
    AfterRg,
    AfterRepoRead,
    AfterRepoReadFrom,
    AfterPatch,
    AfterPatchPath,
    AfterMemAdd,
    AfterMemAddKind,
    AfterMemSearch,
    AfterProve,
    AfterProveLhs,
    BeforeRhs,
    AfterProveRhs,
    InText(usize, TextReturn),
    InThink,
}

/// Compute valid token IDs for the current FSM state.
pub fn valid_tokens_for_state(
    state: FsmState,
    tok: &PlanTokenizer,
    steps_emitted: usize,
) -> Vec<u32> {
    match state {
        FsmState::Start => {
            let mut v = vec![tok.eop_id()];
            if steps_emitted < 8 { v.push(tok.step_id()); }
            if steps_emitted == 0 { v.push(tok.id("THINK")); }
            v
        }
        FsmState::AfterStep => tok.action_ids(),
        FsmState::AfterActionNoArgs => {
            let mut v = vec![tok.eop_id()];
            if steps_emitted < 8 { v.push(tok.step_id()); }
            v
        }
        FsmState::AfterRg => tok.pat_ids(),
        FsmState::AfterRepoRead => {
            let mut v = tok.file_ids();
            v.extend(tok.from_ids());
            v
        }
        FsmState::AfterRepoReadFrom => tok.pick_ids(),
        FsmState::AfterPatch => vec![tok.id("PATH")],
        FsmState::AfterPatchPath => {
            let mut v = tok.file_ids();
            v.extend(tok.textlen_ids());
            v
        }
        FsmState::AfterMemAdd => tok.kind_ids(),
        FsmState::AfterMemAddKind | FsmState::AfterMemSearch => {
            let mut v = tok.textlen_ids();
            v.push(tok.id("TEXT"));
            v
        }
        FsmState::AfterProve => vec![tok.id("LHS")],
        FsmState::BeforeRhs => vec![tok.id("RHS")],
        FsmState::AfterProveLhs | FsmState::AfterProveRhs => {
            let mut v = tok.textlen_ids();
            v.push(tok.id("TEXT"));
            v
        }
        FsmState::InText(_, _) => tok.payload_ids(),
        FsmState::InThink => {
            let mut v = vec![tok.id("ENDTHINK")];
            for kw in ["NEEDS", "THEN", "IF", "BECAUSE"] {
                v.push(tok.id(kw));
            }
            v.extend(tok.action_ids());
            v
        }
    }
}

/// Transition the FSM state given a token string.
pub fn fsm_transition(state: FsmState, tok_str: &str, steps_emitted: &mut usize) -> FsmState {
    match state {
        FsmState::Start => {
            if tok_str == "STEP" { *steps_emitted += 1; FsmState::AfterStep }
            else if tok_str == "THINK" { FsmState::InThink }
            else { FsmState::Start }
        }
        FsmState::AfterStep => match tok_str {
            "RG" => FsmState::AfterRg,
            "REPOREAD" => FsmState::AfterRepoRead,
            "PATCHDRYRUN" => FsmState::AfterPatch,
            "MEMADD" => FsmState::AfterMemAdd,
            "MEMSEARCH" => FsmState::AfterMemSearch,
            "PROVEALGEBRA" => FsmState::AfterProve,
            _ => FsmState::AfterActionNoArgs,
        },
        FsmState::AfterActionNoArgs => {
            if tok_str == "STEP" { *steps_emitted += 1; FsmState::AfterStep }
            else { FsmState::Start }
        }
        FsmState::AfterRg => FsmState::AfterActionNoArgs,
        FsmState::AfterRepoRead => {
            if tok_str.starts_with("FROM") { FsmState::AfterRepoReadFrom }
            else { FsmState::AfterActionNoArgs }
        }
        FsmState::AfterRepoReadFrom => FsmState::AfterActionNoArgs,
        FsmState::AfterPatch => FsmState::AfterPatchPath,
        FsmState::AfterPatchPath => {
            if tok_str.starts_with("FILE") { FsmState::AfterActionNoArgs }
            else if let Some(ns) = tok_str.strip_prefix("TEXTLEN") {
                if let Ok(n) = ns.parse::<usize>() { FsmState::InText(n.min(32), TextReturn::Action) }
                else { FsmState::AfterPatchPath }
            } else { FsmState::AfterPatchPath }
        }
        FsmState::AfterMemAdd => FsmState::AfterMemAddKind,
        FsmState::AfterMemAddKind | FsmState::AfterMemSearch => {
            if let Some(ns) = tok_str.strip_prefix("TEXTLEN") {
                if let Ok(n) = ns.parse::<usize>() { FsmState::InText(n.min(32), TextReturn::Action) }
                else { FsmState::AfterActionNoArgs }
            } else { FsmState::AfterActionNoArgs }
        }
        FsmState::AfterProve => FsmState::AfterProveLhs,
        FsmState::AfterProveLhs => {
            if let Some(ns) = tok_str.strip_prefix("TEXTLEN") {
                if let Ok(n) = ns.parse::<usize>() { FsmState::InText(n.min(32), TextReturn::BeforeRhs) }
                else { FsmState::AfterProveLhs }
            } else { FsmState::AfterProveLhs }
        }
        FsmState::BeforeRhs => {
            if tok_str == "RHS" { FsmState::AfterProveRhs }
            else { FsmState::BeforeRhs }
        }
        FsmState::AfterProveRhs => {
            if let Some(ns) = tok_str.strip_prefix("TEXTLEN") {
                if let Ok(n) = ns.parse::<usize>() { FsmState::InText(n.min(32), TextReturn::Action) }
                else { FsmState::AfterProveRhs }
            } else { FsmState::AfterProveRhs }
        }
        FsmState::InText(rem, ret) => {
            if rem <= 1 {
                match ret {
                    TextReturn::Action => FsmState::AfterActionNoArgs,
                    TextReturn::BeforeRhs => FsmState::BeforeRhs,
                    TextReturn::ProveRhs => FsmState::AfterActionNoArgs,
                }
            } else {
                FsmState::InText(rem - 1, ret)
            }
        }
        FsmState::InThink => {
            if tok_str == "ENDTHINK" { FsmState::Start } else { FsmState::InThink }
        }
    }
}

// ---------------------------------------------------------------------------
// Constrained Greedy Decoding
// ---------------------------------------------------------------------------

/// Apply FSM mask to logits: set invalid tokens to -inf.
fn apply_fsm_mask(logits: &Tensor, valid_ids: &[u32], vocab_size: usize, device: &Device) -> Result<Tensor> {
    let mut mask_data = vec![f32::NEG_INFINITY; vocab_size];
    for &id in valid_ids {
        if (id as usize) < vocab_size {
            mask_data[id as usize] = 0.0;
        }
    }
    let mask = Tensor::from_vec(mask_data, (1, vocab_size), device)?;
    (logits + mask).map_err(Into::into)
}

/// Greedy constrained decoding: generate plan tokens given a prompt.
pub fn greedy_decode(
    model: &WiredTransformer,
    tok: &PlanTokenizer,
    prompt_ids: &[u32],
    max_tokens: usize,
    device: &Device,
) -> Result<Vec<String>> {
    let seq_len = model.config.max_seq_len;
    let vocab_size = model.config.vocab_size;

    let mut context = tok.pad_or_truncate(prompt_ids, seq_len);
    let mut generated: Vec<String> = Vec::new();
    let mut state = FsmState::Start;
    let mut steps_emitted = 0usize;

    for _ in 0..max_tokens {
        let input = Tensor::from_vec(context.clone(), (1, seq_len), device)?;
        let logits = model.forward(&input)?;

        let last_logits = logits.i((0, seq_len - 1, ..))?;
        let last_logits = last_logits.unsqueeze(0)?;

        let valid = valid_tokens_for_state(state, tok, steps_emitted);
        let masked = apply_fsm_mask(&last_logits, &valid, vocab_size, device)?;

        let token_id = masked.argmax(D::Minus1)?.squeeze(0)?.to_scalar::<u32>()?;
        let token_str = tok.token(token_id).to_string();

        if token_str == "EOP" {
            generated.push("EOP".into());
            break;
        }

        generated.push(token_str.clone());

        state = fsm_transition(state, &token_str, &mut steps_emitted);

        context.remove(0);
        context.push(token_id);
    }

    Ok(generated)
}

// ---------------------------------------------------------------------------
// Training Data Preparation
// ---------------------------------------------------------------------------

/// Prepare training sequences for Plan-LM.
/// Returns Vec of (input_ids, target_ids, weights) — all padded to seq_len.
pub fn prepare_plan_training_data(
    tok: &PlanTokenizer,
    seq_len: usize,
) -> Vec<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    let goals = plan_bench_goals();
    let mut data = Vec::new();

    for (goal, intent) in &goals {
        let prompt_ids = tok.encode_prompt(goal);
        let plan_tokens = reference_plan_tokens(tok, goal, *intent);
        let plan_ids: Vec<u32> = plan_tokens.iter().map(|t| tok.id(t)).collect();

        let mut full_ids: Vec<u32> = Vec::new();
        full_ids.extend_from_slice(&prompt_ids);
        full_ids.extend_from_slice(&plan_ids);

        if full_ids.len() < 2 { continue; }

        let input_raw: Vec<u32> = full_ids[..full_ids.len() - 1].to_vec();
        let target_raw: Vec<u32> = full_ids[1..].to_vec();

        let prompt_len = prompt_ids.len();
        let mut weights: Vec<f32> = vec![0.0; input_raw.len()];
        for i in (prompt_len - 1)..weights.len() {
            let target_id = target_raw[i];
            if target_id == TOK_STEP {
                weights[i] = 0.1;
            } else {
                weights[i] = 1.0;
            }
        }

        let pad_input = tok.pad_or_truncate(&input_raw, seq_len);
        let pad_target = tok.pad_or_truncate(&target_raw, seq_len);

        let pad_offset = if input_raw.len() < seq_len {
            seq_len - input_raw.len()
        } else {
            0
        };
        let mut pad_weights = vec![0.0f32; seq_len];
        if input_raw.len() <= seq_len {
            for (i, &w) in weights.iter().enumerate() {
                if pad_offset + i < seq_len {
                    pad_weights[pad_offset + i] = w;
                }
            }
        } else {
            let skip = input_raw.len() - seq_len;
            pad_weights[..seq_len].copy_from_slice(&weights[skip..(seq_len + skip)]);
        }

        data.push((pad_input, pad_target, pad_weights));
    }

    data
}

// ---------------------------------------------------------------------------
// SFT Training Loop
// ---------------------------------------------------------------------------

/// Train Plan-LM with teacher forcing (SFT).
pub fn train_sft(
    config: &PlanLmConfig,
    device: &Device,
) -> Result<(WiredTransformer, VarMap, Vec<f32>)> {
    let tok = PlanTokenizer::new();
    let vocab_size = tok.vocab_size();

    let tf_config = TransformerConfig {
        d_model: config.d_model,
        n_layers: config.n_layers,
        n_heads: config.n_heads,
        d_ff: config.d_ff,
        vocab_size,
        max_seq_len: config.max_seq_len,
    };

    let varmap = VarMap::new();
    let model = WiredTransformer::new(tf_config, &varmap, device)?;

    let data = prepare_plan_training_data(&tok, config.max_seq_len);
    let n_goals = data.len();

    let batch_size = config.batch_size.min(n_goals);
    let n_batches = (n_goals + batch_size - 1) / batch_size;
    let mut batches: Vec<(Tensor, Tensor, Tensor)> = Vec::new();

    for b_idx in 0..n_batches {
        let start = b_idx * batch_size;
        let end = (start + batch_size).min(n_goals);
        let bs = end - start;

        let mut batch_input = Vec::with_capacity(bs * config.max_seq_len);
        let mut batch_target = Vec::with_capacity(bs * config.max_seq_len);
        let mut batch_weight = Vec::with_capacity(bs * config.max_seq_len);

        for i in start..end {
            batch_input.extend_from_slice(&data[i].0);
            batch_target.extend_from_slice(&data[i].1);
            batch_weight.extend_from_slice(&data[i].2);
        }

        batches.push((
            Tensor::from_vec(batch_input, (bs, config.max_seq_len), device)?,
            Tensor::from_vec(batch_target, (bs, config.max_seq_len), device)?,
            Tensor::from_vec(batch_weight, (bs, config.max_seq_len), device)?,
        ));
    }

    let scaled_lr = config.sft_lr * (batch_size as f64).sqrt();

    let training_config = TrainingConfig {
        lr: scaled_lr,
        min_lr: scaled_lr * 0.01,
        weight_decay: 0.01,
        warmup_fraction: 0.1,
        total_steps: config.sft_steps,
        grad_accum_steps: 1,
        max_grad_norm: 1.0,
        label_smoothing: 0.0,
    };

    let mut trainer = Trainer::new(varmap.clone(), training_config)?;
    eprintln!("[plan-lm] SFT: {} goals, batch={}, {} steps, lr={:.2e} (scaled), d={}, layers={}",
        n_goals, batch_size, config.sft_steps, scaled_lr,
        config.d_model, config.n_layers);

    let mut losses = Vec::new();

    let mut early_stop = EarlyStopping::new(
        config.early_stop_loss,
        config.early_stop_patience,
        if config.early_stop_loss > 0.0 { Some("planner_best.safetensors".to_string()) } else { None },
    );

    for step in 0..config.sft_steps {
        let (input, targets, weight_tensor) = &batches[step % n_batches];

        let logits = model.forward(input)?;
        let loss = weighted_cross_entropy(&logits, targets, 0.0, Some(weight_tensor))?;
        let loss_val = loss.to_scalar::<f32>()?;
        losses.push(loss_val);

        trainer.backward_step(&loss)?;

        if step % 500 == 0 || step == config.sft_steps - 1 {
            let window = losses.len().min(10);
            let avg_loss: f32 = losses.iter().rev().take(window).sum::<f32>() / window as f32;
            eprintln!("[plan-lm] step {step}/{} avg_loss={avg_loss:.4} lr={:.2e}",
                config.sft_steps, trainer.current_lr());

            // Early stopping + best checkpoint check
            if step > 0 {
                if let EarlyStopAction::Stop = early_stop.check(avg_loss, step, &varmap) {
                    eprintln!("[plan-lm] Early stopped at step {step}/{}", config.sft_steps);
                    break;
                }
            }
        }
    }

    trainer.print_timer("plan-lm-sft");

    let (ok, total) = score_plan_bench(&tok, &|goal| {
        let prompt_ids = tok.encode_prompt(goal);
        greedy_decode(&model, &tok, &prompt_ids, 128, device).unwrap_or_default()
    });
    eprintln!("[plan-lm] SFT result: {ok}/{total} plan_bench");

    Ok((model, varmap, losses))
}

// ---------------------------------------------------------------------------
// Diagnostic: Per-position probability tracing
// ---------------------------------------------------------------------------

/// Trace per-position probabilities during greedy decode.
pub fn diagnostic_decode(
    model: &WiredTransformer,
    tok: &PlanTokenizer,
    prompt_ids: &[u32],
    reference: &[String],
    max_tokens: usize,
    device: &Device,
) -> Result<Vec<DiagEntry>> {
    let seq_len = model.config.max_seq_len;
    let vocab_size = model.config.vocab_size;

    let mut context = tok.pad_or_truncate(prompt_ids, seq_len);
    let mut diag: Vec<DiagEntry> = Vec::new();
    let mut state = FsmState::Start;
    let mut steps_emitted = 0usize;

    for pos in 0..max_tokens {
        let input = Tensor::from_vec(context.clone(), (1, seq_len), device)?;
        let logits = model.forward(&input)?;
        let last_logits = logits.i((0, seq_len - 1, ..))?;

        let probs = candle_nn::ops::softmax(&last_logits.unsqueeze(0)?, 1)?;
        let probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;

        let valid = valid_tokens_for_state(state, tok, steps_emitted);
        let masked = apply_fsm_mask(&last_logits.unsqueeze(0)?, &valid, vocab_size, device)?;
        let token_id = masked.argmax(D::Minus1)?.squeeze(0)?.to_scalar::<u32>()?;
        let token_str = tok.token(token_id).to_string();

        let mut sorted_probs: Vec<(u32, f32)> = probs_vec.iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();
        sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<(String, f32)> = sorted_probs.iter()
            .take(5)
            .map(|(id, p)| (tok.token(*id).to_string(), *p))
            .collect();

        let ref_tok = reference.get(pos).map(|s| s.as_str()).unwrap_or("???");
        let ref_prob = if pos < reference.len() {
            let ref_id = tok.id(ref_tok);
            probs_vec.get(ref_id as usize).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        diag.push(DiagEntry {
            position: pos,
            generated: token_str.clone(),
            reference: ref_tok.to_string(),
            ref_probability: ref_prob,
            generated_probability: probs_vec.get(token_id as usize).copied().unwrap_or(0.0),
            top5,
            diverged: token_str != ref_tok,
        });

        if token_str == "EOP" { break; }

        state = fsm_transition(state, &token_str, &mut steps_emitted);
        context.remove(0);
        context.push(token_id);
    }

    Ok(diag)
}

#[derive(Debug)]
pub struct DiagEntry {
    pub position: usize,
    pub generated: String,
    pub reference: String,
    pub ref_probability: f32,
    pub generated_probability: f32,
    pub top5: Vec<(String, f32)>,
    pub diverged: bool,
}

impl DiagEntry {
    pub fn print(&self) {
        let marker = if self.diverged { "!!!" } else { "   " };
        eprintln!("{marker} pos={:>2} gen={:>15} ref={:>15} ref_prob={:.4} top5={:?}",
            self.position, self.generated, self.reference, self.ref_probability,
            self.top5.iter().map(|(t, p)| format!("{t}:{p:.3}")).collect::<Vec<_>>());
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsm_hello_plan() {
        let tok = PlanTokenizer::new();
        let mut state = FsmState::Start;
        let mut steps = 0;

        let valid = valid_tokens_for_state(state, &tok, steps);
        assert!(valid.contains(&tok.step_id()));
        assert!(valid.contains(&tok.eop_id()));

        state = fsm_transition(state, "STEP", &mut steps);
        assert_eq!(state, FsmState::AfterStep);
        assert_eq!(steps, 1);

        let valid = valid_tokens_for_state(state, &tok, steps);
        assert!(valid.contains(&tok.id("TALK")));
        assert!(valid.contains(&tok.id("RG")));

        state = fsm_transition(state, "TALK", &mut steps);
        assert_eq!(state, FsmState::AfterActionNoArgs);

        let valid = valid_tokens_for_state(state, &tok, steps);
        assert!(valid.contains(&tok.eop_id()));
    }

    #[test]
    fn test_fsm_rg_flow() {
        let tok = PlanTokenizer::new();
        let mut state = FsmState::Start;
        let mut steps = 0;

        state = fsm_transition(state, "STEP", &mut steps);
        state = fsm_transition(state, "RG", &mut steps);
        assert_eq!(state, FsmState::AfterRg);

        let valid = valid_tokens_for_state(state, &tok, steps);
        assert!(valid.contains(&tok.id("PAT0")));
        assert!(!valid.contains(&tok.id("TALK")));

        state = fsm_transition(state, "PAT0", &mut steps);
        assert_eq!(state, FsmState::AfterActionNoArgs);
    }

    #[test]
    fn test_fsm_memory_flow() {
        let tok = PlanTokenizer::new();
        let mut state = FsmState::Start;
        let mut steps = 0;

        state = fsm_transition(state, "STEP", &mut steps);
        state = fsm_transition(state, "MEMADD", &mut steps);
        assert_eq!(state, FsmState::AfterMemAdd);

        let valid = valid_tokens_for_state(state, &tok, steps);
        assert!(valid.contains(&tok.id("KINDFACT")));
        assert!(valid.contains(&tok.id("KINDPREFERENCE")));

        state = fsm_transition(state, "KINDPREFERENCE", &mut steps);
        assert_eq!(state, FsmState::AfterMemAddKind);

        let valid = valid_tokens_for_state(state, &tok, steps);
        assert!(valid.contains(&tok.id("TEXTLEN3")));
    }

    #[test]
    fn test_prepare_training_data() {
        let tok = PlanTokenizer::new();
        let data = prepare_plan_training_data(&tok, 128);
        assert_eq!(data.len(), 21, "should have 21 training sequences");

        for (i, (input, target, weights)) in data.iter().enumerate() {
            assert_eq!(input.len(), 128, "seq {i} input wrong len");
            assert_eq!(target.len(), 128, "seq {i} target wrong len");
            assert_eq!(weights.len(), 128, "seq {i} weights wrong len");
            let nonzero: usize = weights.iter().filter(|&&w| w > 0.0).count();
            assert!(nonzero > 0, "seq {i} has no non-zero weights");
        }
    }

    #[test]
    fn test_greedy_decode_untrained() -> Result<()> {
        let device = Device::Cpu;
        let tok = PlanTokenizer::new();
        let config = TransformerConfig {
            d_model: 64,
            n_layers: 1,
            n_heads: 2,
            d_ff: 128,
            vocab_size: tok.vocab_size(),
            max_seq_len: 64,
        };
        let varmap = VarMap::new();
        let model = WiredTransformer::new(config, &varmap, &device)?;

        let prompt_ids = tok.encode_prompt("hello");
        let generated = greedy_decode(&model, &tok, &prompt_ids, 32, &device)?;

        assert!(!generated.is_empty(), "should generate at least one token");
        Ok(())
    }

    #[test]
    fn test_train_sft_tiny() -> Result<()> {
        let device = Device::Cpu;
        let config = PlanLmConfig {
            d_model: 64,
            n_layers: 1,
            n_heads: 2,
            d_ff: 128,
            max_seq_len: 128,
            batch_size: 4,
            sft_steps: 100,
            sft_lr: 1e-3,
            scheduled_sampling_steps: 0,
            ss_lr: 5e-4,
            step_token_weight: 0.1,
            action_token_weight: 1.0,
            early_stop_loss: 0.0,
            early_stop_patience: 3,
        };

        let (_model, _varmap, losses) = train_sft(&config, &device)?;
        assert!(!losses.is_empty());

        let first_10_avg: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let last_10_avg: f32 = losses[losses.len()-10..].iter().sum::<f32>() / 10.0;
        assert!(last_10_avg < first_10_avg,
            "loss should decrease: {first_10_avg:.4} -> {last_10_avg:.4}");

        Ok(())
    }
}
