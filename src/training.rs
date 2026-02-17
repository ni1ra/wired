use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use std::process::Command;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Cosine LR Scheduler with Linear Warmup
// ---------------------------------------------------------------------------

pub struct CosineScheduler {
    base_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl CosineScheduler {
    pub fn new(base_lr: f64, min_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    pub fn step(&mut self) -> f64 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    pub fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            self.base_lr * (self.current_step as f64 + 1.0) / self.warmup_steps as f64
        } else {
            let progress = (self.current_step - self.warmup_steps) as f64
                / (self.total_steps - self.warmup_steps).max(1) as f64;
            let progress = progress.min(1.0);
            self.min_lr
                + 0.5
                    * (self.base_lr - self.min_lr)
                    * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }

    pub fn current_step(&self) -> usize {
        self.current_step
    }
}

// ---------------------------------------------------------------------------
// Training Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub lr: f64,
    pub min_lr: f64,
    pub weight_decay: f64,
    pub warmup_fraction: f64,
    pub total_steps: usize,
    pub grad_accum_steps: usize,
    pub max_grad_norm: f64,
    pub label_smoothing: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            min_lr: 1e-5,
            weight_decay: 0.01,
            warmup_fraction: 0.1,
            total_steps: 1000,
            grad_accum_steps: 1,
            max_grad_norm: 1.0,
            label_smoothing: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

pub struct Trainer {
    pub optimizer: AdamW,
    pub scheduler: CosineScheduler,
    pub config: TrainingConfig,
    pub varmap: VarMap,
    accum_count: usize,
    step_count: usize,
    timer_start: Instant,
}

impl Trainer {
    pub fn new(varmap: VarMap, config: TrainingConfig) -> Result<Self> {
        let warmup_steps = (config.total_steps as f64 * config.warmup_fraction) as usize;
        let scheduler =
            CosineScheduler::new(config.lr, config.min_lr, warmup_steps, config.total_steps);

        let params = ParamsAdamW {
            lr: config.lr,
            weight_decay: config.weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        let optimizer = AdamW::new(varmap.all_vars(), params)?;

        Ok(Self {
            optimizer,
            scheduler,
            config,
            varmap,
            accum_count: 0,
            step_count: 0,
            timer_start: Instant::now(),
        })
    }

    /// Accumulate a loss. Returns Some(step_number) when optimizer actually steps.
    pub fn accumulate_and_step(&mut self, loss: &Tensor) -> Result<Option<usize>> {
        let scale = 1.0 / self.config.grad_accum_steps as f64;
        let scaled_loss = (loss * scale)?;

        self.optimizer.backward_step(&scaled_loss)?;
        self.accum_count += 1;

        if self.accum_count >= self.config.grad_accum_steps {
            self.accum_count = 0;
            self.step_count += 1;
            let new_lr = self.scheduler.step();
            self.optimizer.set_learning_rate(new_lr);
            return Ok(Some(self.step_count));
        }
        Ok(None)
    }

    /// Simple backward step (no accumulation).
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<usize> {
        self.optimizer.backward_step(loss)?;
        self.step_count += 1;
        let new_lr = self.scheduler.step();
        self.optimizer.set_learning_rate(new_lr);
        Ok(self.step_count)
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    pub fn current_lr(&self) -> f64 {
        self.scheduler.get_lr()
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.timer_start.elapsed().as_secs_f64()
    }

    pub fn print_timer(&self, label: &str) {
        let elapsed = self.elapsed_secs();
        eprintln!(
            "[TIMER] {label}: {elapsed:.1}s ({} steps, {:.1} steps/sec)",
            self.step_count,
            self.step_count as f64 / elapsed.max(0.001)
        );
    }
}

// ---------------------------------------------------------------------------
// Early Stopping + Best Checkpoint Tracking
// ---------------------------------------------------------------------------

/// Tracks convergence and saves the best checkpoint automatically.
/// Usage: call `check()` at each log interval. Returns `EarlyStopAction`.
pub struct EarlyStopping {
    threshold: f32,
    patience: usize,
    below_count: usize,
    stale_count: usize,
    stale_stop: bool,
    best_loss: f32,
    best_step: usize,
    best_path: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum EarlyStopAction {
    Continue,
    NewBest,
    Stop,
}

impl EarlyStopping {
    /// Create with threshold and patience. `best_path` is the file to save best weights to.
    /// threshold=0.0 disables early stopping (but best checkpoint still tracked).
    pub fn new(threshold: f32, patience: usize, best_path: Option<String>) -> Self {
        Self {
            threshold,
            patience,
            below_count: 0,
            stale_count: 0,
            stale_stop: false,
            best_loss: f32::MAX,
            best_step: 0,
            best_path,
        }
    }

    /// Disabled early stopping (no-op that never stops, no best checkpoint).
    pub fn disabled() -> Self {
        Self::new(0.0, usize::MAX, None)
    }

    /// Enable patience-based stopping on no improvement (stale loss).
    /// Stops when best loss hasn't improved for `patience` consecutive checks.
    pub fn with_stale_stop(mut self) -> Self {
        self.stale_stop = true;
        self
    }

    /// Check current average loss. Returns action to take.
    /// Saves best checkpoint automatically if `best_path` is set.
    pub fn check(&mut self, avg_loss: f32, step: usize, varmap: &VarMap) -> EarlyStopAction {
        let mut action = EarlyStopAction::Continue;

        // Track best loss and save checkpoint
        if avg_loss < self.best_loss {
            self.best_loss = avg_loss;
            self.best_step = step;
            if let Some(ref path) = self.best_path {
                if let Err(e) = save_checkpoint(varmap, path) {
                    eprintln!("[BEST] Warning: failed to save best checkpoint: {e}");
                } else {
                    eprintln!("[BEST] New best loss={avg_loss:.6} at step {step} -> {path}");
                }
            }
            action = EarlyStopAction::NewBest;
        }

        // Check early stopping: two modes
        if self.threshold > 0.0 {
            // Threshold mode: stop when loss stays below threshold for `patience` checks
            if avg_loss < self.threshold {
                self.below_count += 1;
                if self.below_count >= self.patience {
                    eprintln!("[EARLY STOP] Loss {avg_loss:.6} < threshold {:.6} for {} consecutive checks. \
                               Stopping at step {step} (best was {:.6} at step {}).",
                        self.threshold, self.patience, self.best_loss, self.best_step);
                    return EarlyStopAction::Stop;
                }
            } else {
                self.below_count = 0;
            }
        }

        // Stale-stop mode: stop when no improvement for `patience` consecutive checks
        if self.stale_stop {
            if action == EarlyStopAction::NewBest {
                self.stale_count = 0;
            } else {
                self.stale_count += 1;
                if self.stale_count >= self.patience {
                    eprintln!("[EARLY STOP] No improvement for {} consecutive checks. \
                               Stopping at step {step} (best was {:.6} at step {}).",
                        self.patience, self.best_loss, self.best_step);
                    return EarlyStopAction::Stop;
                }
            }
        }

        action
    }

    pub fn best_loss(&self) -> f32 { self.best_loss }
    pub fn best_step(&self) -> usize { self.best_step }
}

// ---------------------------------------------------------------------------
// Cross-Entropy with Label Smoothing + Per-Position Weights
// ---------------------------------------------------------------------------

/// Cross-entropy loss with optional label smoothing and per-position weights.
/// logits: (batch, seq, vocab), targets: (batch, seq), weights: optional (batch, seq)
pub fn weighted_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    label_smoothing: f64,
    weights: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, s, v) = logits.dims3()?;
    let logits_flat = logits.reshape((b * s, v))?;
    let targets_flat = targets.reshape(b * s)?;

    if label_smoothing > 0.0 {
        let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)?;
        let targets_u32 = targets_flat.to_dtype(DType::U32)?;
        let one_hot = one_hot_tensor(&targets_u32, v, logits.device())?;
        let smooth = ((one_hot * (1.0 - label_smoothing))? + (label_smoothing / v as f64))?;
        let per_pos = (smooth * log_probs)?.sum(1)?.neg()?;

        if let Some(w) = weights {
            let w_flat = w.reshape(b * s)?;
            let weighted = (per_pos * w_flat.clone())?;
            let denom = w_flat.sum_all()?;
            Ok((weighted.sum_all()? / denom)?)
        } else {
            per_pos.mean_all().map_err(Into::into)
        }
    } else if let Some(w) = weights {
        let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)?;
        let targets_u32 = targets_flat.to_dtype(DType::U32)?;
        let one_hot = one_hot_tensor(&targets_u32, v, logits.device())?;
        let per_pos = (one_hot * log_probs)?.sum(1)?.neg()?;
        let w_flat = w.reshape(b * s)?;
        let weighted = (per_pos * w_flat.clone())?;
        let denom = w_flat.sum_all()?;
        Ok((weighted.sum_all()? / denom)?)
    } else {
        candle_nn::loss::cross_entropy(&logits_flat, &targets_flat).map_err(Into::into)
    }
}

pub fn one_hot_tensor(indices: &Tensor, num_classes: usize, device: &Device) -> Result<Tensor> {
    let n = indices.elem_count();
    let indices_vec: Vec<u32> = indices.to_vec1()?;
    let mut data = vec![0.0f32; n * num_classes];
    for (i, &idx) in indices_vec.iter().enumerate() {
        let idx = idx as usize;
        if idx < num_classes {
            data[i * num_classes + idx] = 1.0;
        }
    }
    Tensor::from_vec(data, (n, num_classes), device).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Model Checkpointing (safetensors)
// ---------------------------------------------------------------------------

pub fn save_checkpoint(varmap: &VarMap, path: &str) -> Result<()> {
    let tensors = varmap.all_vars();
    let data = varmap.data().lock().unwrap();
    let named: std::collections::HashMap<String, Tensor> = data
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect();
    candle_core::safetensors::save(&named, path)?;
    eprintln!("[CHECKPOINT] Saved {} params to {path}", tensors.len());
    Ok(())
}

pub fn load_checkpoint(varmap: &VarMap, path: &str, device: &Device) -> Result<()> {
    let tensors = candle_core::safetensors::load(path, device)?;
    let data = varmap.data().lock().unwrap();
    let mut loaded = 0usize;
    for (name, var) in data.iter() {
        if let Some(saved_tensor) = tensors.get(name) {
            var.set(saved_tensor)?;
            loaded += 1;
        }
    }
    eprintln!(
        "[CHECKPOINT] Loaded {loaded}/{} params from {path}",
        data.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Timing Report
// ---------------------------------------------------------------------------

pub struct TimingReport {
    pub phases: Vec<(String, f64)>,
    start: Instant,
}

impl TimingReport {
    pub fn new() -> Self {
        Self {
            phases: Vec::new(),
            start: Instant::now(),
        }
    }

    pub fn mark(&mut self, label: &str) {
        let elapsed = self.start.elapsed().as_secs_f64();
        eprintln!("[TIMER] {label}: {elapsed:.1}s");
        self.phases.push((label.to_string(), elapsed));
        self.start = Instant::now();
    }

    pub fn print_report(&self) {
        let total: f64 = self.phases.iter().map(|(_, t)| t).sum();
        eprintln!("\n[TIMING REPORT]");
        for (label, secs) in &self.phases {
            let pct = if total > 0.0 {
                secs / total * 100.0
            } else {
                0.0
            };
            eprintln!("  {label}: {secs:.1}s ({pct:.0}%)");
        }
        eprintln!("  TOTAL: {total:.1}s");
    }
}

impl Default for TimingReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GPU Stats (via nvidia-smi)
// ---------------------------------------------------------------------------

pub struct GpuStats {
    pub sm_util_pct: f32,
    pub mem_used_mb: u64,
    pub mem_total_mb: u64,
    pub power_w: f32,
    pub temp_c: u32,
}

impl GpuStats {
    pub fn query() -> Option<Self> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let s = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = s.trim().split(',').map(|p| p.trim()).collect();
        if parts.len() < 5 {
            return None;
        }
        Some(GpuStats {
            sm_util_pct: parts[0].parse().unwrap_or(0.0),
            mem_used_mb: parts[1].parse().unwrap_or(0),
            mem_total_mb: parts[2].parse().unwrap_or(0),
            power_w: parts[3].parse().unwrap_or(0.0),
            temp_c: parts[4].parse().unwrap_or(0),
        })
    }

    pub fn log(&self, label: &str) {
        let mem_pct = if self.mem_total_mb > 0 {
            self.mem_used_mb as f32 / self.mem_total_mb as f32 * 100.0
        } else {
            0.0
        };
        eprintln!(
            "[GPU] {label}: SM={:.0}% VRAM={}/{}MB ({:.1}%) Power={:.0}W Temp={}C",
            self.sm_util_pct, self.mem_used_mb, self.mem_total_mb, mem_pct, self.power_w,
            self.temp_c
        );
    }
}

/// Sample GPU stats periodically during a closure. Returns (result, avg_sm_util, peak_vram_mb).
pub fn with_gpu_monitoring<F, R>(sample_interval_ms: u64, f: F) -> (R, f32, u64)
where
    F: FnOnce() -> R,
{
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use std::thread;

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    let monitor = thread::spawn(move || {
        let mut sm_sum = 0.0f64;
        let mut vram_peak = 0u64;
        let mut count = 0u64;
        while running_clone.load(Ordering::Relaxed) {
            if let Some(stats) = GpuStats::query() {
                sm_sum += stats.sm_util_pct as f64;
                vram_peak = vram_peak.max(stats.mem_used_mb);
                count += 1;
            }
            thread::sleep(std::time::Duration::from_millis(sample_interval_ms));
        }
        (
            if count > 0 {
                (sm_sum / count as f64) as f32
            } else {
                0.0
            },
            vram_peak,
        )
    });

    let result = f();
    running.store(false, Ordering::Relaxed);

    let (avg_sm, peak_vram) = monitor.join().unwrap_or((0.0, 0));
    (result, avg_sm, peak_vram)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_scheduler_warmup() {
        let mut sched = CosineScheduler::new(1e-3, 1e-5, 10, 100);
        let lr0 = sched.step();
        let lr5 = {
            for _ in 0..4 {
                sched.step();
            }
            sched.step()
        };
        assert!(lr5 > lr0, "LR should increase during warmup: {lr0} -> {lr5}");
    }

    #[test]
    fn test_cosine_scheduler_decay() {
        let mut sched = CosineScheduler::new(1e-3, 1e-5, 10, 100);
        for _ in 0..10 {
            sched.step();
        }
        let lr_after_warmup = sched.get_lr();
        for _ in 0..80 {
            sched.step();
        }
        let lr_near_end = sched.get_lr();
        assert!(
            lr_near_end < lr_after_warmup,
            "LR should decay: {lr_after_warmup} -> {lr_near_end}"
        );
        assert!(
            lr_near_end >= 1e-5,
            "LR should not go below min: {lr_near_end}"
        );
    }

    #[test]
    fn test_cosine_scheduler_bounds() {
        let mut sched = CosineScheduler::new(1e-3, 1e-5, 10, 100);
        for _ in 0..200 {
            let lr = sched.step();
            assert!(lr >= 1e-5 - 1e-10, "LR below min: {lr}");
            assert!(lr <= 1e-3 + 1e-10, "LR above max: {lr}");
        }
    }

    #[test]
    fn test_weighted_cross_entropy_basic() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::randn(0.0f32, 1.0, (1, 4, 8), &device)?;
        let targets = Tensor::new(vec![0u32, 1, 2, 3], &device)?.unsqueeze(0)?;

        let loss = weighted_cross_entropy(&logits, &targets, 0.0, None)?;
        let val = loss.to_scalar::<f32>()?;
        assert!(
            val.is_finite() && val > 0.0,
            "loss should be finite positive: {val}"
        );
        Ok(())
    }

    #[test]
    fn test_weighted_cross_entropy_with_smoothing() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::randn(0.0f32, 1.0, (1, 4, 8), &device)?;
        let targets = Tensor::new(vec![0u32, 1, 2, 3], &device)?.unsqueeze(0)?;

        let loss_no_smooth =
            weighted_cross_entropy(&logits, &targets, 0.0, None)?.to_scalar::<f32>()?;
        let loss_smooth =
            weighted_cross_entropy(&logits, &targets, 0.1, None)?.to_scalar::<f32>()?;

        assert!(
            (loss_no_smooth - loss_smooth).abs() > 1e-6,
            "smoothing should change loss: {loss_no_smooth} vs {loss_smooth}"
        );
        Ok(())
    }

    #[test]
    fn test_weighted_cross_entropy_with_weights() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::randn(0.0f32, 1.0, (1, 4, 8), &device)?;
        let targets = Tensor::new(vec![0u32, 1, 2, 3], &device)?.unsqueeze(0)?;

        let weights = Tensor::new(vec![0.0f32, 1.0, 1.0, 1.0], &device)?.unsqueeze(0)?;
        let loss_weighted = weighted_cross_entropy(&logits, &targets, 0.0, Some(&weights))?;
        let val = loss_weighted.to_scalar::<f32>()?;
        assert!(
            val.is_finite() && val > 0.0,
            "weighted loss should be finite positive: {val}"
        );
        Ok(())
    }

    #[test]
    fn test_timing_report() {
        let mut report = TimingReport::new();
        report.mark("phase1");
        report.mark("phase2");
        assert_eq!(report.phases.len(), 2);
        report.print_report();
    }

    #[test]
    fn test_trainer_creation() -> Result<()> {
        let varmap = VarMap::new();
        let _var = varmap.get(
            (4, 4),
            "test_weight",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
            DType::F32,
            &Device::Cpu,
        )?;
        let config = TrainingConfig::default();
        let trainer = Trainer::new(varmap, config)?;
        assert_eq!(trainer.step_count(), 0);
        assert!(trainer.current_lr() > 0.0);
        Ok(())
    }

    #[test]
    fn test_early_stopping_triggers() -> Result<()> {
        let varmap = VarMap::new();
        let _var = varmap.get(
            (4, 4), "w",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 },
            DType::F32, &Device::Cpu,
        )?;

        let mut es = EarlyStopping::new(0.01, 3, None);

        // Above threshold — should continue
        assert_eq!(es.check(0.5, 1, &varmap), EarlyStopAction::NewBest);
        assert_eq!(es.check(0.3, 2, &varmap), EarlyStopAction::NewBest);

        // Below threshold, but patience not met yet
        assert_eq!(es.check(0.005, 3, &varmap), EarlyStopAction::NewBest);
        assert_eq!(es.check(0.004, 4, &varmap), EarlyStopAction::NewBest);

        // Third consecutive below-threshold = STOP
        assert_eq!(es.check(0.003, 5, &varmap), EarlyStopAction::Stop);
        assert!(es.best_loss() < 0.01);
        Ok(())
    }

    #[test]
    fn test_early_stopping_reset_on_spike() -> Result<()> {
        let varmap = VarMap::new();
        let _var = varmap.get(
            (4, 4), "w",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 },
            DType::F32, &Device::Cpu,
        )?;

        let mut es = EarlyStopping::new(0.01, 3, None);

        es.check(0.005, 1, &varmap); // below
        es.check(0.004, 2, &varmap); // below
        es.check(0.05, 3, &varmap);  // spike! resets counter

        // Counter reset — need 3 more consecutive below
        let action = es.check(0.003, 4, &varmap);
        assert_ne!(action, EarlyStopAction::Stop);
        Ok(())
    }

    #[test]
    fn test_early_stopping_disabled() -> Result<()> {
        let varmap = VarMap::new();
        let _var = varmap.get(
            (4, 4), "w",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 },
            DType::F32, &Device::Cpu,
        )?;

        let mut es = EarlyStopping::new(0.0, 3, None); // threshold=0 = disabled
        for i in 0..100 {
            assert_ne!(es.check(0.0001, i, &varmap), EarlyStopAction::Stop);
        }
        Ok(())
    }

    #[test]
    fn test_early_stopping_stale_stop() -> Result<()> {
        let varmap = VarMap::new();
        let _var = varmap.get(
            (4, 4), "w",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.01 },
            DType::F32, &Device::Cpu,
        )?;

        let mut es = EarlyStopping::new(0.0, 3, None).with_stale_stop();

        // First check is always NewBest (loss < f32::MAX)
        assert_eq!(es.check(1.0, 1, &varmap), EarlyStopAction::NewBest);

        // Improving = NewBest, resets stale counter
        assert_eq!(es.check(0.9, 2, &varmap), EarlyStopAction::NewBest);

        // Stale (worse): count 1, 2
        assert_eq!(es.check(1.0, 3, &varmap), EarlyStopAction::Continue);
        assert_eq!(es.check(1.1, 4, &varmap), EarlyStopAction::Continue);

        // Third stale = Stop
        assert_eq!(es.check(1.2, 5, &varmap), EarlyStopAction::Stop);
        assert_eq!(es.best_step(), 2);
        Ok(())
    }
}
