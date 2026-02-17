# Engineering Principles

These are load-bearing walls. Every contributor reads this first.

## 1. Status Quo Code

Code represents the current system, not its history. No internal version control:

```rust
// BAD:
// Old V3 approach (deprecated):
// fn old_thing() { ... }
// New V4 approach:
fn new_thing() { ... }

// GOOD:
fn the_thing() { ... }
```

No `_old`, `_legacy`, `_v2` suffixes. No commented-out code blocks. No "temporary" markers that become permanent. If code exists, it's the current implementation. If it's replaced, delete it.

**Anti-pattern: Compatibility shims**
```rust
// BAD: keeping old interface alive
fn encode_v2(&self, text: &str) -> Vec<u32> { self.encode(text) }
fn encode_legacy(&self, text: &str) -> Vec<u32> { self.encode(text) }

// GOOD: one function, one name
fn encode(&self, text: &str) -> Vec<u32> { ... }
```

## 2. Minimal Line Count

Less code is more maintainable. Measure twice, write once.

- Three similar lines > premature abstraction
- One function doing one thing > clever multi-purpose function
- Flat is better than nested (max 3 levels of indentation)
- If a function exceeds 50 lines, it's doing too much
- If a file exceeds 800 lines, it needs splitting

The goal is the minimum code that passes all tests and serves all requirements. Every line must justify its existence.

**Anti-pattern: Premature abstraction**
```rust
// BAD: abstraction for one call site
trait Poolable { fn pool(&self, x: &Tensor) -> Tensor; }
struct MeanPool;
impl Poolable for MeanPool { fn pool(&self, x: &Tensor) -> Tensor { x.mean(1) } }
struct Brain { pooler: Box<dyn Poolable> }

// GOOD: inline the operation
struct Brain { /* no pooler field */ }
impl Brain {
    fn encode(&self, x: &Tensor) -> Tensor { x.mean(1)? }
}
```

## 3. No Dead Code

`#![deny(dead_code, unused_imports, unused_variables)]` in every module.

Dead code is a liability:
- It rots (dependencies change, it stops compiling silently)
- It confuses (readers don't know if it's used elsewhere)
- It bloats (more code = more context = slower comprehension)

If you might need it later, git has your back. Delete it now.

**Anti-pattern: "Maybe we'll need this"**
```rust
// BAD: keeping unused config options
pub struct TransformerConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub future_moe_experts: usize,  // unused, "for Phase 3"
    pub reserved_field: Option<f32>, // unused, "might need"
}

// GOOD: only what's needed now
pub struct TransformerConfig {
    pub d_model: usize,
    pub n_layers: usize,
}
```

## 4. Tests Are Proof

"It works" means nothing without a passing test. Every public function has at least one test. Tests verify behavior, not implementation.

```rust
// BAD: Tests implementation details
assert_eq!(model.internal_state.len(), 42);

// GOOD: Tests observable behavior
let output = model.forward(&input)?;
assert_eq!(output.dims(), &[1, seq_len, vocab_size]);
```

Tests must be deterministic. Same seed = same result. Flaky tests are bugs.

**Anti-pattern: Mocked tests that verify nothing**
```rust
// BAD: test passes but verifies nothing meaningful
#[test]
fn test_brain() {
    let brain = Brain::new(config);
    assert!(brain.is_ok()); // only checks construction, not behavior
}

// GOOD: test verifies end-to-end behavior
#[test]
fn test_brain_encode_shape() {
    let brain = Brain::new(test_config())?;
    let vec = brain.encode("search for main")?;
    assert_eq!(vec.dims(), &[1, 64]); // (batch, d_model)
}
```

## 5. One Brain, Not Four Models

V4 had four separate models (Policy, Plan-LM, Talk, Regions) with no integration. V5 has ONE brain with multiple regions sharing concept space.

- All regions use the same d_model
- All regions share the same transformer backbone where possible
- Inference is one pipeline: `brain.run_goal(goal)` -> result
- No model needs external context about other models

**Anti-pattern: Independent models with separate encoders**
```rust
// BAD: V4 pattern — each model has its own transformer
let policy_encoder = WiredTransformer::new(policy_config);
let plan_encoder = WiredTransformer::new(plan_config);
let talk_encoder = WiredTransformer::new(talk_config);

// GOOD: V5 pattern — one shared backbone
let brain = Brain::new(config); // single encoder, multiple heads
let concept = brain.encode(goal);
let intent = brain.classify(&concept);
let plan = brain.plan(&concept);
let response = brain.generate(&concept);
```

## 6. Fail Loud, Fail Early

```rust
// BAD: Silent failure
let result = risky_operation().unwrap_or_default();

// GOOD: Explicit error propagation
let result = risky_operation().context("failed during X because Y")?;
```

Errors carry context. Panics are for invariant violations. `unwrap()` only on proven invariants with a comment explaining why.

**Anti-pattern: Swallowing errors in training loops**
```rust
// BAD: training silently produces garbage
let loss = cross_entropy(&logits, &targets).unwrap_or(Tensor::zeros(&[1]));

// GOOD: NaN/inf detection with abort
let loss = cross_entropy(&logits, &targets)?;
if loss.to_scalar::<f32>()?.is_nan() {
    anyhow::bail!("NaN loss at step {step}. Last good loss: {prev_loss}");
}
```

## 7. GPU by Default

Every matrix operation >= 4096 elements goes to GPU when available. CPU is the fallback, not the default. Benchmark GPU utilization on every training change.

Build command always includes CUDA:
```bash
PATH="/usr/local/cuda-12.6/bin:$PATH" CUDA_HOME="/usr/local/cuda-12.6" CUDA_COMPUTE_CAP=89 cargo build --release --features cuda
```

**Anti-pattern: CPU-only backward pass (M-015)**
```rust
// BAD: forward on GPU, backward on CPU
fn forward(&self, x: &Tensor) -> Tensor { x.matmul(&self.w)? } // GPU via candle
fn backward(&self, grad: &Tensor) -> Tensor {
    // manual CPU triple loop
    for i in 0..m { for j in 0..n { for k in 0..p { ... } } }
}

// GOOD: both paths use the same device
fn forward(&self, x: &Tensor) -> Tensor { x.matmul(&self.w)? }
// backward is automatic via candle autograd
```

## 8. Data Quality Over Quantity

For a 200M parameter model, every training example matters. Curate ruthlessly:
- One correct example > ten noisy examples
- Synthetic textbook data > raw web scrapes
- Domain-specific Rust > general programming
- Verified outputs only (compile-tested, proof-checked)

**Anti-pattern: Bulk scraping without filtering**
```rust
// BAD: "more data is always better"
let corpus: Vec<String> = scrape_all_github_rust_files();

// GOOD: curated with quality gate
let corpus: Vec<String> = scrape_rust_files()
    .filter(|f| f.has_tests())          // has tests = likely correct
    .filter(|f| f.compiles())           // verified compilable
    .filter(|f| f.lines() < 500)        // not auto-generated
    .collect();
```

## 9. Instrument Before Optimizing

When something doesn't work:
1. Add per-position probability tracing
2. Run ONE diagnostic
3. The diagnostic reveals the fix

Never sweep hyperparameters blindly. Never retry the same approach more than twice. Understand WHY before changing WHAT.

**Anti-pattern: Blind hyperparameter sweeping (M-010)**
```rust
// BAD: 30 experiments varying LR, d_model, layers
for lr in [1e-3, 5e-4, 1e-4, 5e-5] {
    for d in [32, 64, 128, 256] {
        train_and_pray(lr, d); // all produce 0/21
    }
}

// GOOD: one diagnostic run that reveals the root cause
let diag = diagnostic_decode(&model, &goals);
for (pos, entry) in diag.iter().enumerate() {
    eprintln!("pos {pos}: target={}, top1={} (p={:.3}), fsm_state={:?}",
        entry.target, entry.top1, entry.top1_prob, entry.fsm_state);
}
// Output: "STEP tokens at 94% prob, action tokens at 0.7%"
// Fix: position-weighted loss. 3 lines of code.
```

## 10. Ship Incrementally

Every phase is independently useful. No phase depends on future phases being complete. After each phase:
- All existing tests still pass
- New capability is end-to-end verified
- The system is strictly better than before

**Anti-pattern: "We'll test it when Phase 3 is done"**
```
// BAD: Phase 0-2 all untested, hoping Phase 3 validates everything
Phase 0: port transformer (no tests)
Phase 1: port planner (no tests)
Phase 2: port brain (no tests)
Phase 3: write tests for everything (everything is broken)

// GOOD: each phase is independently verified
Phase 0: port transformer + gradient_check + shape tests -> ALL GREEN
Phase 1: port planner + FSM tests + plan_bench -> 21/21
Phase 2: port brain + policy bench + generation test -> 16/16
```

---

## Anti-Patterns Quick Reference

| Pattern | Looks Like | Actually Is | Rule |
|---------|-----------|-------------|------|
| Compatibility shim | `fn old_name() { new_name() }` | Dead code with extra steps | Principle 1 |
| Premature abstraction | `trait + impl for one type` | Indirection that helps nobody | Principle 2 |
| "Might need later" | `reserved_field: Option<T>` | Dead code that rots | Principle 3 |
| Mocked verification | `assert!(x.is_ok())` | Tests construction, not behavior | Principle 4 |
| Independent encoders | 4 transformers, 4 configs | Duplicate params, no sharing | Principle 5 |
| Swallowed errors | `.unwrap_or_default()` | Hidden bugs that surface later | Principle 6 |
| CPU backward pass | GPU forward, CPU backward | 95% GPU idle time | Principle 7 |
| Bulk data ingestion | "scrape everything" | Noise drowns signal | Principle 8 |
| Hyperparameter sweep | 30 experiments, 0 diagnostics | Wrong level of analysis | Principle 9 |
| Big-bang testing | "test after Phase 3" | Everything broken at once | Principle 10 |

---

## Code Review Checklist

Before approving any change, verify these in order:

### Correctness
- [ ] Does it compile with `#![deny(dead_code, unused_imports, unused_variables)]`?
- [ ] Are all public functions tested?
- [ ] Do tests verify behavior, not implementation?
- [ ] Is `gradient_check()` run after any architecture change?
- [ ] Are tensor shapes documented and tested?

### Architecture
- [ ] Does it use the shared Brain encoder, not a separate one?
- [ ] Is d_model consistent across all regions?
- [ ] Are new parameters added to optimizer AND gradient clipping?
- [ ] Is the decoder right-padded? (M-003)
- [ ] Is memory prefix included in training? (M-004)

### Training
- [ ] Is seq_len >= max prompt + plan length? (M-008)
- [ ] Are loss weights balanced across token classes? (M-002)
- [ ] Does training steps scale with data size? (M-011)
- [ ] Is loss convergence verified (not just "it decreases")?
- [ ] Is per-prompt diversity monitored, not just aggregate loss? (M-007)

### GPU
- [ ] Are all matmul paths using GPU when available? (M-015)
- [ ] Is GPU util checked in the first 60s of training? (M-015)
- [ ] Is PTX ASCII-only? (M-005)
- [ ] Is `cargo build --release` clean (no stale artifacts)? (M-016)

### Process
- [ ] Is the change the minimum needed? (Principle 2)
- [ ] Are there no "future use" fields or commented-out blocks? (Principle 3)
- [ ] Is every error propagated with context? (Principle 6)
- [ ] Are meta tools (eval/train) blocked from executing inside training loops or ReAct? (M-017)
- [ ] Does this change break existing tests?

---

## Decision Framework

### When to abstract vs. duplicate

**Duplicate** when:
- Only 2-3 call sites exist
- The "common" code is < 5 lines
- The abstraction would need parameters for each call site's differences
- You're not sure the pattern is stable yet

**Abstract** when:
- 4+ call sites with identical logic
- The common code is > 10 lines
- Bug in one copy would need fixing in all copies
- The pattern has been stable across 2+ phases

### When to optimize vs. ship

**Ship first** when:
- The feature doesn't exist yet (correctness > speed)
- GPU util > 50% (good enough for now)
- Training time < 5 minutes for test mode
- The optimization would change the architecture

**Optimize first** when:
- GPU util < 20% (something is fundamentally wrong — M-015)
- Training time blocks iteration (> 30 min for test mode)
- The bottleneck is clear from profiling (not guessing)
- The optimization is a constant change (buffer pool, weight cache), not architectural

### When to add a diagnostic vs. try another approach

**Add a diagnostic** when:
- 2 attempts at the same approach have failed (M-010)
- You're varying hyperparameters without understanding the failure
- Loss decreases but bench accuracy is 0 (teacher-forcing illusion — M-014)
- Multiple goals produce identical output (mode collapse)

**Try another approach** when:
- The diagnostic revealed a clear root cause
- The fix is surgical (< 10 lines)
- The new approach addresses the diagnosed cause, not a guess
