# Current Tasks -- GESTALT

## Active Sprint: Phase 2 (BPE + Language Scaling)

**Goal:** Break the 373-token ceiling. Real vocabulary, scaled model, language training.

### In Progress
- **T-016: v15 GPU training — SFT-only (DA disabled)**
  - v14 COMPLETE: all 4 phases ran, policy 64/64 perfect
  - **DA post-mortem (M-039):** DA at lr=1e-4 destroyed autoregressive coherence.
    SFT model was perfect, DA corrupted first-byte accuracy and long-range generation.
  - v15 retraining with da_steps=0 (SFT-only), early stopping + best checkpoint active
  - PAPER.md at 2,364 lines, sent to Discord
  - Early stopping implemented: saves ~87% wasted planner steps, ~30% policy steps
  - `gestalt gallery` command: 53 prompts across 9 categories

### What Made v14 Work (v2-v13 all failed)
1. **candle-nn gradient fix (M-032)**: `RmsNorm` and `softmax_last_dim` had broken backward
   passes. Replaced with `GradRmsNorm` and `grad_softmax_last_dim` using basic tensor ops.
   ALL 12 prior runs failed because the encoder NEVER received a single gradient.
2. **Mean pooling (replacing last-token extraction)**: Last-token extraction produced
   identical concept vectors (sim=0.96) because all inputs end with EOS. Mean pooling
   over non-PAD positions produces discriminative vectors (sim=0.25).
3. **PAD-denoising (10% noise)**: Forces decoder to attend to concept prefix instead of
   relying only on local bigram statistics. Without this, loss plateaus at ~2.15.

### Up Next (After v15 completes)
- Run `gestalt gallery --config default` — comprehensive generation samples
- Send gallery results to Discord
- Fix DA: either full-sequence DA at lr=1e-5, or skip entirely
- T-017: Language region training on expanded corpus
- T-018: DPO alignment (optional, evaluate after T-017)
- Phase 2 full run: `--config phase2` (d=1024, 8+8 layers, ~200M params)

### Ahead-of-Schedule Completions
- T-015: Corpus expanded to 242 dialogue pairs (was 80 in V4) ✅
- T-019: memory.rs fully implemented (339 LOC, 8 tests) ✅
- ConceptTokenizer foundation laid in tokenizer.rs (11 tests) ✅
- PAPER.md: 2,340-line technical paper with full v14 narrative ✅

---

## v14 Training Results (COMPLETE — 2h 1m wall clock)

### SFT Phase (COMPLETE — 11,378s, 2.2 steps/sec)
```
Config: d_model=512, enc=1L, dec=4L, heads=8, batch=48, lr=3e-4, noise=10%
Step     0: loss=6.37 | concept_sim=N/A
Step  6250: loss=0.02 | concept_sim=0.42 | Greedy: coherent JARVIS responses
Step 12500: loss=0.00 | concept_sim=0.28 | Greedy: same quality, encoder more discriminative
Step 25000: loss=0.00 | concept_sim=0.25 | Generation perfect on all test prompts
GPU: 84-98% SM, 13.6/16.3GB VRAM
```

### DA Phase (COMPLETE but HARMFUL — M-039)
```
Config: batch=16, lr=1e-4, 8192 steps, 1360s
Step  1000: loss=0.25
Step  4600: loss=0.003
Step  8191: loss=0.0015 (final)
```
**FINDING:** DA destroyed generation quality. SFT model was perfect; after DA,
"hello" → "Rello" (first byte wrong). DA trains on isolated byte positions
which breaks autoregressive coherence. Default config now uses da_steps=0.

### Planner SFT (COMPLETE — 593s, 6.7 steps/sec)
```
21/21 plan_bench (perfect). Loss hit 0.0000 at step 500/4000 — 87% wasted.
```

### Policy (COMPLETE — 16384 steps)
```
64/64 bench (perfect). Loss 0.0001 by step 10000 — ~30% wasted.
```

### Concept Tokenizer (COMPLETE)
```
2,259 tokens = 259 base + 2,000 merges. Saved to concept_tokenizer.bin (27KB).
"hello" → 3 tokens (1.67x), "what can you do" → 4 tokens (3.75x)
```

### v2-v13 Failure Summary (for posterity)
All 12 runs produced concept_sim=0.9591 (collapsed encoder). Root cause: candle-nn
RmsNorm/softmax broken backward passes → zero gradient to all transformer parameters
except lm_head. Diagnosed in 2 hours with isolation test after 40+ hours of blind
iteration (M-031, M-032).

---

## Completed Phases

### Done (Phase 1 — Tool Execution + Pipeline)
- T-010: executor.rs — 15-tool execution engine (472 LOC, 9 tests) ✅
- T-011: pipeline.rs — run_goal() orchestration (434 LOC, 8 tests) ✅
- T-012: integration.rs — 6 E2E pipeline tests (147 LOC) ✅
- T-013: main.rs — CLI binary: run/train/eval/serve/gallery ✅

### Done (Phase 0 — Foundation Port)
- T-001: transformer.rs (623 LOC, 8 tests) ✅
- T-002: training.rs (535 LOC, 8 tests) ✅
- T-003: tokenizer.rs (1,043 LOC, 22 tests — includes ConceptTokenizer) ✅
- T-004: planner.rs (685→710 LOC, 6 tests) ✅
- T-005: brain.rs (1,937→2,100+ LOC, 20 tests — unified brain) ✅
- T-006: eval.rs (497 LOC, 10 tests) ✅
- T-007: Talk corpus embedded in brain.rs (242 pairs) ✅
- T-008: lib.rs module root (deny dead_code, all modules) ✅
- T-009: Full build verification ✅

---

## Build Commands
```bash
# CPU check
cargo check --release

# Full GPU build
PATH="/usr/local/cuda-12.6/bin:$PATH" CUDA_HOME="/usr/local/cuda-12.6" CUDA_COMPUTE_CAP=89 cargo build --release --features cuda

# Test (105 tests)
cargo test --release

# Train (config tiers)
./target/release/gestalt train --config default   # d=512, GPU
./target/release/gestalt train --config phase2     # d=1024, GPU

# Gallery (after training, loads checkpoint)
./target/release/gestalt gallery --config default

# Run/Serve with trained checkpoint
./target/release/gestalt run "hello" --config default
./target/release/gestalt serve --config default
```

## Key Decisions Resolved
- ConceptEncoder: **mean pooling** over non-PAD positions (v14+, was last-token in v2-v13)
- Gradient safety: `GradRmsNorm` + `grad_softmax_last_dim` replace broken candle-nn ops
- Denoising: PAD-replacement at 10% max with quadratic ramp
- Direct encoder training (v14): no codebook bypass, gradients flow through encoder
- Policy encoding: BYTE_VOCAB=256 (raw bytes), NOT TalkTokenizer
- Memory default: K=8 top memories, capacity=1024 entries, FIFO eviction
- Config tiers: test (d=64, CPU) / default (d=512, GPU) / phase2 (d=1024, GPU)
- Checkpoints: safetensors format, auto-save after training phases

## Cumulative Stats
- Total: ~7,200+ LOC across 13 files, 105 tests passing
- Warnings: 0, Dead code: 0
- Phase 0+1: COMPLETE
- Phase 2: IN PROGRESS — v14 training active, first successful run
- Phase 3: T-019 done ahead of schedule
