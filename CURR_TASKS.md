# Current Tasks -- GESTALT

## Active Sprint: Phase 2.5 Evaluation + Documentation

**Goal:** Evaluate v22 hero run, update documentation, push to GitHub.

### Just Completed
- **v22 hero gallery evaluation** — 166 prompts, ~40-45% coherent openings
- **Sampling improvements** — windowed repetition penalty (50 tokens), top-p (0.9)
- **PAPER.md rewrite** — 2,364→941 lines, restructured for accessibility
- **127 tests passing** (119 lib + 8 integration)

### In Progress
- Documentation updates (this file, ALL_TASKS.md)
- Git commit + push to `phase-2.5/concept-tokenizer-memory`

### Up Next
- **Scale to d=1024** — 4x more capacity, 8+8 layers, ~200M parameters
- **Expand corpus** — target 100K+ pairs from public datasets
- **Memory-augmented training** — train decoder WITH memory prefix (T-020 architecture ready)

---

## v22 Hero Results (merges=200, dropout=0.1, 30K steps)

### What Works (~40-45% coherent openings)
```
"good morning"       → "Morning! How'd you sleep?"                    ← perfect
"The build broke"    → "Step one: read the error message."            ← excellent
"tell me about yourself" → "I'm a language model with opinions..."    ← personality captured
"NGE?"               → "A mecha show about depression, parental..."   ← remarkable
"I'm scared of failing" → "Good. Fear of failure means you care..."   ← emotionally intelligent
```

### What Breaks
After ~15 tokens, outputs degenerate into word salad. The model interpolates between training examples mid-sentence. Capacity bottleneck: 512-dim, 21K pairs.

### Training Run History

| Run | Corpus | Vocab | Dropout | Steps | Best Val | Gallery |
|-----|--------|-------|---------|-------|----------|---------|
| v2-v12 | 242 | 259 | 0.0 | varies | N/A | Encoder dead (M-032) |
| v14 | 242 | 259 | 0.0 | 25K | ~0.00 | Perfect memorization |
| v18 | 1,749 | 259 | 0.0 | 12K | 2.02 | Memorized verbatim |
| v19 | 21,786 | 2,259 | 0.0 | 6K | 3.55 | 0% coherent |
| v20 | 21,786 | 2,259 | 0.1 | 8K | 3.26 | ~30% coherent |
| v22 mock | 21,786 | 459 | 0.1 | 5K | 2.04 | Grammar good |
| **v22 hero** | **21,786** | **459** | **0.1** | **30K** | **~1.9** | **~40-45% coherent** |

### Key Discovery: Merge Count
```
Grid search at 300 steps:
  merges=50  → val=2.79    merges=200  → val=3.56 ← sweet spot
  merges=500 → val=4.08    merges=2000 → val=5.53
```

---

## Phase Progress

```
Phase 0: Foundation Port        ████████████████████ 100%
Phase 1: Tool Execution         ████████████████████ 100%
Phase 2: BPE + Language         ██████████████░░░░░░  70%  (v22 trained + evaluated)
Phase 2.5: Memory Integration   ████████████████████ 100%  (code complete)
Phase 3: Memory-Augmented Train ██░░░░░░░░░░░░░░░░░░  10%  (architecture ready)
Phase 4: Multi-Turn + ReAct     ████░░░░░░░░░░░░░░░░  20%  (session ring buffer done)
Phase 5: Online Learning        ░░░░░░░░░░░░░░░░░░░░   0%
Phase 6: Proactive + JARVIS     ░░░░░░░░░░░░░░░░░░░░   0%
```

---

## Completed Phases

### Phase 2.5: Memory Integration (CODE COMPLETE)
- ConceptTokenizer with BPE merges (459 vocab, 200 merges) ✅
- brain.rs decoder_vocab_size replaces hardcoded TALK_VOCAB_SIZE ✅
- Memory-augmented forward pass (build_prefix with optional memory) ✅
- EpisodicMemory with retrieve_recent() for bulk loading ✅
- Cross-session recall integration tests ✅
- 21,786-pair corpus from Dolly/OASST2/Alpaca ✅
- Grid search infrastructure (env vars for merges, dropout, steps) ✅

### Phase 1: Tool Execution + Pipeline
- T-010: executor.rs — 15 tools, 3 safety levels ✅
- T-011: pipeline.rs — run_goal() orchestration ✅
- T-012: integration.rs — 8 E2E tests ✅
- T-013: main.rs — CLI: train/gallery/serve/run ✅

### Phase 0: Foundation Port
- T-001 through T-009: all complete ✅

---

## Build Commands
```bash
# GPU build
PATH="/usr/local/cuda-12.6/bin:$PATH" CUDA_HOME="/usr/local/cuda-12.6" \
  CUDA_COMPUTE_CAP=89 cargo build --release --features cuda

# Tests (127 total)
cargo test --release --features cuda

# Gallery evaluation
GESTALT_MERGES=200 ./target/release/gestalt gallery --config default

# Training with env vars
GESTALT_MERGES=200 GESTALT_DROPOUT=0.1 GESTALT_SFT_STEPS=30000 \
  ./target/release/gestalt train --config default

# Grid search (quick 300-step sweeps)
./scripts/grid_search.sh
```

## Key Architecture Decisions
- ConceptEncoder: mean pooling over non-PAD (v14+)
- Gradient safety: GradRmsNorm + grad_softmax_last_dim (M-032)
- Denoising: PAD-replacement at 10% max, quadratic ramp
- Tokenizer: merges=200 (459 vocab, ~2x compression)
- Dropout: 0.1 (extends training from 6K to 13K+ before overfitting)
- Memory: K=8, capacity=1024, FIFO eviction, SQLite persistence
- Early stopping: stale_stop mode (no threshold, pure patience)

## Cumulative Stats
- Total: ~7,400 LOC across 13 files, 127 tests passing
- Warnings: 0, Dead code: 0
- Training runs: v2-v22 (14 total, 12 failed due to M-032)
- Documented mistakes: 43 in MISTAKES.md
