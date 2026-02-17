# GESTALT -- WIRED-BRAIN V5

> "Language is just an abstraction of concepts" -- Lain
> "It must be based on memory too, not just concepts" -- Lain

One brain. Multiple regions. No external LLM. From scratch.

## What

A from-scratch AI brain in Rust that thinks in concepts, reasons over them, stores memories, executes tools, and uses language as an interface. Built on candle-rs with CUDA acceleration.

## Architecture

```
            ReasoningCore (shared MoE trunk)
                    |
    +-------+-------+-------+-------+
    |       |       |       |       |
 Language Action  Memory Planning  Session
 Region  Region  Region  Region   State
```

All regions share the same d_model concept space. Language maps text to/from concepts. Reasoning happens in concept space.

## Inherited from V4 (verified, green)

| Component | Metric | Status |
|-----------|--------|--------|
| WiredTransformer | 3.3e-2 grad error | Port directly |
| Plan-LM FSM | 21/21 plan bench | Port decoder + FSM |
| Brain Policy | 16/16 test bench | Port classification heads |
| Brain Regions | E2E gradient flow | Port + fix memory |
| Training infra | AdamW + cosine LR | Port directly |

## New in V5

- Tool execution engine (brain predicts AND runs)
- BPE tokenizer (8K vocab, replaces 373-token ceiling)
- Persistent memory (SQLite-backed episodic store)
- Multi-turn sessions (ring buffer context)
- ReAct loop (Reason -> Act -> Observe -> Reason)
- Online learning (micro-training from interactions)
- Unified binary: `gestalt serve`

## Build

```bash
# CPU
cargo build --release

# GPU (RTX 5070 Ti)
PATH="/usr/local/cuda-12.6/bin:$PATH" \
CUDA_HOME="/usr/local/cuda-12.6" \
CUDA_COMPUTE_CAP=89 \
cargo build --release --features cuda
```

## Test

```bash
cargo test --release
```

## Hardware

RTX 5070 Ti (16GB VRAM), WSL2, 24GB RAM.
