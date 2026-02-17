# GESTALT WIRED-V5 Project Instructions

## Hardware Utilization (L10-EXTINCTION)
- **80% total component utilization minimum.** RTX 5070 Ti (16GB VRAM) is the primary compute.
- GPU training in `--release` mode. NEVER debug-mode CPU training as a substitute.
- Report metrics as total system %, not per-core. `ps` shows per-core — divide by total cores.
- If GPU is at 0% during any compute task, STOP and ask why.

## candle-nn v0.8.4 Known Bugs
- `candle_nn::RmsNorm` — **BROKEN backward.** Use `GradRmsNorm` (transformer.rs).
- `candle_nn::ops::softmax_last_dim` — **BROKEN backward.** Use `grad_softmax_last_dim` (transformer.rs).
- Before first training with ANY new architecture, run `test_candle_op_gradient_flow` to verify all Vars get gradients.

## Training Discipline
- After 2 failed experiments with same metric, STOP and build a diagnostic test. (M-010, M-031)
- If a metric is identical across fundamentally different approaches, the problem is structural, not training. (M-031)
- Verify gradient flow BEFORE training, not after 12 failed runs. (M-032)

## Communication
- ALL results go to Discord via `discord_send_message`. NO embeds.
- Heartbeat V3: automated while-loop curl to sidecar. See global CLAUDE.md.
- Never claim compaction without verifying context was actually lost. (M-034)
- Never present guesses as measurements. Measure or say "I don't know." (M-036)
