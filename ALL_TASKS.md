# ALL_TASKS -- GESTALT WIRED-V5

> Every task has: ID, description, acceptance criteria, source files.
> Status: [ ] pending, [~] in progress, [x] done.

---

## Dependency Graph

```
Phase 0 (Foundation Port):
  T-001 (transformer) ─┬─> T-004 (planner)  ─┐
  T-002 (training)     ─┤                      ├─> T-008 (lib.rs) -> T-009 (build verify)
  T-003 (tokenizer)    ─┤                      │
  T-005 (brain)        ─┼─> T-006 (eval)      ─┤
  T-007 (talk corpus)  ─┘                      │
                                                │
Phase 1 (Tool Execution):                      │
  T-010 (executor) ────┬─> T-011 (pipeline)    │
  T-005 (brain)  ──────┘        │              │
  T-004 (planner) ─────────────>│              │
                          T-012 (integration)  │
                          T-013 (binary)       │
                                                │
Phase 2 (BPE + Language):                      │
  T-014 (BPE tokenizer) -> T-015 (corpus) -> T-016 (scale) -> T-017 (train) -> T-018 (DPO)
                                                │
Phase 3 (Memory):                              │
  T-019 (memory store) -> T-020 (mem training) -> T-021 (recall test)
                                                │
Phase 4 (Multi-Turn + ReAct):                  │
  T-022 (session) -> T-023 (react) -> T-024 (concept CoT) -> T-025 (multi-turn test)
                                                │
Phase 5 (Online Learning):                     │
  T-026 (experience buf) -> T-027 (micro-train) -> T-028 (consolidation)
                                                │
Phase 6 (Proactive):                           │
  T-029 (server) -> T-030 (context monitor) -> T-031 (dynamic tools) -> T-032 (personality)
```king 

### Parallel Work Opportunities

| Group | Tasks | Why Parallel |
|-------|-------|--------------|
| Foundation trio | T-001, T-002, T-003 | Independent V4 ports, no shared state |
| Brain + Planner | T-004, T-005 | Both depend on transformer but not on each other |
| Eval + Talk | T-006, T-007 | Independent ports, can merge into brain later |
| Executor + Pipeline | T-010, T-011 (partial) | Executor interface can be stubbed for pipeline work |
| BPE + Corpus | T-014, T-015 (partial) | Tokenizer design can start while corpus is collected |

---

## Phase 0: Foundation Port (V4 -> V5)

Port proven V4 components into GESTALT's unified structure. No legacy code. No dead weight. Every ported line must compile and pass tests in the new crate.

### Source: `WIRED-BRAIN-V3/crates/wired_v4/src/`

### 0.1 Transformer Backbone
- [ ] **T-001** Port `transformer.rs` (563 LOC) -> `src/transformer.rs`
  - **V4 source:** `transformer.rs:1-563`
  - **Port:** WiredTransformer, TransformerConfig, RoPE (precompute_rope, apply_rope), Attention (q/k/v/o_proj), Mlp (gate+down GELU), TransformerBlock (pre-norm residual)
  - **Port:** `forward()`, `encode()`, `forward_with_prefix()`
  - **Port:** `cross_entropy_loss()`, `gradient_check()`, `run_benchmark()`
  - **Drop:** `TransformerConfig::spike()` (V4 benchmarking artifact, not needed)
  - **Keep:** `TransformerConfig::tiny()` for gradient checks
  - **LOC estimate:** ~530 (minimal reduction — transformer is clean)
  - **Risk:** LOW. Direct port, well-tested in V4.
  - **Tests:**
    - `test_forward_shapes` — verify output dims match (B, S, vocab)
    - `test_encode_shapes` — verify concept_vec is (B, d_model)
    - `test_forward_with_prefix` — verify prefix tokens prepended correctly
    - `test_gradient_check` — numerical vs autograd, threshold < 5e-1
    - `test_rope_positions` — verify RoPE frequencies match analytical expectation
  - **Accept:** gradient_check < 5e-1 relative error. forward/encode shape tests pass.

### 0.2 Training Infrastructure
- [ ] **T-002** Port `training.rs` (479 LOC) -> `src/training.rs`
  - **V4 source:** `training.rs:1-479`
  - **Port:** Trainer (AdamW optimizer + scheduler + timing), TrainingConfig, CosineScheduler (linear warmup + cosine decay), TimingReport, GpuStats
  - **Port:** `weighted_cross_entropy()` (with label smoothing + per-position weights), `with_gpu_monitoring()` (threaded nvidia-smi polling)
  - **Port:** `save_checkpoint()`/`load_checkpoint()` (safetensors), `one_hot_tensor()`
  - **LOC estimate:** ~470
  - **Risk:** LOW. Utility code, no architecture changes.
  - **Tests:**
    - `test_cosine_scheduler_warmup` — LR ramps from 0 to max during warmup
    - `test_cosine_scheduler_decay` — LR follows cosine curve after warmup
    - `test_weighted_ce_basic` — loss correct on known input/target pair
    - `test_weighted_ce_positions` — zero-weighted positions contribute 0 loss
    - `test_checkpoint_roundtrip` — save + load produces identical tensors
  - **Accept:** CosineScheduler warmup + decay curve matches V4. Loss decreases on dummy data.

### 0.3 Plan Tokenizer (temporary, replaced in Phase 2)
- [ ] **T-003** Port `tokenizer.rs` (444 LOC) -> `src/tokenizer.rs`
  - **V4 source:** `tokenizer.rs:1-444`
  - **Port:** PlanTokenizer with 373-token vocab, all constants (TOK_PAD=0 through TOK_MEMSEARCH=22), range tokens (PAT, FILE, PICK, FROM, CHAR), encode/decode with character fallback, plan_prompt(), pad_or_truncate (left-pad)
  - This is a TEMPORARY port. Phase 2 replaces with BPE. Mark clearly with `// TEMPORARY: replaced by BPE in Phase 2`.
  - **LOC estimate:** ~430
  - **Risk:** LOW. Direct port, 11 tests in V4.
  - **Tests:**
    - `test_vocab_size` — `vocab_size() == 373`
    - `test_encode_decode_roundtrip` — encode then decode returns original
    - `test_special_tokens` — PAD, BOS, EOS, STEP, EOP at expected indices
    - `test_pad_or_truncate` — left-padding behavior correct
    - `test_plan_prompt` — goal text formatted correctly
  - **Accept:** `vocab_size() == 373`. All roundtrip tests pass.

### 0.4 FSM Planner
- [ ] **T-004** Port `plan_lm.rs` (718 LOC) -> `src/planner.rs`
  - **V4 source:** `plan_lm.rs:1-718`
  - **Port:** PlanLmConfig (test + default_plan presets), FsmState enum (17 states), TextReturn enum
  - **Port:** `valid_tokens_for_state()`, `fsm_transition()`, `apply_fsm_mask()`, `greedy_decode()`, `diagnostic_decode()`
  - **Port:** `prepare_plan_training_data()` (per-position weights: STEP=0.1, action=1.0), `train_sft()` (batched, LR scaling)
  - **Depends on:** T-001 (transformer), T-003 (tokenizer)
  - **LOC estimate:** ~690
  - **Risk:** MEDIUM. FSM state machine is complex. Must verify all 17 state transitions.
  - **Tests:**
    - `test_fsm_valid_tokens` — each state produces correct valid token set
    - `test_fsm_transitions` — all 17 states transition correctly
    - `test_greedy_decode_valid` — output always ends with EOP, all tokens valid per FSM
    - `test_position_weights` — STEP positions get weight 0.1, action positions get 1.0
    - `test_diagnostic_decode` — per-position probability trace is complete
    - `test_train_sft_loss_decreases` — loss goes down on reference plans
  - **Accept:** FSM state transitions match V4. Greedy decode produces valid token sequences.

### 0.5 Brain (unified from brain_regions + brain_policy)
- [ ] **T-005** Port + merge `brain_regions.rs` (973 LOC) + `brain_policy.rs` (641 LOC) -> `src/brain.rs`
  - **V4 source:** `brain_regions.rs:1-973` + `brain_policy.rs:1-641`
  - **From brain_regions:** BrainConfig, ConceptEncoder, ConceptProjector, LanguageDecoder, MemoryBank (FIFO, cosine similarity retrieval), encode_concept(), project_concepts(), build_prefix(), forward(), right_pad(), prepare_brain_data(), train_brain_talk() (Phase 1 SFT + Phase 2 DA), brain_generate() (repetition penalty + top-k), brain_diagnostic_decode()
  - **From brain_policy:** PolicyConfig, PolicyOutput, 16 core + 48 expansion curriculum, encode_goal (byte-level), policy_loss (weighted per-head), train_and_bench(), brain_bench()
  - **Merge:** Single `Brain` struct combining both. Policy heads become fields of Brain, not a separate model.
  - **Drop:** Separate `BrainPolicy` struct. Standalone `BrainRegions` struct. Duplicate encoder initialization.
  - **LOC estimate:** ~1100 (consolidated from 1614 V4 LOC — 32% reduction)
  - **Risk:** HIGH. Largest merge. Two architectures becoming one. Careful interface design needed.
  - **Risk flag:** Memory prefix training (M-004 fix) must be correct from day one.
  - **Tests:**
    - `test_encode_concept_shape` — concept_vec is (B, d_model)
    - `test_project_concepts_shape` — prefix is (B, N, d_model)
    - `test_policy_heads_shapes` — all 5 heads produce correct output dims
    - `test_memory_store_retrieve` — store 10, retrieve top-3 by similarity
    - `test_brain_generate` — produces non-empty byte sequence without panic
    - `test_right_padding` — BOS at position 0, PAD fills end
    - `test_policy_loss_weights` — intent head weighted 1.0, others correctly weighted
    - `test_train_brain_loss_decreases` — SFT loss decreases over 100 steps
  - **Accept:** concept encoding produces (B, d_model). Policy heads produce correct shapes. Memory store/retrieve works.

### 0.6 Eval Harness
- [ ] **T-006** Port `eval_adapter.rs` (507 LOC) -> `src/eval.rs`
  - **V4 source:** `eval_adapter.rs:1-507`
  - **Port:** plan_bench_goals (21 goals), PlanIntent enum (14 intents), reference_plan_tokens() (with CoT prefix for multi-step), PlanStep enum, steps_for_intent(), parse_composite_steps(), score_plan_bench() (with CoT stripping)
  - **Depends on:** T-005 (brain), T-004 (planner)
  - **LOC estimate:** ~490
  - **Risk:** LOW. Direct port, scoring logic is well-defined.
  - **Tests:**
    - `test_21_goals_load` — plan_bench_goals returns exactly 21 goals
    - `test_reference_plans_valid` — all 21 reference plans are FSM-valid
    - `test_oracle_21_of_21` — perfect oracle scores 21/21
    - `test_cot_stripping` — THINK/ENDTHINK prefix removed before comparison
    - `test_composite_parsing` — "X and then Y" parsed into correct steps
  - **Accept:** 21 goals load. Reference plans generate valid token sequences.

### 0.7 Talk Corpus
- [ ] **T-007** Port dialogue corpus from `talk.rs` (788 LOC) -> embed in `src/brain.rs`
  - **V4 source:** `talk.rs:1-788`
  - **Port:** build_corpus() -> Vec<(String, String)> (~80 JARVIS dialogue pairs)
  - **Port:** TalkTokenizer (byte-level, vocab=259: 256 bytes + PAD/BOS/EOS), sample_with_temperature()
  - **Port:** Generation logic into Brain.generate()
  - **Drop:** Standalone Talk model. TalkConfig. Separate talk training pipeline.
  - Language generation is now a Brain Region, not a separate model.
  - **LOC estimate:** ~250 (embedded in brain.rs, stripped of standalone infrastructure)
  - **Risk:** LOW. Corpus is data, generation is straightforward.
  - **Tests:**
    - `test_corpus_loads` — build_corpus() returns >= 70 pairs
    - `test_corpus_no_empty` — no empty prompts or responses
    - `test_byte_tokenize_roundtrip` — encode/decode bytes matches
  - **Accept:** Corpus loads. Generation produces bytes without panic.

### 0.8 Unified Module Root
- [ ] **T-008** Write `src/lib.rs` with all module declarations
  - `#![deny(dead_code, unused_imports, unused_variables)]`
  - Public re-exports for key types: Brain, Pipeline, Executor, etc.
  - **Depends on:** T-001 through T-007
  - **LOC estimate:** ~30
  - **Risk:** LOW. Boilerplate.
  - **Tests:** None (compile check only)
  - **Accept:** `cargo check` clean. Zero warnings.

### 0.9 Build Verification
- [ ] **T-009** Full build + test on CPU and GPU
  - `cargo test --release`
  - `cargo test --release --features cuda`
  - **Depends on:** T-008 (all modules declared)
  - **LOC estimate:** 0 (no new code)
  - **Risk:** MEDIUM. CUDA build may need env var adjustment.
  - **Tests:** All tests from T-001 through T-007, plus:
    - `test_gpu_forward` — transformer forward on CUDA device
    - `test_gpu_gradient_check` — gradient_check on CUDA matches CPU
  - **Accept:** All tests pass. Zero dead code warnings. clippy clean.

**Phase 0 total: ~3,990 LOC ported from V4, target ~2,800 LOC after consolidation (30% reduction from brain merge).**

---

## Phase 1: Tool Execution Engine

Brain predicts AND executes. Close the loop.

### 1.1 Executor
- [ ] **T-010** Write `src/executor.rs` (~300 LOC)
  - `trait Tool { fn name(&self) -> &str; fn execute(&self, args: &ToolArgs) -> Result<ToolOutput>; }`
  - 15 tool implementations: cargo_test, cargo_check, rg, repo_read, repo_list, docs_lint, prove_algebra, lean_suite, patch_dry_run, wired_eval, wired_train, memory_add, memory_search, fix_tests, talk
  - ToolArgs: enum with per-tool argument structs
  - ToolOutput: stdout + stderr + exit_code + structured data
  - Sandbox: tools run in subprocess with timeout (30s default)
  - Safety levels: ReadOnly (rg, repo_read) vs Mutating (patch, fix_tests) vs Meta (wired_eval)
  - **Depends on:** None (trait-based, can be written first)
  - **LOC estimate:** ~300
  - **Risk:** MEDIUM. Subprocess management + timeout handling.
  - **Risk flag:** Meta tools (wired_eval, wired_train) must NEVER execute inside training loops (V3 M: training recursion).
  - **Tests:**
    - `test_rg_execution` — rg("fn main") finds main.rs
    - `test_cargo_test_output` — parses pass/fail count from cargo test
    - `test_timeout_enforcement` — long-running tool killed after timeout
    - `test_safety_levels` — mutating tools refuse without --allow-writes
  - **Accept:** Each tool executes and returns structured output. cargo_test returns pass/fail count.

### 1.2 Pipeline
- [ ] **T-011** Write `src/pipeline.rs` (~400 LOC)
  - `fn run_goal(brain: &Brain, goal: &str) -> Result<ExecutionResult>`
  - Steps: encode goal -> policy forward -> get intent + actions -> plan decode -> compile plan to steps -> execute steps -> collect results
  - Step result feedback: output of step N becomes context for step N+1
  - ExecutionResult: Vec<StepResult> + final_output + success flag
  - **Depends on:** T-005 (brain), T-004 (planner), T-010 (executor)
  - **LOC estimate:** ~400
  - **Risk:** MEDIUM. First end-to-end integration of all components.
  - **Tests:**
    - `test_run_goal_hello` — conversational intent, no tool execution
    - `test_run_goal_cargo_test` — classifies RunTests, executes cargo test
    - `test_run_goal_composite` — multi-step with step chaining
    - `test_run_goal_failure` — tool failure stops pipeline, returns partial results
  - **Accept:** `run_goal("cargo test")` classifies intent, generates plan, runs cargo test, returns results.

### 1.3 End-to-End Integration Test
- [ ] **T-012** Write `tests/integration.rs`
  - Test: "search jarviscmd and then open it" -> Composite -> plan -> rg -> repo_read -> file content
  - Test: "hello" -> Hello -> TALK plan -> greeting
  - Test: "cargo test" -> RunTests -> cargo test -> results
  - **Depends on:** T-011 (pipeline)
  - **LOC estimate:** ~150
  - **Risk:** HIGH. First time all components run together. Likely to expose integration bugs.
  - **Tests:** The file IS the tests.
  - **Accept:** 3/3 integration tests pass end-to-end.

### 1.4 Unified Binary
- [ ] **T-013** Write `src/main.rs` (~200 LOC)
  - `gestalt serve` -- persistent process, stdin/stdout interface
  - `gestalt train` -- full training pipeline
  - `gestalt eval` -- run eval harness
  - `gestalt run <goal>` -- single goal execution
  - **Depends on:** T-011 (pipeline), T-006 (eval)
  - **LOC estimate:** ~200
  - **Risk:** LOW. CLI wiring.
  - **Tests:**
    - `test_cli_run_hello` — `gestalt run "hello"` exits 0 with output
    - `test_cli_eval` — `gestalt eval` reports plan_bench scores
  - **Accept:** `gestalt run "hello"` produces output. `gestalt eval` reports scores.

---

## Phase 2: BPE Tokenizer + Language Scaling

Break the 373-token ceiling. Real vocabulary for real text.

### 2.1 BPE Tokenizer
- [ ] **T-014** Write `src/tokenizer.rs` (replace V4 port) (~500 LOC)
  - Use `tokenizers` crate (HuggingFace Rust) for BPE training
  - Target: 8,192 tokens
  - Training corpus: Rust stdlib docs + popular crate docs + project code
  - Special tokens: PAD, BOS, EOS, UNK, STEP, EOP + all action tokens (preserve semantic meaning)
  - Must handle both plan tokenization (structured) and text tokenization (natural language)
  - **Depends on:** T-003 (understands current token scheme)
  - **LOC estimate:** ~500
  - **Risk:** HIGH. Tokenizer change affects every downstream component. Embedding layer resizes. All training data needs re-tokenization.
  - **Risk flag:** Action tokens (RG, REPOREAD, etc.) must keep stable IDs across tokenizer versions.
  - **Tests:**
    - `test_bpe_roundtrip` — encode/decode matches input for Rust code
    - `test_special_tokens_preserved` — all action tokens in vocab at stable positions
    - `test_compression_ratio` — "fn main() { println!(\"hello\"); }" < 15 tokens
    - `test_rust_keywords` — common keywords are single tokens
  - **Accept:** Encodes `fn main() { println!("hello"); }` into < 15 tokens. Roundtrip decode matches input.

### 2.2 Corpus Pipeline
- [ ] **T-015** Write `src/corpus.rs` (~300 LOC)
  - Download + process training data from HuggingFace (`hf-hub` crate)
  - Sources: Rust documentation, programming Q&A, technical dialogue
  - Dedup, clean, format into training sequences
  - Target: 1-2B tokens curated
  - **Depends on:** T-014 (BPE tokenizer)
  - **LOC estimate:** ~300
  - **Risk:** MEDIUM. Data quality is critical (Principle 8). Web scrapes need heavy filtering.
  - **Tests:**
    - `test_corpus_pipeline` — produces tokenized output from sample input
    - `test_dedup` — duplicate sequences removed
    - `test_token_distribution` — no single token > 5% of corpus
  - **Accept:** Pipeline produces tokenized sequences on disk. Token distribution is reasonable.

### 2.3 Scale Transformer
- [ ] **T-016** Update `src/transformer.rs` configs
  - New config: d=1024, 8 layers, 8 heads, d_ff=4096, vocab=8192
  - ~200M parameters
  - Fits in ~800MB VRAM at fp32
  - **Depends on:** T-014 (vocab size change)
  - **LOC estimate:** ~50 (config changes only)
  - **Risk:** LOW. Same architecture, bigger numbers.
  - **Tests:**
    - `test_scaled_forward` — forward pass produces (B, S, 8192) output
    - `test_scaled_gradient_check` — gradients correct at d=1024
  - **Accept:** Model builds. Forward pass produces correct shapes. Gradient check passes.

### 2.4 Language Region Training
- [ ] **T-017** Train language model on curated corpus
  - SFT on technical text + dialogue
  - Cosine LR with warmup
  - Checkpoint every 10K steps
  - GPU monitoring throughout
  - **Depends on:** T-015 (corpus), T-016 (scaled model)
  - **LOC estimate:** ~200 (training script additions)
  - **Risk:** HIGH. First large-scale training run. May need curriculum tuning.
  - **Risk flag:** At 200M params, training will take hours-days on a single 5070 Ti.
  - **Tests:**
    - `test_language_generation` — generates multi-sentence coherent text
    - `test_loss_convergence` — loss decreases monotonically over 1000 steps
  - **Accept:** Loss converges. Model generates coherent multi-sentence technical text.

### 2.5 DPO Alignment (optional, evaluate)
- [ ] **T-018** Implement Direct Preference Optimization
  - Preference pairs: idiomatic Rust vs non-idiomatic, correct vs incorrect
  - Complements GRPO (already proven in V3)
  - **Depends on:** T-017 (trained base model)
  - **LOC estimate:** ~200
  - **Risk:** MEDIUM. DPO is well-understood but requires quality preference data.
  - **Tests:**
    - `test_dpo_loss` — DPO loss computable on sample pairs
    - `test_preference_alignment` — preferred output ranks higher after training
  - **Accept:** Measurable preference alignment on held-out pairs.

---

## Phase 3: Memory Integration

The brain remembers. Episodic memory that persists across sessions.

### 3.1 Persistent Memory Store
- [ ] **T-019** Write `src/memory.rs` (~400 LOC)
  - SQLite-backed: `(id, timestamp, concept_vec BLOB, goal_text, response_text, success BOOL)`
  - Store: insert new episodic memory
  - Retrieve: top-K by cosine similarity on concept_vec
  - Capacity management: FIFO eviction past N entries (default 1024)
  - Cross-session persistence (survives process restart)
  - **Depends on:** T-005 (brain produces concept_vecs)
  - **LOC estimate:** ~400
  - **Risk:** LOW. SQLite is well-understood. cosine similarity is straightforward.
  - **Tests:**
    - `test_store_and_retrieve` — store 10 memories, retrieve top-5 by similarity
    - `test_persistence` — close DB, reopen, retrieve same results
    - `test_fifo_eviction` — store 1025 entries, oldest is evicted
    - `test_empty_retrieval` — empty DB returns empty results (no panic)
  - **Accept:** Store 100 memories, retrieve top-5 by similarity, restart process, retrieve same top-5.

### 3.2 Memory-Augmented Training
- [ ] **T-020** Fix brain_regions decoder training
  - Train decoder WITH memory prefix from epoch 0 (V4 bug M-004 fix)
  - During SFT: randomly sample 0-K prior dialogues as "memory"
  - Project through memory_projector -> prefix tokens
  - Decoder learns to attend to memory naturally
  - **Depends on:** T-019 (memory store), T-005 (brain)
  - **LOC estimate:** ~150 (modifications to brain.rs training)
  - **Risk:** HIGH. This is the V4 bug fix. Must be correct from the start.
  - **Risk flag:** If memory prefix corrupts generation, the entire memory system is useless.
  - **Tests:**
    - `test_generation_with_memory` — decoder produces coherent text with memory prefix
    - `test_generation_without_memory` — quality doesn't degrade when memory is empty
    - `test_memory_attention` — decoder attention weights include memory positions
  - **Accept:** Decoder generates coherent text with memory prefix present. Quality does not degrade.

### 3.3 Cross-Session Recall Test
- [ ] **T-021** End-to-end memory test
  - Store "favorite color is blue" in session 1
  - Kill process, restart
  - Ask "what's my favorite color" in session 2
  - Brain retrieves memory, responds correctly
  - **Depends on:** T-020 (memory-augmented decoder)
  - **LOC estimate:** ~80 (test code only)
  - **Risk:** HIGH. Full stack test. Many failure modes.
  - **Tests:**
    - `test_cross_session_recall` — the test IS the acceptance criterion
  - **Accept:** Correct recall without retraining.

---

## Phase 4: Multi-Turn Context + ReAct Loop

Conversations and step-by-step reasoning.

### 4.1 Session State
- [ ] **T-022** Write `src/session.rs` (~400 LOC)
  - Ring buffer: Vec<Turn> with max 32 turns
  - Turn: { prompt, concept_vec, response, tool_results, timestamp }
  - Context window: [system; turn_1_concept; ...; current_goal]
  - Oldest turn evicted when buffer full
  - **Depends on:** T-005 (brain produces concept_vecs)
  - **LOC estimate:** ~400
  - **Risk:** LOW. Data structure work.
  - **Tests:**
    - `test_ring_buffer_capacity` — 32 turns stored, accessible
    - `test_eviction_at_33` — turn 1 evicted when 33rd added
    - `test_context_window_format` — correct serialization for decoder input
  - **Accept:** 32 turns stored. Turn 1 accessible from turn 32. Eviction works at 33.

### 4.2 ReAct Loop
- [ ] **T-023** Write ReAct loop in `src/session.rs` or `src/pipeline.rs` (~500 LOC)
  - Loop: Reason -> Act -> Observe -> Reason
  - Max iterations: 10 (prevent infinite loops)
  - Brain decides when to stop (generates DONE action)
  - Each iteration: concept_vec refined with observation
  - **Depends on:** T-022 (session), T-011 (pipeline)
  - **LOC estimate:** ~500
  - **Risk:** HIGH. Autonomous agent loops are notoriously hard to stabilize (V3 M: training recursion, premature action).
  - **Risk flag:** Must prevent meta-tool recursion (eval/train inside ReAct loop).
  - **Tests:**
    - `test_react_simple` — single-tool goal completes in 1 iteration
    - `test_react_multi_step` — "fix failing tests" triggers cargo_test -> read error -> propose fix (3+ iterations)
    - `test_react_max_iterations` — infinite loop capped at 10
    - `test_react_no_meta_tools` — wired_eval/wired_train blocked inside ReAct
  - **Accept:** "the tests are failing, fix them" triggers: cargo_test -> read error -> propose fix. 3+ tool calls in one goal.

### 4.3 Concept-Space Chain of Thought
- [ ] **T-024** Multiple forward passes through ReasoningCore before decoding
  - N iterations in concept space (N=2-4, learned or fixed)
  - Each pass refines concept vector via self-attention over concept history
  - Language decoder fires only at the end
  - **Depends on:** T-005 (brain encoder)
  - **LOC estimate:** ~200
  - **Risk:** HIGH. Novel architecture component. No V4 precedent.
  - **Risk flag:** Multiple passes may collapse to identity function if not properly trained.
  - **Tests:**
    - `test_multi_pass_changes_concept` — pass 2 concept_vec != pass 1 concept_vec
    - `test_multi_pass_quality` — multi-pass produces measurably better policy output
  - **Accept:** Multi-pass produces measurably different (and better) concept vectors than single-pass.

### 4.4 Multi-Turn Integration Test
- [ ] **T-025** 5-turn conversation test
  - Turn 1: "my project is a web server"
  - Turn 3: "run the tests" (should know context is web server)
  - Turn 5: references turn 1 context
  - **Depends on:** T-022 (session), T-023 (react)
  - **LOC estimate:** ~100 (test code only)
  - **Risk:** HIGH. Full stack integration. Context retention across turns.
  - **Tests:**
    - `test_5_turn_context_retention` — the test IS the acceptance criterion
  - **Accept:** Turn 5 response is contextually aware of turn 1.

---

## Phase 5: Online Learning

GESTALT improves from every interaction.

### 5.1 Experience Buffer
- [ ] **T-026** Persistent experience store (~200 LOC addition to memory.rs)
  - SQLite table: (goal, plan, execution_result, user_feedback, reward)
  - Automatic reward: tool success = +1, failure = -1, user "thanks" = +2
  - **Depends on:** T-019 (memory.rs SQLite infrastructure)
  - **LOC estimate:** ~200
  - **Risk:** LOW. Extension of existing SQLite code.
  - **Tests:**
    - `test_experience_store` — 50 interactions stored and queryable
    - `test_reward_assignment` — success/failure/feedback rewards correct
    - `test_query_by_reward` — highest-reward experiences returned first
  - **Accept:** 50 interactions stored. Queryable by reward, recency, goal type.

### 5.2 Micro-Training Loop
- [ ] **T-027** After every N interactions, fine-tune (~300 LOC)
  - Load recent experience buffer
  - SFT on successful interactions (100-500 steps)
  - GRPO on plan quality (preferred vs rejected)
  - ~200M model trains in seconds on GPU
  - **Depends on:** T-026 (experience buffer), T-002 (training infrastructure)
  - **LOC estimate:** ~300
  - **Risk:** HIGH. Online learning can catastrophically forget. Must preserve base capabilities.
  - **Risk flag:** Need EWC or similar to prevent catastrophic forgetting during micro-training.
  - **Tests:**
    - `test_micro_train_improves` — held-out accuracy improves after 50 interactions
    - `test_micro_train_no_regression` — base plan_bench stays >= 21/21 after micro-training
  - **Accept:** After 50 interactions, held-out accuracy measurably improves.

### 5.3 Memory Consolidation
- [ ] **T-028** Episodic -> semantic consolidation
  - Pattern detection: "user asked about X three times" -> store fact
  - Compress old episodes into abstract summaries
  - **Depends on:** T-019 (memory), T-026 (experience buffer)
  - **LOC estimate:** ~250
  - **Risk:** MEDIUM. Summarization quality depends on language model capability (Phase 2).
  - **Tests:**
    - `test_consolidation_reduces_count` — episodic count decreases after consolidation
    - `test_consolidation_preserves_retrieval` — top-5 retrieval accuracy maintained
  - **Accept:** Consolidation reduces episodic count while preserving retrieval accuracy.

---

## Phase 6: Proactive Intelligence + Integration

GESTALT anticipates, suggests, warns.

### 6.1 Unified Server
- [ ] **T-029** `gestalt serve` full implementation
  - HTTP + stdin interface
  - All regions loaded once at startup
  - Persistent session state across requests
  - Models hot-loaded from checkpoints
  - **Depends on:** T-013 (binary), T-022 (session)
  - **LOC estimate:** ~400
  - **Risk:** MEDIUM. Server architecture, connection handling.
  - **Tests:**
    - `test_server_startup` — starts in <5s
    - `test_server_10_requests` — handles 10 sequential requests without memory leak
    - `test_server_session_persistence` — session state maintained across requests
  - **Accept:** Server starts in <5s. Handles 10 sequential requests without memory leak.

### 6.2 Context Monitoring
- [ ] **T-030** Watch git status, file changes, test results
  - Background polling of workspace state
  - Proactive suggestions: "you modified 3 files but haven't tested"
  - **Depends on:** T-029 (server), T-010 (executor for git/cargo)
  - **LOC estimate:** ~200
  - **Risk:** LOW. Polling + heuristics.
  - **Tests:**
    - `test_detect_untested_changes` — modified files trigger suggestion
  - **Accept:** 1+ useful unprompted suggestion per session.

### 6.3 Dynamic Tool Registry
- [ ] **T-031** Tools loaded from config, not hardcoded
  - TOML config: tool name, command template, timeout, safety level
  - New tools without recompiling
  - **Depends on:** T-010 (executor trait)
  - **LOC estimate:** ~200
  - **Risk:** LOW. Config parsing + trait object construction.
  - **Tests:**
    - `test_load_custom_tool` — tool from TOML config executes correctly
    - `test_plan_uses_custom_tool` — brain generates plan with custom tool
  - **Accept:** Add a custom tool via config. Brain uses it in a plan.

### 6.4 JARVIS Personality
- [ ] **T-032** Personality embedded in language region
  - JARVIS wit/candor/flair trained into responses
  - Consistent persona across all interactions
  - **Depends on:** T-017 (trained language model)
  - **LOC estimate:** ~100 (training data curation + eval)
  - **Risk:** MEDIUM. Personality is subjective. Need eval framework.
  - **Tests:**
    - `test_personality_consistency` — 10 responses all maintain JARVIS style
  - **Accept:** Blind test: 8/10 responses identifiable as JARVIS-style.

---

## Task Summary

| Phase | Tasks | New LOC (est) | Port LOC | Description |
|-------|-------|---------------|----------|-------------|
| 0 | T-001 to T-009 | 0 | ~2,800 | V4 port + consolidation |
| 1 | T-010 to T-013 | ~1,050 | 0 | Tool execution engine |
| 2 | T-014 to T-018 | ~1,250 | 0 | BPE + language scaling |
| 3 | T-019 to T-021 | ~630 | 0 | Persistent memory |
| 4 | T-022 to T-025 | ~1,200 | 0 | Multi-turn + ReAct |
| 5 | T-026 to T-028 | ~750 | 0 | Online learning |
| 6 | T-029 to T-032 | ~900 | 0 | Proactive integration |
| **Total** | **32 tasks** | **~5,780** | **~2,800** | **~8,580 LOC target** |

V4 was 5,452 LOC across 10 files. GESTALT targets ~8,580 LOC across 12+ files while adding tool execution, memory, sessions, ReAct, online learning, and a unified server. The ~30% reduction in ported code (5,452 -> 2,800) comes from merging brain_policy into brain, eliminating standalone talk model, and removing dead weight.

### Risk Summary

| Risk Level | Tasks | Notes |
|------------|-------|-------|
| HIGH | T-005, T-012, T-020, T-021, T-023, T-024, T-025, T-027 | Architecture merges, novel components, full-stack integration |
| MEDIUM | T-004, T-009, T-010, T-011, T-014, T-015, T-017, T-018, T-028, T-029, T-032 | Complex but well-understood patterns |
| LOW | T-001, T-002, T-003, T-006, T-007, T-008, T-013, T-019, T-022, T-026, T-030, T-031 | Direct ports, data structures, config |
