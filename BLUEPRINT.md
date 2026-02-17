# BLUEPRINT -- GESTALT Architecture

## Core Concept

One brain, multiple regions, shared concept space. Language is an interface, not the reasoning substrate. Memory is first-class, not bolted on.

```
         goal_text (UTF-8 bytes)
             |
      [ConceptTokenizer] (Phase 2; byte-level until then)
             |
         token_ids: (1, seq_len)
             |
      [ConceptEncoder] -- transformer layers -> mean pool -> concept_vec
             |
         concept_vec: (1, d_model)
             |
      [ConceptProjector] -- linear -> reshape -> N prefix embeddings
             |
         prefix: (1, N, d_model)    where N=16 default
             |
    +--------+--------+--------+--------+
    |        |        |        |        |
 [Policy] [Planner] [Memory] [Language] [Session]
  heads     FSM      store    decoder    ring buf
    |        |        |        |        |
  intent   plan    retrieve  response  context
 +actions  tokens   top-K    bytes     turns
    |        |        |        |
    +--------+--------+--------+
             |
      [Executor] -- trait Tool { fn execute() -> Result<ToolOutput> }
             |
      [Pipeline] -- run_goal() -> ExecutionResult
```

---

## Tensor Shapes Through the Full Pipeline

Notation: `(B, S, D)` = batch, sequence length, dimension. All shapes at fp32.

### Phase 0-1 (373-token vocab, d_model=512)

```
Input:
  goal_text: &str                           "search jarviscmd and then open it"
  token_ids: (B, S_enc)                     (1, 87) -- byte-level, 1 char = 1 token

ConceptEncoder:
  embed:     (B, S_enc, d_model)            (1, 87, 512) -- token embeddings
  + RoPE:    applied to Q,K in each layer
  Layer 0-3: (B, S_enc, d_model)            (1, 87, 512) -- residual stream
  pool:      (B, d_model)                   (1, 512) -- last-token hidden state extraction

ConceptProjector:
  linear:    (B, N * d_model)               (1, 8192) -- project to N=16 prefixes
  reshape:   (B, N, d_model)                (1, 16, 512) -- prefix embeddings

PolicyHeads (from concept_vec):
  intent:    (B, num_intents)               (1, 16) -- softmax over 16 intent slots (14 defined + 2 reserved)
  actions:   (B, max_actions, num_tools)    (1, 6, 16) -- per-slot tool distribution (15 tools + END)
  patterns:  (B, max_actions, num_pats)     (1, 6, 6) -- per-slot pattern index (PAT0-PAT5)
  files:     (B, max_actions, num_files)    (1, 6, 10) -- per-slot file index (FILE0-FILE9)
  picks:     (B, max_actions, num_picks)    (1, 6, 129) -- per-slot pick index (PICK0-PICK128)

MemoryBank:
  query:     (1, d_model)                   (1, 512) -- concept_vec as query
  keys:      (K, d_model)                   (8, 512) -- top-K stored concept_vecs (default K=8)
  values:    K entries                      8 full (concept_vec, response) tuples
  mem_proj:  (B, K, d_model)               (1, 8, 512) -- projected memory prefix

LanguageDecoder:
  prefix:    (B, N+K, d_model)              (1, 24, 512) -- [concept_prefix(16); memory_prefix(8)]
  response:  (B, S_dec, d_model)            (1, 256, 512) -- autoregressive generation
  logits:    (B, S_dec, vocab)              (1, 256, 259) -- per-position token probs (TALK_VOCAB=259)
  output:    Vec<u8>                        UTF-8 bytes (greedy or sampled)

Plan-LM FSM:
  input:     (B, S_plan)                    (1, 128) -- goal prompt + plan tokens
  logits:    (B, S_plan, 373)               (1, 128, 373) -- full vocab logits
  masked:    (B, S_plan, 373)               (1, 128, 373) -- FSM-masked (-inf on invalid)
  plan:      Vec<u32>                       [STEP, RG, PAT1, STEP, REPOREAD, FILE0, EOP]
```

### Phase 2+ (Concept-Space Tokenizer, adaptive vocab, d_model=1024)

```
  token_ids: (B, S_enc)                     (1, ~15) -- concept tokenizer compresses text
  embed:     (B, S_enc, 1024)               (1, 15, 1024)
  concept_vec: (B, 1024)                    (1, 1024)
  logits:    (B, S_dec, vocab)              (1, 256, vocab) -- adaptive vocab output
```

---

## Data Flow: `run_goal("search jarviscmd and then open it")`

```
1. TOKENIZE
   "search jarviscmd and then open it" -> [115, 101, 97, 114, ...]  (byte-level)
   Future: concept tokenizer merges semantically → fewer tokens, same information

2. ENCODE
   token_ids -> ConceptEncoder (4-layer transformer + mean pool)
   -> concept_vec: (1, 512) float32

3. CLASSIFY (Policy Heads)
   concept_vec -> intent=Composite (idx 10), actions=[RG, REPOREAD, END, PAD, PAD, PAD]
   Confidence: intent 0.94, action0 0.97, action1 0.91

4. PLAN (FSM-Constrained Decode)
   goal tokens ++ [PLAN_SEP] -> Plan-LM greedy_decode with FSM mask
   FSM states: Start -> AfterStep -> AfterAction -> AfterRhs -> ...
   -> [STEP, RG, PAT1, STEP, REPOREAD, FILE0, EOP]

5. COMPILE (Plan -> Steps)
   Plan tokens -> Vec<ToolStep>:
     Step 0: { tool: Rg, args: { pattern: "jarviscmd" } }
     Step 1: { tool: RepoRead, args: { file: step0.output.first_match } }

6. EXECUTE
   Step 0: Rg("jarviscmd") -> "src/main.rs:42:    fn jarviscmd() {"
   Step 1: RepoRead("src/main.rs", line=42) -> file content (200 bytes)

7. RETURN
   ExecutionResult {
     steps: [StepResult { tool: Rg, output: "...", success: true },
             StepResult { tool: RepoRead, output: "...", success: true }],
     final_output: "fn jarviscmd() { ... }",
     success: true,
   }
```

---

## Module Map

| File | Purpose | LOC (est) | Depends On |
|------|---------|-----------|------------|
| `transformer.rs` | RoPE + RMSNorm + MHA + MLP backbone | ~550 | candle-core, candle-nn |
| `tokenizer.rs` | Concept-space tokenizer (adaptive vocab, Phase 2) / plan tokenizer (373, Phase 0-1) | ~500 | none (custom) |
| `brain.rs` | Unified brain: encoder + projector + decoder + policy heads + memory bank | ~1200 | transformer, tokenizer |
| `planner.rs` | 17-state FSM constrained decoder | ~700 | transformer, tokenizer |
| `executor.rs` | Tool trait + 15 tool implementations | ~300 | std::process |
| `pipeline.rs` | `run_goal()` orchestration, step chaining, result collection | ~400 | brain, planner, executor |
| `memory.rs` | SQLite-backed persistent episodic store | ~400 | rusqlite |
| `session.rs` | Multi-turn ring buffer + ReAct loop | ~500 | brain, executor |
| `training.rs` | AdamW, cosine LR, weighted CE, GPU monitoring | ~480 | candle-nn |
| `eval.rs` | 21-goal plan bench + brain policy bench | ~500 | brain, planner |
| `corpus.rs` | Training data pipeline: download, dedup, tokenize (Phase 2) | ~300 | tokenizer |
| `main.rs` | CLI: serve, train, eval, run | ~200 | all |
| `lib.rs` | Module root + re-exports | ~30 | all |

---

## Training Pipeline

Training proceeds in phases. Each phase is independently useful — the system is strictly better after each phase completes.

### Phase 0-1 Training (Foundation + Tool Execution)

```
1. BRAIN POLICY TRAINING (brain.rs)
   Data:    64-task curriculum (16 core + 48 expansion)
   Input:   goal bytes -> concept_vec -> 5 policy heads
   Loss:    weighted CE per head (intent=1.0, actions=1.0, patterns=0.5, files=0.5, picks=0.5)
   Config:  test: 6144 steps, d=64, 1 layer
            full: 16384 steps, d=256, 3 layers
   Output:  trained policy head weights

2. PLAN-LM SFT (planner.rs)
   Data:    21 plan_bench goals -> reference plan token sequences
   Input:   goal prompt + plan tokens (left-padded for encoder, right-padded for decoder)
   Loss:    weighted CE per position (STEP=0.1, action=1.0, prompt=0.0)
   Config:  test: 2000 steps, d=64, 2 layers, 1 head
            full: 40000 steps, d=512, 4 layers, 8 heads
   Output:  trained Plan-LM weights. Gate: plan_bench >= 21/21

3. BRAIN REGIONS SFT (brain.rs)
   Phase 1: Standard SFT with token noise injection (0% -> 2% linear ramp)
   Phase 2: Dialogue-aligned finetuning (position-specific prediction, weighted sampling)
   Data:    ~80 JARVIS dialogue pairs (byte-level)
   Config:  test: 200 SFT + 200 DA steps, d=64, 1+2 layers
            full: 25000 SFT + 8192 DA steps, d=512, 4+4 layers
   Output:  trained encoder + projector + decoder weights

Order: 1 and 2 are independent (parallel). 3 depends on 1 (brain regions use the same encoder).
```

### Phase 2 Training (Concept Tokenizer + Language + Memory)

```
4. CONCEPT-SPACE TOKENIZER BOOTSTRAP
   Method:  Custom tokenizer that learns merges from the concept space itself.
            Phase 1: byte-level bootstrap (existing encoder maps bytes → concept vecs)
            Phase 2: analyze which byte sequences produce similar concept vecs
            Phase 3: merge rules derived from semantic similarity, not frequency
   Target:  Adaptive vocab starting at 256 bytes, growing organically
   Output:  merge table + vocab (serialized, no external deps)
   Key advantage over BPE: compression guided by MODEL'S learned representations,
   not statistical frequency. Compresses what the brain considers meaningful.

5. LANGUAGE REGION SFT (with memory prefix from epoch 0)
   Data:    Expanded JARVIS corpus (300+ dialogues) + technical text
   Method:  Standard SFT with cosine LR, memory prefix always present
   Config:  d=1024, 8 layers, 8 heads, 4096 ff
   Output:  ~200M parameter model. Gate: JARVIS speaks AND remembers.
   Key change from V4: decoder trained WITH memory from day one (M-004 fix).

6. CROSS-SESSION RECALL TEST (Phase 2 gate)
   Store "favorite color is blue" in session 1. Kill process.
   Ask "what's my favorite color?" in session 2. Must answer correctly.
   Gate: speak AND remember across sessions.
```

### Phase 3+ Training (Memory, ReAct, Online Learning)

```
7. MEMORY-AUGMENTED DECODER TRAINING
   Key change from V4: decoder trained WITH memory prefix from epoch 0.
   During SFT, randomly sample 0-K prior dialogues as "memory" context.
   Gate: generation quality does not degrade when memory prefix is present.

8. ONLINE MICRO-TRAINING (Phase 5)
   After every N interactions: SFT on successful ones (100-500 steps).
   GRPO on plan quality (preferred vs rejected plans).
   Gate: held-out accuracy improves after 50 interactions.
```

---

## Inference Pipeline

```
fn run_goal(brain: &Brain, goal: &str) -> Result<ExecutionResult> {
    // 1. Encode
    let tokens = brain.tokenize(goal);
    let concept_vec = brain.encode(&tokens);            // (1, d_model)

    // 2. Classify
    let policy = brain.classify(&concept_vec);           // PolicyOutput
    // policy.intent: Intent enum
    // policy.actions: Vec<ToolId>

    // 3. Plan
    let plan_tokens = brain.plan(&tokens, &concept_vec); // Vec<u32>, FSM-constrained
    let steps = compile_plan(&plan_tokens);              // Vec<ToolStep>

    // 4. Retrieve memory (if available)
    let memories = brain.memory.retrieve(&concept_vec, top_k=5);

    // 5. Execute
    let mut results = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let context = if i > 0 { Some(&results[i-1].output) } else { None };
        let result = executor.execute(step, context)?;   // subprocess, 30s timeout
        results.push(result);
        if !result.success { break; }                    // fail fast
    }

    // 6. Generate response (if conversational)
    let response = if policy.intent.is_conversational() {
        Some(brain.generate(&concept_vec, &memories))    // byte-level or BPE decode
    } else {
        None
    };

    // 7. Store experience
    brain.memory.store(concept_vec, goal, &results);

    Ok(ExecutionResult { steps: results, response, success: all_ok })
}
```

---

## Configuration Matrix

All configs use the same architecture — only dimensions and training budgets differ.

### Transformer (shared backbone)

| Parameter | Test | Default (Phase 0-1) | Full (Phase 2+) |
|-----------|------|---------------------|------------------|
| d_model | 64 | 512 | 1024 |
| n_layers | 1-2 | 4 | 8 |
| n_heads | 1 | 8 | 8 |
| d_ff | 256 | 2048 | 4096 |
| vocab_size | 373 | 373 | adaptive (starts 256, grows) |
| max_seq_len | 128 | 256 | 512 |
| dropout | 0.0 | 0.1 | 0.1 |

### Brain Policy

| Parameter | Test | Full |
|-----------|------|------|
| d_model | 64 | 256 |
| n_layers | 1 | 3 |
| train_steps | 6144 | 16384 |
| curriculum | 16 tasks | 64 tasks |
| num_intents | 14 | 14 |
| num_tools | 15 | 15 |
| max_actions | 6 | 6 |

### Plan-LM

| Parameter | Test | Default |
|-----------|------|---------|
| d_model | 64 | 512 |
| n_layers | 2 | 4 |
| n_heads | 1 | 8 |
| d_ff | 256 | 2048 |
| train_steps | 2000 | 40000 |
| step_weight | 0.1 | 0.1 |
| action_weight | 1.0 | 1.0 |
| FSM states | 17 | 17 |

### Brain Regions (ConceptEncoder + ConceptProjector + LanguageDecoder)

| Parameter | Test | Default |
|-----------|------|---------|
| encoder d_model | 64 | 512 |
| encoder layers | 1 | 4 |
| decoder layers | 2 | 4 |
| n_heads | 1 | 8 |
| prefix_len (N) | 4 | 16 |
| memory_top_k | 3 | 8 |
| sft_steps | 200 | 25000 |
| da_steps | 200 | 8192 |
| noise_max | 0.02 | 0.02 |

---

## Key Architecture Decisions

### Concept Bottleneck Width
- ConceptProjector expands 1 concept_vec -> N prefix embeddings (default N=16)
- Decoder sees 16 concept tokens, not 1 vector
- d_model=512 (Phase 0-1), scaling to d_model=1024 (Phase 2)
- If 1024 proves insufficient, escape hatch to 2048 (fits in 16GB VRAM at fp32)

### Right-Padding Convention
- Decoder sequences: RIGHT-padded (BOS at position 0, pad fills end)
- Encoder sequences: LEFT-padded (content at end for causal attention)
- This ensures RoPE positions match between training and inference
- Violating this caused V4 M-003: decoder generated EOS immediately

### FSM Constrained Decoding
- Plan-LM outputs are ALWAYS syntactically valid
- 17 FSM states govern which tokens are legal at each position
- Invalid tokens get -inf logits before argmax
- This eliminates syntactic hallucination in tool plans
- State machine: Start -> AfterStep -> AfterAction -> AfterRhs -> AfterPat/File/Pick/From -> ... -> Complete

### Memory Architecture
- Episodic: SQLite `(id, timestamp, concept_vec BLOB, goal, response, success)`
- Memory prefix: retrieved concept_vecs projected through memory_projector
- Decoder trained WITH memory prefix from epoch 0 (V4 bug M-004: trained without)
- Capacity: 1024 entries, FIFO eviction
- Retrieval: cosine similarity on concept_vec, top-K results

### Tool Safety
- All tool execution in subprocess with timeout (30s default)
- Mutating tools (patch, fix_tests) require `safety_level` check
- Read-only tools (rg, repo_read, cargo_test) execute freely
- No network access from tools unless explicitly configured

---

## Integration Points

These are the interfaces where modules connect. Breaking these contracts breaks the system.

| From | To | Interface | Contract |
|------|----|-----------|----------|
| tokenizer | brain | `encode(&str) -> Vec<u32>` | Output length <= max_seq_len. Special tokens at correct positions. |
| brain.encoder | brain.projector | `concept_vec: (B, d_model)` | Always exactly d_model dims. Never NaN. |
| brain.projector | brain.decoder | `prefix: (B, N, d_model)` | Exactly N prefix tokens. Same d_model as decoder embedding. |
| brain.policy | pipeline | `PolicyOutput { intent, actions, ... }` | intent is valid Intent enum. actions[i] < num_tools. |
| brain.memory | brain.decoder | `mem_prefix: (B, K, d_model)` | 0 <= K <= memory_top_k. Same d_model. |
| planner | pipeline | `plan_tokens: Vec<u32>` | FSM-valid sequence ending with EOP. All tokens < vocab_size. |
| pipeline | executor | `ToolStep { tool, args }` | tool is valid ToolId. args match tool's expected schema. |
| executor | pipeline | `ToolOutput { stdout, stderr, exit_code }` | Always returns (even on timeout). exit_code 0 = success. |
| training | brain | `checkpoint: safetensors` | All parameter tensors named. Shapes match config. |
| eval | brain + planner | `score_plan_bench() -> (ok, total)` | Strips CoT prefix before comparison. ok <= total = 21. |

---

## Error Handling Strategy

```
GESTALT uses Rust's Result type throughout. No unwrap() except on proven invariants.

Errors propagate with context via anyhow:
  risky_op().context("during X because Y")?

Tool execution errors:
  - Timeout (30s) -> ToolOutput { exit_code: -1, stderr: "timeout" }
  - Process spawn failure -> Result::Err (propagated)
  - Non-zero exit -> ToolOutput with stderr captured (not an Err)

Training errors:
  - NaN loss -> immediate abort with diagnostic dump
  - Gradient explosion (norm > 100) -> clip and log warning
  - CUDA unavailable -> fall back to CPU with warning

Memory errors:
  - SQLite failure -> Result::Err (propagated)
  - Memory full -> FIFO eviction (not an error)
  - No relevant memories -> empty prefix (graceful degradation)
```

---

## Serialization Format

### Model Checkpoints
- Format: safetensors (via candle-nn)
- Naming: `{component}_{timestamp}.safetensors`
- Contains: all parameter tensors with named keys
- Load: `candle_nn::VarBuilder::from_safetensors(path, device)`

### Memory Store
- Format: SQLite database (`memory.db`)
- Schema: `(id INTEGER PRIMARY KEY, timestamp TEXT, concept_vec BLOB, goal TEXT, response TEXT, success BOOL)`
- concept_vec stored as raw f32 bytes (d_model * 4 bytes)

### Session State
- Format: JSON (serde)
- Contains: `Vec<Turn>` with concept_vecs base64-encoded
- Persisted to: `session_{id}.json`

### Concept Tokenizer (Phase 2)
- Format: custom binary (merge table + vocab)
- Naming: `concept_tokenizer.bin`
- Contains: merge rules (byte-pair → merged token, sorted by semantic similarity), vocab map, special tokens

---

## Parameter Budget (Phase 2 target: ~200M)

| Component | Params | Notes |
|-----------|--------|-------|
| Embedding (adaptive x 1024) | ~2-8M | Concept tokenizer vocab (grows organically) |
| Encoder (4 layers) | ~42M | Shared with decoder where possible |
| Decoder (4 layers) | ~42M | Language generation |
| Concept Projector | ~1M | d_model -> 16*d_model |
| Memory Projector | ~1M | d_model -> d_model |
| Policy Heads (5) | ~2M | intent + actions + patterns + files + picks |
| LM Head | ~2-8M | vocab projection (tied with embedding) |
| **Total** | **~105M** | Room to double layers if needed |

Note: 105M with 4+4 layers. Scale to 8+8 layers = ~200M. Fits in ~800MB fp32.

---

## VRAM Budget (RTX 5070 Ti, 16GB)

| Item | Size | Notes |
|------|------|-------|
| Model weights (200M fp32) | 800MB | |
| Optimizer states (AdamW) | 2.4GB | 3x model for m, v, params |
| Activations (batch=21, seq=256) | ~2GB | Depends on layer count |
| KV cache (inference) | 134MB | 8 layers, 256 seq |
| Headroom | ~10GB | Plenty for scaling |
