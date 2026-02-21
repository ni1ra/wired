# GESTALT: Building an AI Brain From Scratch

**A technical guide to WIRED-V5 — what it is, how it works, and why every decision was made.**

*February 2026.*

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [Why Build Your Own AI?](#2-why-build-your-own-ai)
3. [The Big Picture](#3-the-big-picture)
4. [The Concept Bottleneck — The Core Idea](#4-the-concept-bottleneck)
5. [The Transformer — The Building Block](#5-the-transformer)
6. [The Brain — Where Everything Meets](#6-the-brain)
7. [The Policy Heads — Making Decisions](#7-the-policy-heads)
8. [The Planner — Plans That Can't Be Wrong](#8-the-planner)
9. [The Executor — Running Real Tools](#9-the-executor)
10. [The Language Decoder — How It Speaks](#10-the-language-decoder)
11. [The Tokenizer Story — From Bytes to Words](#11-the-tokenizer-story)
12. [Memory — The Brain Remembers](#12-memory)
13. [Training — Teaching the Brain](#13-training)
14. [The Training Story — v2 to v22](#14-the-training-story)
15. [Where It Stands Now](#15-where-it-stands-now)
16. [The Road Ahead](#16-the-road-ahead)
17. [GESTALT vs ChatGPT](#17-gestalt-vs-chatgpt)
18. [Glossary](#18-glossary)

---

## 1. What Is This?

GESTALT is an AI brain built from scratch in Rust. Not a wrapper around OpenAI. Not a fine-tuned LLaMA. From scratch — every matrix multiplication, every attention head, every training loop, written by hand.

If ChatGPT is a fully furnished apartment you rent, GESTALT is a house you're building from the foundation up. We poured the concrete, framed the walls, and wired the electricity ourselves. It's smaller than the apartment — but we own every nail and know where every pipe runs.

The name "gestalt" means *a whole that is greater than the sum of its parts*. That's the core philosophy: one unified brain with specialized regions sharing a common understanding, rather than separate models duct-taped together.

### What It Does (When Finished)

Give it a goal in plain English:

```
"search for the GPU policy code and then open it"
```

And it will:
1. **Understand** what you want (intent: search then read)
2. **Plan** the steps (step 1: ripgrep search, step 2: read file)
3. **Execute** those steps (actually run the tools on your machine)
4. **Remember** what happened (store the experience for next time)
5. **Respond** in natural language ("Found the GPU policy in src/policy.rs...")

All inside a single neural network. No GPT API call. No clever prompt.

### The JARVIS Connection

GESTALT's personality target is JARVIS from Iron Man. Dry wit, competent, helpful, never annoying. The language model is trained on curated JARVIS-style dialogue so it sounds like a British butler who also happens to be a world-class engineer.

This isn't cosmetic — personality consistency is a real engineering problem. The model needs a stable voice across thousands of interactions without drifting into generic chatbot speak.

---

## 2. Why Build Your Own AI?

### What's Wrong With Existing AI

ChatGPT, Claude, and Copilot are phenomenally capable. But they have fundamental limits:

**No persistent memory.** Every conversation starts from zero. Tell Claude your favorite color today, ask tomorrow — blank stare.

**No local tool execution.** They run code in remote sandboxes, not on YOUR machine. GESTALT runs tools in real subprocesses with access to your actual files.

**No learning from experience.** Use ChatGPT for 1,000 tasks and it hasn't improved at all on task 1,001. GESTALT's architecture is designed to get better with every interaction.

**Massive and opaque.** GPT-4 is estimated at ~1.8 trillion parameters. You can't inspect it, modify it, or understand why it made a decision. GESTALT targets ~200M parameters — small enough to understand end-to-end.

### The V4 Story

GESTALT is V5 of the WIRED project. V4 worked — it could classify intents, generate plans, and execute tools. But it had three architectural problems that V5 fixes:

**Brain split.** V4 had a separate "policy brain" (decides what to do) and "language brain" (generates text). They didn't share any understanding. Like having one person who can think but can't speak, and another who can speak but can't think. V5 merges them into one brain.

**Memory bolted on.** V4's decoder was trained without memory, then memory was added later. The decoder never learned to *use* memories. Like learning to cook for 10 years and then someone gives you a recipe book — you've already developed all your habits without it. V5 trains with memory from day one.

**Tokenizer ceiling.** V4 used a 373-token vocabulary. Enough for structured plans, nowhere near enough for natural language. V5 scales to a concept-space tokenizer.

---

## 3. The Big Picture

Here's the full architecture. Every component is explained in its own section below.

```
         goal_text (plain English)
             |
      [ConceptTokenizer]    converts text to token IDs
             |
         token_ids
             |
      [ConceptEncoder]      transformer layers → mean pool
             |
         concept_vec         THE BOTTLENECK — 512 numbers that ARE the meaning
             |
      [ConceptProjector]    expands to 16 prefix tokens
             |
         prefix: 16 tokens
             |
    +--------+--------+--------+--------+
    |        |        |        |        |
 [Policy] [Planner] [Memory] [Language] [Session]
  heads     FSM      store    decoder    ring buf
    |        |        |        |
  intent   plan    retrieve  response
 +actions  tokens   top-K    text
    |        |        |        |
    +--------+--------+--------+
             |
      [Executor]     runs real tools in subprocesses
             |
      [Pipeline]     orchestrates the whole flow
             |
         Result { steps, response, success }
```

### The Restaurant Analogy

Think of GESTALT as a restaurant:

- **The ConceptEncoder** is the waiter who listens to your order and writes it down on a single notecard (the concept vector).
- **The PolicyHeads** are the head chef reading the notecard and deciding: "This is a pasta order — we need the pasta station and the sauce station."
- **The Planner** is the sous chef writing the exact recipe: "Step 1: boil water. Step 2: add pasta. Step 3: make sauce."
- **The Executor** is the line cook actually doing the work — turning on the stove, boiling the water.
- **The LanguageDecoder** is the waiter again, translating the kitchen's work back into "Here's your carbonara."
- **Memory** is the restaurant's notebook: "Table 4 is allergic to shellfish" — consulted every time that customer returns.

Every region reads from the same notecard. That shared understanding is what makes it a *gestalt*, not a Rube Goldberg machine.

---

## 4. The Concept Bottleneck

This is the single most important idea in the architecture.

### What It Is

When you type "search for the GPU policy code," the ConceptEncoder reads every byte, passes it through transformer layers, and **mean-pools** across all non-padding positions to produce a single vector of 512 floating-point numbers.

```
"search for the GPU policy code"
    ↓
[115, 101, 97, 114, 99, 104, ...]    ← raw bytes
    ↓ ConceptEncoder (transformer)
    ↓ mean pool non-PAD positions
[0.23, -1.07, 0.84, ..., -0.31]      ← concept_vec (512 numbers)
```

That 512-dimensional vector **IS** the meaning. Not a summary. Not a hash. A learned compression where similar meanings map to nearby points in vector space.

### Why Force a Bottleneck?

Imagine explaining a coding task to a colleague. You could read them the entire 2,000-line file. Or you could say "the GPU policy enforcer in the policy module." Both convey the same *intent*, but the second one strips away everything except what matters.

The bottleneck forces the encoder to learn that compression. It has 30+ bytes of input but only 512 numbers of output. It must learn which parts of "search for the GPU policy code" matter (intent: search, target: GPU policy) and which are noise (the words "for" and "the").

### How It Expands

512 numbers are great for decisions (the policy heads) but not enough for writing a paragraph. So the ConceptProjector expands:

```
concept_vec: (1, 512)     ← compressed meaning
    ↓ linear layer + reshape
prefix: (1, 16, 512)      ← 16 "concept tokens" for the decoder
```

One vector becomes 16 tokens. The decoder reads these tokens as a prefix — they set the context for what it's about to say.

Think of the concept_vec as a sticky note that says "search + GPU policy." The prefix is an expanded brief that says "The user wants to search the codebase for GPU-related policy code, probably a Rust file, and they want to see the contents." Same information, more surface area for the decoder.

### Why Mean Pooling?

An earlier version (v2-v13) used **last-token extraction** — taking the hidden state at the final sequence position. This failed catastrophically because every input ends with the same padding/EOS pattern. Different inputs produced nearly identical concept vectors (cosine similarity 0.96). Mean pooling over non-PAD positions gives every token's representation equal weight, producing genuinely discriminative vectors (cosine similarity 0.25 after training).

---

## 5. The Transformer

Every neural computation in GESTALT uses the same building block: a **causal transformer**. If you've heard of GPT, it's the same architecture, implemented from scratch in Rust using the candle library.

### What It Does

A transformer takes a sequence of token embeddings and makes each one aware of context:

```
Input:  [tok_1, tok_2, tok_3, tok_4]     ← 4 embeddings, each 512-dim
         ↓ Layer 0
        [out_1, out_2, out_3, out_4]      ← updated: context-aware
         ↓ Layer 1 ... Layer 3
Output: [out_1, out_2, out_3, out_4]      ← final representations
```

Each layer has two sub-components:
1. **Multi-Head Attention** — each position looks at other positions and decides what to focus on
2. **Feed-Forward Network (MLP)** — processes each position independently

### Attention in Plain English

For each position, the transformer computes:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I have to offer?"
- **Value (V)**: "What information do I carry?"

The attention score between positions is how well a Query matches a Key. High match = more attention.

GESTALT uses **8 attention heads** in parallel. Each head can focus on something different — one tracks syntax, another tracks semantics, another tracks tool references.

### Position: RoPE

Transformers don't inherently know where tokens are. "The cat sat" and "sat cat the" look the same without position info. RoPE (Rotary Position Embedding) solves this by rotating the Q and K vectors based on position. Nearby tokens get similar rotations (strong attention), distant tokens get different rotations (weaker attention).

### RMSNorm and GELU

**RMSNorm** normalizes each vector to prevent numbers from growing too large or too small through layers. Like a volume knob keeping the signal clean.

**GELU** is the activation function — a smooth gate that lets most negative values through slightly. Gives the model more expressiveness than a hard on/off switch.

### Important: candle-nn Bugs

Two critical bugs in candle-nn v0.8.4 broke GESTALT for 12 training runs:

- `candle_nn::RmsNorm` **silently kills the gradient graph**. Tensors passing through it become dead ends for backpropagation.
- `candle_nn::ops::softmax_last_dim` **same problem**. Breaks the gradient chain inside attention.

Both were replaced with custom implementations (`GradRmsNorm` and `grad_softmax_last_dim` in transformer.rs) using only basic tensor ops that correctly propagate gradients. This is described in full in [Section 14: The Training Story](#14-the-training-story).

### GESTALT's Configurations

| Parameter | Test | Default (d=512) | Target (d=1024) |
|-----------|------|-----------------|------------------|
| d_model   | 64   | 512             | 1024             |
| n_layers  | 1-2  | 4               | 8                |
| n_heads   | 2-4  | 8               | 8                |
| d_ff      | 128  | 2048            | 4096             |
| max_seq   | 128  | 256             | 512              |

"Test" is for unit tests — tiny enough to run in milliseconds on CPU. "Default" is what we train on now.

---

## 6. The Brain

The `Brain` struct is GESTALT's central nervous system. It lives in `brain.rs` (~2,000 lines — the largest file).

### Why One Brain?

In V4, two separate models processed the same input into different vector spaces. Like having two translators who both read your email but translate into different languages — they can't share notes.

V5 has ONE encoder. The concept_vec it produces feeds:
- Policy heads (deciding intent and actions)
- The planner (as a prefix for plan generation)
- The language decoder (as a prefix for text generation)
- Memory (as a key for storage and retrieval)

One understanding. Multiple uses.

### Components

```
Brain {
    encoder: WiredTransformer,      // text → concept_vec
    projector: Linear,              // concept_vec → 16 prefix tokens
    language_decoder: WiredTransformer,  // prefix → response text
    policy_intent: Linear,          // intent classification
    policy_actions: Linear,         // tool sequence prediction
    policy_patterns: Linear,        // search patterns
    policy_files: Linear,           // file targets
    policy_picks: Linear,           // result selection
    memory_projector: Linear,       // memory → decoder prefix
    memory_bank: MemoryBank,        // in-memory episodic store
}
```

---

## 7. The Policy Heads

When the brain receives a goal, the first thing it does is classify: what *kind* of task is this? Five linear heads answer five questions from the same concept_vec, in a single forward pass (~1ms on GPU).

**Intent Head** — 16 intent slots: Hello, RunTests, CargoCheck, RepoSearch, RepoRead, Composite, etc. "hello" → Hello (95%). "search for GPU policy" → RepoSearch (90%).

**Action Head** — 6 slots, each predicting one of 15 tools (or END). For "search for the GPU policy code and then open it": slot 0 = rg, slot 1 = repo_read, slot 2 = END.

**Pattern Head** — Which search pattern to use. **File Head** — Which file to target. **Pick Head** — Which search result to select.

All five decisions happen simultaneously from the same concept_vec.

### The Curriculum

64 tasks: 16 core (one per intent) + 48 variations. Training runs for 16,384 steps with weighted cross-entropy.

---

## 8. The Planner

The planner generates executable tool plans with a guarantee that no other system has: **every plan is syntactically valid**, always.

### The Problem

A regular language model might produce:
```
STEP rg STEP STEP repo_read EOP
```
That's nonsense. Two STEPs in a row? An rg without a search pattern? Regular models don't know the grammar of valid plans.

### The Solution: FSM-Constrained Decoding

GESTALT's planner has a **17-state finite state machine** tracking where it is in the plan grammar. At each step, the FSM says: "Given the current state, these are the only tokens that are legal right now." All illegal tokens get set to -infinity probability.

```
State Machine (simplified):

  Start ──[STEP]──→ AfterStep ──[action]──→ AfterAction
                                                 |
                    ┌─[has args]──→ AfterRhs ──[arg]──→ ...
                    └─[no args]──→ Complete? ──[STEP or EOP]──→ ...
```

Think of it like Scrabble where after each word, the rules physically remove certain letters from your rack. You CAN'T make an illegal move because the illegal pieces aren't available.

### Why It Matters

The model might choose the *wrong* action (predicting `rg` when it should be `cargo_test`), but it will never produce a *syntactically invalid* plan. Every plan can be parsed and executed. This eliminates an entire class of failures.

### The STEP Token Problem

STEP tokens make up ~27% of plan tokens but carry almost no information — they're just separators. Without position-weighted loss, the model learns to predict STEP perfectly while ignoring the important action tokens. Fix: weight STEP at 0.1, actions at 1.0. Three lines of code. Found after 30+ experiments (mistake M-002).

---

## 9. The Executor

GESTALT runs tools locally — not in a remote sandbox.

Each of the 15 built-in tools is a real subprocess with a 30-second timeout:

| Tool | What It Does | Safety |
|------|-------------|--------|
| rg | Ripgrep search | ReadOnly |
| repo_read | Read a file | ReadOnly |
| repo_list | List directory | ReadOnly |
| cargo_test | Run tests | Meta |
| cargo_check | Type check | Meta |
| memory_add | Store memory | Mutating |
| memory_search | Search memories | ReadOnly |
| patch_dry_run | Test a patch | Mutating |
| talk | Generate response | Meta |
| ... | 6 more | Various |

**Safety levels** are real, not theatre. ReadOnly tools never modify anything. Mutating tools require explicit `allow_writes: true` or they refuse. Like Unix file permissions — you have to grant `rw-` before the action, not after.

---

## 10. The Language Decoder

The decoder is how GESTALT talks back to you. It's a causal transformer (same architecture as the encoder, different weights) that generates text one token at a time.

### How It Works

```
What the decoder sees:
  Position 0-15:   [concept prefix]     "Here's what the user wants"
  Position 16-23:  [memory prefix]      "Here's what I remember" (up to 8)
  Position 24:     [BOS]                Start generating
  Position 25+:    [response tokens]    Generated one at a time
```

Generation is autoregressive — each new token depends on all previous ones:

```
Step 1: [...prefix...][BOS]                              → predicts "H"
Step 2: [...prefix...][BOS][H]                           → predicts "e"
Step 3: [...prefix...][BOS][He]                          → predicts "l"
...
Step N: [...prefix...][BOS][Hello. What can I do for you] → predicts EOS
```

### From Bytes to Concept Tokens

The decoder originally generated at **byte level** — each token was one of 256 bytes plus PAD, BOS, EOS (259 total). This works for any text (Unicode, code, special characters) but is slow — 100 characters requires 100 forward passes.

Phase 2.5 upgraded to the **ConceptTokenizer** (see [Section 11](#11-the-tokenizer-story)), where common byte sequences are merged into single tokens. "Hello" becomes ~2-3 tokens instead of 5 bytes. Faster generation, better learning.

### The Memory Prefix: V5's Key Innovation

In V4, the decoder was trained WITHOUT memory, then memory was bolted on. The decoder never learned to look at memory tokens — it ignored them entirely. Like giving someone glasses after they've already learned to navigate blind.

In V5, the decoder trains WITH memory from epoch 0. Random previous dialogues are injected as "memory prefix" tokens during training. The decoder learns to attend to these tokens naturally. This was the M-004 fix.

### Sampling Improvements

Generation uses several techniques to improve output quality:

- **Temperature**: Controls randomness. 0 = always pick the highest-probability token (greedy). 0.7 = mix in some randomness. 1.0 = full randomness.
- **Top-K**: Only consider the K most probable tokens (default 40). Prevents low-probability garbage.
- **Top-P (nucleus)**: Only consider tokens whose cumulative probability reaches P (default 0.9). Adapts the candidate pool based on confidence.
- **Repetition penalty**: Penalizes tokens that appeared in the last 50 generated tokens (windowed). Prevents "the the the the" loops.

---

## 11. The Tokenizer Story

This is one of GESTALT's most important evolution stories — and a key discovery about what makes language models learn effectively.

### Phase 0-1: Byte Level

Originally, every character was its own token. "Hello" = 5 tokens: [H, e, l, l, o]. Simple, universal, but slow and hard to learn from — the model has to figure out that "H" + "e" + "l" + "l" + "o" means the same thing as the concept of greeting.

### Phase 2: The ConceptTokenizer

The ConceptTokenizer learns **BPE merges** from the training corpus. BPE (Byte Pair Encoding) works like this:

1. Start with 256 tokens (one per byte)
2. Find the most common adjacent pair in the corpus (e.g., "t" + "h")
3. Create a new token "th" and add it to the vocabulary
4. Repeat for N merges

With 200 merges, the vocabulary grows from 256 to 459 tokens. "Hello" might become 2-3 tokens instead of 5. The text gets **compressed** while preserving meaning.

### The Key Discovery: Fewer Merges = Faster Learning

This was a surprise. We expected more merges (bigger vocabulary) to help because each token carries more information. Instead:

```
Grid search at 300 training steps:
  50 merges  (306 vocab):  val_loss = 2.79   ← best at short training
  100 merges (356 vocab):  val_loss = 3.13
  200 merges (459 vocab):  val_loss = 3.56
  500 merges (759 vocab):  val_loss = 4.08
  1000 merges (1259 vocab): val_loss = 4.70
  2000 merges (2259 vocab): val_loss = 5.53  ← worst
```

**Why?** More tokens = larger embedding table = more parameters to learn from the same data. A 459-token vocabulary with 21K training pairs gives the model enough signal per token. A 2,259-token vocabulary spreads the same signal too thin.

**Sweet spot: 200 merges** (459 tokens, ~2x compression ratio). Below 150 approaches byte-level. Above 500, the model can't learn enough per token. The curve is flat between 180-220 merges — diminishing returns around 200.

### Tokenizer Bootstrap

Originally, the tokenizer called the neural encoder for every byte pair to score merges by "semantic similarity." With 3K training pairs and 8.6M n-gram occurrences, this took 83+ minutes.

The fix: skip the encoder entirely. Score merges by frequency × compression (pure BPE-style). **2 seconds.** The "concept-aware" scoring was useless anyway because the context-free encoder produced the same vector for the same bytes regardless of context (consistency always = 1.0). Real context-aware scoring requires a trained context-dependent encoder, which is future work.

---

## 12. Memory

GESTALT has two memory systems: fast in-memory retrieval during inference, and persistent SQLite storage across sessions.

### In-Memory Bank

The MemoryBank stores recent experiences as concept vectors. When a new goal comes in, we compute its concept_vec and find the K stored vectors most similar to it using **cosine similarity**:

```
cosine_sim = dot(A, B) / (|A| × |B|)
```

Two vectors pointing the same direction = similarity 1.0 (identical meaning). Perpendicular = 0.0 (unrelated). Opposite = -1.0.

Think of memories as stars in the sky. When you ask a question, you point your telescope at the concept_vec direction. The closest stars are the most relevant memories.

### Persistent Store (SQLite)

The `EpisodicMemory` wraps SQLite for disk persistence:

```sql
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY,
    timestamp TEXT, concept_vec BLOB,
    goal TEXT, response TEXT, success INTEGER
);
```

Tell GESTALT your favorite color in session 1, ask about it in session 100, and it will remember. The concept_vec is stored as raw f32 bytes (512 × 4 = 2,048 bytes per memory).

### Capacity: 1,024 entries, FIFO eviction

When full, the oldest memory is deleted. Phase 5 will add **consolidation** — detecting patterns ("user asked about X three times") and compressing episodic memories into abstract facts.

### Cross-Session Persistence

The T-021 integration test verifies the full lifecycle: store memories → persist to SQLite → drop everything → load from SQLite in a new session → validate the memories survived. All 8 integration tests pass.

---

## 13. Training

### The Optimizer: AdamW

Standard transformer optimizer. Maintains momentum (average of past gradients) and velocity (average of past squared gradients) plus weight decay to prevent weights from growing too large.

### Cosine Learning Rate Schedule

```
LR
 ^
 |    /‾‾‾‾\
 |   /       \
 |  /         \__________
 | /
 +----------------------------→ Steps
   warmup  peak  cosine decay
```

**Warmup** (first 10%): LR ramps from 0 to max. Prevents early instability — random initial weights would produce huge gradients at high LR.

**Cosine decay** (remaining 90%): LR smoothly decreases, letting the model make finer adjustments near convergence.

### Weighted Cross-Entropy Loss

The loss measures how surprised the model is by the correct answer. "Weighted" means different positions contribute differently — STEP tokens at 0.1, action tokens at 1.0 — so the model focuses on what matters.

### Early Stopping

Monitors validation loss every N steps. If it hasn't improved for `patience` consecutive checks, training stops and the best checkpoint is kept. This prevents overfitting — training too long causes the model to memorize training data while getting worse on new inputs.

**Dropout** (0.1): During training, randomly zeroes 10% of values in each layer. Forces the model to not rely on any single neuron, improving generalization. Extends useful training from ~6K steps (without dropout) to 13K+ steps (with dropout).

### PAD-Denoising: Forcing the Decoder to Listen

Without noise, the decoder can predict the next byte from autoregressive context alone (bigram statistics). It ignores the concept prefix entirely — loss plateaus at exactly 2.15 (bigram baseline).

The fix: replace random bytes with PAD tokens during training. PAD carries ZERO information, so the only way to predict correctly is to attend to the concept prefix.

Noise schedule: quadratic ramp from 0% to 10%. Start clean so the model learns basics, then gradually corrupt to force prefix dependency.

### Training Pipeline

Training happens in stages:
1. **Brain SFT** (~30K steps): Encoder + decoder on 21K dialogue pairs
2. **Planner SFT** (4K-40K steps): Plan generation on 21 reference plans
3. **Policy** (16K steps): Intent + action classification on 64-task curriculum

Brain SFT is the bottleneck (~90 minutes on GPU). Planner and policy are fast.

---

## 14. The Training Story — v2 to v22

This is the most important section. It chronicles every major training run, what broke, what we learned, and how each failure led to the next breakthrough. Understanding this story is understanding how GESTALT actually works.

### Phase 1: The Invisible Bug (v2-v12)

The first training run (v2) looked fine. Loss started at 6.3 and dropped steadily. But when we checked the encoder's output, something was wrong.

We measured **concept vector similarity**: encode 50 different prompts and compute average pairwise cosine similarity. If the encoder works, different prompts should produce different vectors (similarity 0.2-0.5).

Result: **0.9591**.

"hello" and "what is the meaning of life" were producing essentially the same concept vector. The encoder was collapsed.

What followed was 12 experiments trying to fix this through training strategy changes:

```
v2:  Baseline SFT                    → sim = 0.9591
v3:  Higher noise                    → sim = 0.9591
v4:  Diversity penalty               → sim = 0.9591
v5:  Contrastive loss                → sim = 0.9591
v6:  Learned classifier              → sim = 0.9591
v7:  Codebook bypass                 → sim = 0.9591
v8:  Detached encoder                → sim = 0.9591
v9:  Two-phase training              → sim = 0.9591
v10: Reduced depth                   → sim = 0.9591
v11: Different LR                    → sim = 0.9591
v12: Encoder pre-training            → sim = 0.9591
```

Twelve experiments. Twelve identical results to four decimal places.

**That constancy was the diagnostic signal.** Different strategies, different losses, different architectures — and the metric doesn't move at ALL? The similarity isn't "converging" to 0.96. It's stuck there from step 0. The problem isn't training. It's structural.

### The Diagnostic

We stopped guessing and built a test. One question: does gradient flow from the loss, through the decoder, back through the encoder?

```rust
// The test that changed everything
#[test]
fn test_candle_op_gradient_flow() {
    let emb = candle_nn::embedding(10, 8, vb.pp("emb")).unwrap();
    let norm = candle_nn::rms_norm(8, 1e-5, vb.pp("norm")).unwrap();

    let hidden = emb.forward(&input).unwrap();
    let normed = norm.forward(&hidden).unwrap();
    let loss = normed.sqr().unwrap().mean_all().unwrap();
    loss.backward().unwrap();

    assert!(emb.weight().grad().is_some());   // FAILS — no gradient!
}
```

`candle_nn::RmsNorm` **disconnects the computation graph**. Every tensor that passes through it becomes a dead end. Same bug in `softmax_last_dim`.

In a transformer, RmsNorm appears in every layer (before attention and before MLP) and softmax appears in every attention computation. So: **zero trainable parameters received gradients except the final LM head.** The encoder literally could not learn. Its 0.9591 similarity was just random initialization noise.

### The Fix

Two custom implementations using basic tensor ops:

```rust
// GradRmsNorm — replaces candle_nn::RmsNorm
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (variance + self.eps)?.sqrt()?;
    x.broadcast_div(&rms)?.broadcast_mul(&self.weight)
}

// grad_softmax_last_dim — replaces candle_nn::ops::softmax_last_dim
let max = xs.max_keepdim(D::Minus1)?;
let exp = xs.broadcast_sub(&max)?.exp()?;
exp.broadcast_div(&exp.sum_keepdim(D::Minus1)?)
```

After the fix: **all 57 trainable parameters** receive non-zero gradients.

**Time to find through blind experiments: 40+ hours, 12 runs.**
**Time to find through diagnostic testing: 2 hours, 1 run.**

### v14: First Successful Run (242 pairs)

With working gradients + mean pooling:

```
SFT 25,000 steps on 242 JARVIS dialogue pairs:
  Loss:  6.37 → 0.04 → 0.001 → 0.0000 (near-perfect)
  Similarity: 0.96 → 0.42 → 0.28 → 0.25 (discriminative!)
  GPU: 84-98% utilization, 13.6/16 GB VRAM, 2.2 steps/sec
  Time: 3 hours 10 minutes
```

Greedy generation:
```
"hello"           → "Hello, sir. What can I do for you?"
"What is beauty?"  → "Pattern recognition with emotional payoff."
"good evening"     → "Evening. I take it we have work to do?"
```

The encoder was finally learning. Different prompts produced genuinely different concept vectors. The JARVIS personality came through. **But** — this was memorization of 242 pairs. The model perfectly reproduced training data and couldn't handle novel prompts.

### v18: Scaling to 21K Pairs

We built a proper corpus pipeline (`scripts/build_corpus.py`, ~1,050 LOC) that downloads and processes three public datasets:
- **Dolly** (Databricks): instruction-following pairs
- **OASST2** (Open Assistant): conversational data
- **Alpaca** (Stanford): diverse instruction pairs

After deduplication: **21,786 unique dialogue pairs** (8.3MB). 10x more data than v14.

v18 results (still byte-level, 259 vocab):
```
val_loss = 2.02 at step 2K, early-stopped at 12K
Gallery: coherent outputs are verbatim training data, novel prompts = gibberish
```

Diagnosis: 512-dim model with 1,749 pairs = ~3 dims per example → trivial memorization. With 21K pairs, the model learned openings but couldn't generalize.

### v19-v20: ConceptTokenizer + Dropout Discovery

**v19** (merges=2000, no dropout): best val=3.55 at step 6K, 0% coherent gallery. The vocabulary was too large — 2,259 tokens meant too many parameters per too few examples.

**v20** (merges=2000, dropout=0.1): best val=3.26 at step 8K, ~30% coherent. Dropout extended training life from 6K to 13K+ steps before overfitting. Good openings, but sentences dissolved into word salad.

### v22: The Grid Search Breakthrough

We built grid search infrastructure (env vars: `GESTALT_MERGES`, `GESTALT_DROPOUT`, `GESTALT_SFT_STEPS`) to test merge counts systematically.

**Grid search results at 300 steps:**
```
merges=50   → val=2.79  (nearly byte-level, fast but limited)
merges=100  → val=3.13
merges=200  → val=3.56  ← sweet spot
merges=500  → val=4.08
merges=1000 → val=4.70
merges=2000 → val=5.53  (vocabulary too large to learn)
```

**v22 mock hero** (merges=200, dropout=0.1, 5K steps): val=2.04. Grammar breakthrough but semantics weak.

**v22 hero** (merges=200, dropout=0.1, 30K steps): The definitive Phase 2.5 run.

### v22 Hero Results — 166 Prompts Evaluated

**What works (~40-45% coherent openings, ~10-15% fully coherent):**

```
"good morning"
  → "Morning! How'd you sleep?"
    ✓ Perfect. Natural, conversational, in-character.

"The build broke and I don't know why"
  → "Step one: read the error message. The whole thing, not just the last line."
    ✓ Excellent. Practical, specific, personality intact.

"tell me about yourself"
  → "I'm a language model with opinions, a dry sense of humor, and a generator conversation."
    ✓ Strong opening. JARVIS voice captured. Degrades after ~15 tokens.

"What do you think about Neon Genesis Evangelion?"
  → "A mecha show that's actually about depression, parental abandonment, and the hedgehog's
     dilemma..."
    ✓ Remarkable. Captured the actual critical analysis of NGE.

"I'm scared of failing"
  → "Good. Fear of failure means you care about the..."
    ✓ Emotionally intelligent opening.
```

**What breaks:**

After ~15 tokens, outputs degenerate into word salad. The model interpolates between training examples mid-sentence:
```
"what is a transformer"
  → "A novel that's referring to the computer science. We can be seen as a bad linear fast
     and it's insurance to former."

"What is beauty?"
  → "Hmm, let me think... it's when something hits are the bad. It's the decision notice,
     but fear - that's rightly proved that ating with line is just optimization..."
```

**The pattern**: First ~10-15 tokens are often correct and stylistically on-point. After that, the model runs out of "memorized opening" and starts interpolating between training examples. Common words and phrases from different dialogues get mixed together.

### What v22 Proves

1. **The architecture works.** One encoder → concept bottleneck → decoder produces coherent, personality-consistent openings.
2. **The tokenizer matters.** merges=200 (459 vocab) dramatically outperforms merges=2000 (2,259 vocab) on the same data.
3. **The capacity bottleneck is real.** 512-dim, 21K pairs is enough to learn patterns but not enough to sustain coherence. The model generalizes openings but not full responses.
4. **The path forward is clear:** more data, bigger model (d=1024), or both.

### Training Run Summary

| Run | Corpus | Vocab | Dropout | Steps | Best Val | Gallery |
|-----|--------|-------|---------|-------|----------|---------|
| v2-v12 | 242 | 259 (bytes) | 0.0 | varies | N/A | Encoder dead (M-032) |
| v14 | 242 | 259 (bytes) | 0.0 | 25K | ~0.00 | Perfect memorization |
| v18 | 1,749 | 259 (bytes) | 0.0 | 12K | 2.02 | Memorized verbatim |
| v19 | 21,786 | 2,259 | 0.0 | 6K | 3.55 | 0% coherent |
| v20 | 21,786 | 2,259 | 0.1 | 8K | 3.26 | ~30% coherent |
| v22 mock | 21,786 | 459 | 0.1 | 5K | 2.04 | Grammar good, semantics weak |
| **v22 hero** | **21,786** | **459** | **0.1** | **30K** | **~1.9** | **~40-45% coherent openings** |

---

## 15. Where It Stands Now

### Source Files (February 2026)

| File | ~LOC | Tests | What It Does |
|------|------|-------|-------------|
| brain.rs | 2,000 | 20 | Unified brain: encoder, decoder, policy, memory, generation |
| transformer.rs | 650 | 8 | Causal transformer with gradient-safe RMSNorm and softmax |
| tokenizer.rs | 1,100 | 22 | PlanTokenizer (373) + ConceptTokenizer (adaptive BPE) |
| planner.rs | 700 | 6 | 17-state FSM constrained plan decoder |
| training.rs | 550 | 10 | AdamW, cosine LR, weighted CE, early stopping |
| eval.rs | 500 | 10 | 21-goal plan bench + brain policy bench |
| executor.rs | 475 | 9 | 15 tools, 3 safety levels, subprocess execution |
| pipeline.rs | 450 | 8 | run_goal() orchestration, step chaining |
| memory.rs | 400 | 9 | SQLite episodic memory, persistence, consolidation |
| session.rs | 200 | 10 | Ring buffer, ReAct phases |
| main.rs | 200 | - | CLI: train / gallery / serve / run |
| lib.rs | 14 | - | Module root |
| integration.rs | 200 | 8 | End-to-end pipeline tests |
| **Total** | **~7,400** | **127** | |

### Test Results

```
$ cargo test --release --features cuda
running 119 tests ... ok         (lib)
running 8 tests ... ok           (integration)
test result: 127 passed; 0 failed; 0 ignored
```

### Phase Progress

```
Phase 0: Foundation Port        ████████████████████ 100%
Phase 1: Tool Execution         ████████████████████ 100%
Phase 2: BPE + Language         ██████████████░░░░░░  70%  (tokenizer done, v22 trained, eval done)
Phase 2.5: Memory Integration   ████████████████████ 100%  (code complete, T-020/T-021 done)
Phase 3: Memory-Augmented Train ██░░░░░░░░░░░░░░░░░░  10%  (architecture ready, training pending)
Phase 4: Multi-Turn + ReAct     ████░░░░░░░░░░░░░░░░  20%  (session ring buffer done)
Phase 5: Online Learning        ░░░░░░░░░░░░░░░░░░░░   0%
Phase 6: Proactive + JARVIS     ░░░░░░░░░░░░░░░░░░░░   0%
```

### Hardware

- **GPU**: NVIDIA RTX 5070 Ti — 16GB VRAM, Blackwell architecture
- **RAM**: 24GB system memory
- **OS**: WSL2 on Windows (Linux kernel 6.6.87)
- **VRAM usage**: ~5.3GB during training (model + optimizer + activations), ~10.7GB headroom

---

## 16. The Road Ahead

### Next: Scale Up (d=1024)

The biggest single improvement available. Going from d=512 to d=1024 means:
- 4x more capacity per concept vector
- 8 encoder + 8 decoder layers (vs 4+4 now)
- ~200M parameters total (vs ~50M now)
- Should sustain coherence well past 15 tokens

### More Data

21K pairs isn't enough. The corpus pipeline can scale — add more public datasets, filter for quality, target 100K+ pairs. More data = more patterns to learn = less interpolation in generation.

### Phase 3: Memory-Augmented Training

Train the decoder with memory prefix containing randomly sampled prior dialogues. The memory_projector weights (currently random) will learn to project retrieved memories into useful context. This is the M-004 fix in action — memory from day one.

### Phase 4: Multi-Turn + ReAct

Make GESTALT conversational and autonomous:
- **Session state**: 32-turn ring buffer for conversation context
- **ReAct loop**: Reason → Act → Observe → Reason for complex tasks
- **Multi-turn context**: Maintain coherence across conversation turns

### Phase 5: Online Learning

GESTALT improves from every interaction:
- Store interactions with automatic reward assignment
- Micro-training: fine-tune on successful interactions every N tasks
- Memory consolidation: compress episodes into abstract knowledge

### Phase 6: Proactive Intelligence

The endgame. GESTALT anticipates and suggests:
- Persistent HTTP server
- Watch file changes, test results, git status
- Dynamic tool registration
- JARVIS personality consistency trained into the weights

---

## 17. GESTALT vs ChatGPT

| Aspect | ChatGPT (GPT-4) | GESTALT V5 |
|--------|-----------------|------------|
| Parameters | ~1.8 trillion | ~50M (current), ~200M (target) |
| Architecture | Mixture of Experts | Single unified transformer |
| Vocabulary | ~100K tokens (BPE) | 459 tokens (concept-space BPE) |
| Training data | Trillions of tokens | 21K curated pairs |
| Memory | None (per-session) | Persistent SQLite + episodic |
| Tool use | Function calling (text) | Real subprocess execution |
| Planning | Implicit | Explicit FSM-constrained |
| Personality | System prompt (fragile) | Trained into weights |
| Training time | Months on thousands of GPUs | ~90 minutes on one GPU |
| Cost to run | API pricing | Free (local GPU) |
| Customizability | Black box | Full (own every weight) |

### What GESTALT Can't Do

**General knowledge.** ChatGPT has read the internet. GESTALT has read 21K dialogue pairs. It won't answer your history questions.

**Instruction following.** ChatGPT has been RLHF'd on millions of preferences. GESTALT has a 64-task curriculum.

**Language quality.** ChatGPT generates fluid prose. GESTALT generates ~10-15 coherent tokens before degrading. Phase 2 scaling will help.

### What GESTALT Can Do Better

**Memory.** Across sessions. Across restarts. ChatGPT doesn't.

**Local execution.** Real tools on your machine with timeout enforcement and safety levels.

**Plan validity.** Every plan is syntactically valid, guaranteed. ChatGPT can generate malformed JSON.

**Transparency.** Inspect every weight, every attention pattern, every training step.

**Cost.** Once trained, runs on your GPU for free.

---

## 18. Glossary

**Attention** — Mechanism letting each position look at others and decide what to focus on.

**Autoregressive** — Generating one token at a time, each depending on all previous tokens.

**BOS/EOS** — Begin/End Of Sequence. Special tokens marking start and end of text.

**BPE (Byte Pair Encoding)** — Tokenization that learns to merge common byte pairs. "th" → one token.

**Causal transformer** — Transformer where each position only sees itself and earlier positions (can't peek at future tokens).

**Concept vector** — 512-dimensional vector representing the compressed meaning of input text. The central representation in GESTALT.

**Cosine similarity** — How similar two vectors are by direction. 1.0 = same, 0.0 = unrelated, -1.0 = opposite.

**Cross-entropy loss** — How surprised the model is by the correct answer. Low = correct, high = wrong.

**d_model** — Dimensionality of the transformer's hidden representations. Currently 512, targeting 1024.

**Dropout** — During training, randomly zero 10% of values to prevent over-reliance on any single neuron.

**Early stopping** — Stop training when validation loss stops improving. Prevents overfitting.

**FIFO** — First In, First Out. Oldest item removed when container is full.

**FSM (Finite State Machine)** — Fixed states with transition rules. Used to constrain plan generation.

**GELU** — Smooth activation function in the MLP. Lets some negative values through.

**Logits** — Raw unnormalized output scores before softmax converts them to probabilities.

**Mean pooling** — Average all non-padding positions to produce one vector. Used by ConceptEncoder.

**MHA (Multi-Head Attention)** — Multiple attention mechanisms in parallel, each focusing on different things.

**RMSNorm** — Normalizes vectors to unit energy. Volume knob for the signal.

**RoPE** — Rotary Position Embedding. Encodes position by rotating Q and K vectors.

**Safetensors** — Safe file format for neural network weights. No arbitrary code execution.

**SFT (Supervised Fine-Tuning)** — Training on (input, correct output) pairs.

**Top-K** — Only consider the K most probable next tokens during generation.

**Top-P (nucleus sampling)** — Only consider tokens whose cumulative probability reaches P.

**VRAM** — GPU memory. All weights, gradients, and computations must fit here during training.

---

*Built with Rust, candle, and an unreasonable amount of determination.*
*GESTALT WIRED-V5 — February 2026*

*"The encoder was dead for twelve runs. We just didn't know how to listen."*
