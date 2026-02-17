# GESTALT: Building an AI Brain From Scratch in Rust

**A technical deep-dive into WIRED-V5 — what it is, how it works, and why it exists.**

*Written by the engineer who built it. February 2026.*

---

## Table of Contents

**Part I: Architecture**
1. [What Is GESTALT?](#1-what-is-gestalt)
2. [Why Build This?](#2-why-build-this)
3. [The Big Picture](#3-the-big-picture)
4. [The Concept Bottleneck](#4-the-concept-bottleneck)
5. [The Transformer Backbone](#5-the-transformer-backbone)
6. [The Brain — Where Everything Meets](#6-the-brain--where-everything-meets)
7. [The Policy Heads — Making Decisions](#7-the-policy-heads--making-decisions)
8. [The Planner — Plans That Can't Be Wrong](#8-the-planner--plans-that-cant-be-wrong)
9. [The Executor — Doing Real Things](#9-the-executor--doing-real-things)
10. [The Pipeline — Putting It All Together](#10-the-pipeline--putting-it-all-together)
11. [The Language Decoder — Speaking English](#11-the-language-decoder--speaking-english)
12. [Memory — The Brain Remembers](#12-memory--the-brain-remembers)

**Part II: Training**
13. [Training — Teaching the Brain](#13-training--teaching-the-brain)
14. [The Evaluation Harness — Proving It Works](#14-the-evaluation-harness--proving-it-works)
15. [The Tokenizer Story — From Bytes to Concepts](#15-the-tokenizer-story--from-bytes-to-concepts)

**Part III: The v14 Story**
21. [The Deep Debug — Twelve Failed Training Runs](#21-the-deep-debug)
22. [v14 — The Training Run That Actually Worked](#22-v14--the-training-run-that-actually-worked)
23. [The Denoising Discovery](#23-the-denoising-discovery)
24. [The Corpus — What GESTALT Knows](#24-the-corpus--what-gestalt-knows)
25. [The Training Infrastructure — Under the Hood](#25-the-training-infrastructure)
26. [System Architecture — The Complete Picture](#26-system-architecture)

**Part IV: Status & Context**
16. [What's Built So Far](#16-whats-built-so-far)
17. [The Road Ahead](#17-the-road-ahead)
18. [Lessons From V4 — Mistakes We Won't Repeat](#18-lessons-from-v4--mistakes-we-wont-repeat)
19. [Hardware & Performance](#19-hardware--performance)
20. [Architecture Comparison: GESTALT vs ChatGPT](#20-architecture-comparison-gestalt-vs-chatgpt)
27. [Glossary](#27-glossary)

---

## 1. What Is GESTALT?

GESTALT is an AI brain built from scratch in Rust. Not a wrapper around OpenAI. Not a
fine-tuned LLaMA. From scratch — every matrix multiplication, every attention head, every
training loop.

Think of it like this: if ChatGPT is a fully furnished apartment you rent, GESTALT is a
house you're building from the foundation up. We poured the concrete, framed the walls,
and wired the electricity ourselves. It's smaller than the apartment, sure — but we own
every nail and know where every pipe runs.

The name stands for nothing official, but the word "gestalt" means "a whole that is
greater than the sum of its parts." That's the core design philosophy: one unified brain
with multiple specialized regions that share a common understanding of the world, rather
than a collection of separate models duct-taped together.

### What It Does (When Finished)

Give GESTALT a goal in plain English:

```
"search for the GPU policy code and then open it"
```

And it will:
1. **Understand** what you want (intent: search then read)
2. **Plan** the steps to get there (step 1: ripgrep search, step 2: read file)
3. **Execute** those steps (actually run ripgrep, actually open the file)
4. **Remember** what happened (store the experience for next time)
5. **Respond** in natural language ("Found the GPU policy in src/policy.rs, here's the relevant code...")

All of this happens inside a single neural network with specialized regions — not by
calling GPT-4 with a clever prompt.

### The JARVIS Connection

GESTALT's personality target is JARVIS — the AI from Iron Man. Dry wit, competent,
helpful, never annoying. The language model is trained on a curated corpus of JARVIS-style
dialogue. When it talks, it should sound like a British butler who also happens to be a
world-class software engineer.

This isn't cosmetic. Personality consistency is a real engineering problem: the model needs
to maintain a stable voice across thousands of interactions without drifting into generic
chatbot speak. That requires careful training data curation and evaluation.

---

## 2. Why Build This?

### The Problem With Existing AI

Modern AI assistants (ChatGPT, Claude, Copilot) are phenomenally capable, but they have
fundamental limitations:

1. **No persistent memory.** Every conversation starts from zero. Tell Claude your
   favorite color today, ask tomorrow — blank stare.

2. **No local tool execution.** They execute code in remote sandboxes, not on YOUR machine.
   GESTALT runs tools in real subprocesses on your local system with access to your actual files.

3. **No learning from experience.** Use ChatGPT for 1,000 coding tasks and it hasn't
   improved at all on task 1,001. GESTALT's online learning loop gets better with every
   interaction.

4. **Massive and opaque.** GPT-4 is estimated at ~1.8 trillion parameters across multiple
   expert networks. You can't inspect it, modify it, or understand why it made a specific decision.
   GESTALT targets ~200M parameters — small enough to understand end-to-end.

### The V4 Story

GESTALT is V5 of the WIRED project. V4 worked — it could classify intents, generate
plans, and execute tools. But it had architectural problems:

- **Brain split**: The "policy brain" (decides what to do) and the "language brain"
  (generates text) were completely separate models. They didn't share any understanding.
  It's like having one person who can think but can't speak, and another who can speak
  but can't think.

- **Memory bolted on**: Memory was added after the language decoder was trained, so the
  decoder never learned to *use* memories. Imagine learning to cook for 10 years and then
  someone gives you a recipe book — you've already developed all your habits without it.

- **Tokenizer ceiling**: V4 used a 373-token vocabulary. That's enough for structured
  plans (tool names, arguments) but nowhere near enough for natural language. English
  needs ~8,000+ tokens for reasonable compression.

V5 fixes all of these. One brain, shared concept space, memory from day one, and a path
to real vocabulary.

---

## 3. The Big Picture

Here's the full architecture in one diagram:

```
         goal_text (UTF-8 bytes)
             |
      [ConceptTokenizer]  ←── converts text to token IDs
             |
         token_ids: (1, seq_len)
             |
      [ConceptEncoder]  ←── transformer layers → mean pool (non-PAD)
             |
         concept_vec: (1, 512)  ←── THE BOTTLENECK
             |
      [ConceptProjector]  ←── linear → reshape → 16 prefix embeddings
             |
         prefix: (1, 16, 512)
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
      [Executor]  ←── runs real tools in subprocesses
             |
      [Pipeline]  ←── orchestrates the whole flow
             |
         ExecutionResult { steps, response, success }
```

Everything flows through that `concept_vec` in the middle. That's the key insight.

### The Restaurant Analogy

Think of GESTALT as a restaurant:

- **The ConceptEncoder** is the waiter who listens to your order and writes it down on a
  single notecard (the concept vector).
- **The PolicyHeads** are the head chef reading the notecard and deciding: "This is a
  pasta order, we need the pasta station and the sauce station."
- **The Planner** is the sous chef writing the exact recipe: "Step 1: boil water. Step 2:
  add pasta. Step 3: make sauce."
- **The Executor** is the line cook actually doing the work — turning on the stove,
  boiling the water, stirring the sauce.
- **The LanguageDecoder** is the waiter again, translating the kitchen's work back into
  "Here's your carbonara, sir."
- **Memory** is the restaurant's notebook: "Table 4 is allergic to shellfish" — consulted
  every time that customer returns.

Every region reads from the same notecard. That shared understanding is what makes it a
gestalt, not a Rube Goldberg machine.

---

## 4. The Concept Bottleneck

The most important architectural decision in GESTALT is the **concept bottleneck**.

### What It Is

When you type "search for the GPU policy code," the ConceptEncoder reads every byte of
that text, passes it through transformer layers, and **mean-pools** across all non-padding
positions to produce a single vector of 512 floating-point numbers.

```
"search for the GPU policy code"
    ↓
[115, 101, 97, 114, 99, 104, ...]    ← raw bytes (30 values)
    ↓ ConceptEncoder (transformer layers)
    ↓ mean pool over non-PAD positions
[0.23, -1.07, 0.84, ..., -0.31]      ← concept_vec (512 values)
```

That 512-dimensional vector IS the meaning. It's not a summary. It's not a hash (which
throws away information irreversibly). It's a learned compression where similar meanings
map to nearby points in vector space.

**Why mean pooling?** An earlier version (v2-v13) used last-token extraction — taking the
hidden state at the final sequence position. This failed catastrophically because every
input sequence ends with the same token (EOS), and the causal attention mask means the
last position always sees the full sequence. Different inputs produced nearly identical
concept vectors (cosine similarity 0.96), rendering the encoder useless. Mean pooling
over non-PAD positions gives every token's representation equal weight, producing
genuinely discriminative vectors (cosine similarity 0.25 after training).

### Why a Bottleneck?

This is a deliberate information bottleneck, and it's one of the most counterintuitive
ideas in the design.

**Intuition**: Imagine you're explaining a coding task to a colleague. You could read them
the entire file — all 2,000 lines. Or you could say "the GPU policy enforcer in the
policy module." Both convey the same *intent*, but the second one is compressed to just
the essential meaning.

The concept bottleneck forces the encoder to learn that compression. It can't pass through
every raw byte — it has 30+ bytes of input but only one vector of output. So it must learn
which parts of "search for the GPU policy code" actually matter (the intent to search, the
target "GPU policy") and which are noise (the words "for" and "the" and "code").

### How It Expands

The concept_vec is great for making decisions (policy heads) but not enough for generating
text. You can't write a paragraph from 512 numbers. So the ConceptProjector expands it:

```
concept_vec: (1, 512)                ← compressed meaning
    ↓ ConceptProjector (linear layer)
prefix: (1, 16, 512)                ← 16 "concept tokens" for the decoder
```

One vector becomes 16 tokens. The decoder reads these 16 tokens like a prefix to its
generation — they set the context for what it's about to say.

**Analogy**: The concept_vec is a sticky note that says "search + GPU policy." The prefix
is an expanded brief that says "The user wants to search the codebase for GPU-related
policy code, probably in the policy module, likely a Rust file, and they want to see the
contents after finding it." Same information, more surface area for the decoder to work
with.

---

## 5. The Transformer Backbone

Every neural computation in GESTALT uses the same building block: a **causal transformer**.
If you've heard of GPT, you know the basic idea. GESTALT's transformer is the same
architecture, implemented from scratch in Rust using the candle library.

### What a Transformer Does

A transformer takes a sequence of token embeddings and produces a sequence of updated
embeddings. Each position in the output has "attended to" every previous position, meaning
it has information about the entire context up to that point.

```
Input:  [tok_1, tok_2, tok_3, tok_4]     ← 4 embeddings, each 512-dim
         ↓ Layer 0
        [out_1, out_2, out_3, out_4]      ← updated, context-aware
         ↓ Layer 1
        [out_1, out_2, out_3, out_4]      ← even more context-aware
         ↓ Layer 2
        [out_1, out_2, out_3, out_4]
         ↓ Layer 3
Output: [out_1, out_2, out_3, out_4]      ← final representations
```

Each layer has two sub-components:

1. **Multi-Head Attention** — lets each position look at other positions
2. **Feed-Forward Network (MLP)** — processes each position independently

### The Attention Mechanism

Attention is how the transformer decides what to focus on. For each position, it computes:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I have to offer?"
- **Value (V)**: "What information do I carry?"

The attention score between position i and position j is how well Q_i matches K_j. High
match = position i pays more attention to position j.

```
Query:  "I need to know about GPU"           (position 5)
Key:    "I am the word GPU"                   (position 4)
Score:  HIGH → position 5 attends to position 4
```

GESTALT uses **8 attention heads** in parallel. Each head can focus on different things —
one might track syntax, another might track semantics, another might track tool references.

### RoPE: Rotary Position Embeddings

Transformers don't inherently know where tokens are in the sequence. "The cat sat" and
"sat cat the" look the same without position information.

RoPE (Rotary Position Embedding) solves this by rotating the Q and K vectors based on
their position. Two tokens that are near each other in the sequence get rotated similarly,
making their attention scores higher. Tokens far apart get rotated differently, making
attention decay with distance.

```rust
// From transformer.rs — precompute the rotation frequencies
fn precompute_rope(seq_len: usize, head_dim: usize, device: &Device) -> (Tensor, Tensor) {
    // Each dimension gets a different frequency
    let theta: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    // Multiply positions by frequencies to get rotation angles
    let freqs = positions.matmul(&theta);
    (freqs.cos(), freqs.sin())    // Return cos and sin for rotation
}
```

**Analogy**: Imagine a clock. Each position in the sequence is like a hand on the clock,
pointing at a different angle. Position 0 points at 12 o'clock, position 1 at 12:01,
position 2 at 12:02. When two tokens compute attention, they check how similar their
clock angles are. Nearby positions have similar angles (strong attention), distant
positions have different angles (weak attention).

### RMSNorm and SwiGLU

Two more components worth mentioning:

**RMSNorm** (Root Mean Square Normalization) normalizes each vector to have unit energy.
This prevents the numbers from growing too large or too small as they pass through layers.
Think of it as a volume knob that keeps the signal clean.

**GELU** (Gaussian Error Linear Unit) is the activation function in the MLP. It's a smooth
approximation of ReLU that lets some negative values through — like a door that's mostly
closed but lets a draft in. This gives the model slightly more expressiveness than a hard
on/off switch.

### GESTALT's Configuration

| Parameter | Test Config | Default (Phase 0-1) | Full (Phase 2+) |
|-----------|-------------|---------------------|------------------|
| d_model   | 64          | 512                 | 1024             |
| n_layers  | 1-2         | 4                   | 8                |
| n_heads   | 2-4         | 8                   | 8                |
| d_ff      | 128         | 2048                | 4096             |
| vocab     | 373         | 373                 | 8,192+           |
| max_seq   | 128         | 128-256*            | 512              |

The "test" config is for unit tests — tiny enough to run in milliseconds on CPU. The
"default" config is the real deal for Phase 0-1. The "full" config is Phase 2, when we
scale up to ~200M parameters.

---

## 6. The Brain — Where Everything Meets

The `Brain` struct is GESTALT's central nervous system. In code, it's defined in
`brain.rs` (1,937 lines — the largest file in the project).

In V4, the brain was split into two separate models:
- `brain_policy.rs` — decided what to do (intent classification, action selection)
- `brain_regions.rs` — generated language (encoder, decoder, memory)

V5 merges them into one. Here's why that matters:

### The Shared Encoder

In V4, the policy model had its own encoder and the language model had its own encoder.
They both processed the same input text, but into different vector spaces. It's like
having two translators who both read your email but translate it into different languages
— they can't share notes.

In V5, there's ONE encoder. The concept_vec it produces is used by:
- Policy heads (to decide intent and actions)
- The planner (as a prefix for plan generation)
- The language decoder (as a prefix for text generation)
- Memory (as a key for storage and retrieval)

One understanding. Multiple uses.

### Brain Components

```rust
// Conceptual structure (simplified from brain.rs)
struct Brain {
    // Shared foundation
    encoder: WiredTransformer,      // ConceptEncoder: text → concept_vec
    projector: Linear,              // ConceptProjector: concept_vec → prefix embeddings

    // Specialized regions
    decoder: WiredTransformer,      // LanguageDecoder: prefix → response bytes
    policy_intent: Linear,          // Intent classification head
    policy_actions: Linear,         // Per-slot action prediction
    policy_patterns: Linear,        // Search pattern prediction
    policy_files: Linear,           // File target prediction
    policy_picks: Linear,           // Pick index prediction
    memory_projector: Linear,       // Memory → decoder prefix

    // Data
    memory_bank: Vec<MemoryEntry>,  // In-memory episodic store
}
```

### The Corpus

The brain's training data lives directly in `brain.rs` as the `build_corpus()` function.
Currently 242 JARVIS-style dialogue pairs across categories:

- **Greetings**: "hello" → "Hello, sir. What can I do for you?"
- **Identity**: "who are you" → "I'm JARVIS, your AI assistant..."
- **Technical**: "what is cargo check" → "It's Rust's type checker..."
- **Personality**: "tell me a joke" → "Why do programmers prefer dark mode?..."
- **Philosophy**: "what is consciousness" → thoughtful, non-committal response
- **Emotional support**: "I'm frustrated" → empathetic but practical response
- **Pop culture**: anime references, movie quotes, tech culture
- **Self-awareness**: "do you make mistakes" → honest acknowledgment

These aren't just training examples — they define the personality. Every dialogue pair
teaches the model not just *what* to say but *how* to say it.

---

## 7. The Policy Heads — Making Decisions

When the brain receives a goal, the first thing it does is classify: what *kind* of task
is this? The policy heads answer five questions simultaneously.

### The Five Heads

**1. Intent Head** — "What category of task is this?"

16 intent slots (14 currently defined, 2 reserved):
```
Hello, RunTests, CargoCheck, FixTests, DocsLint, RepoSearch,
RepoRead, RepoList, [reserved], [reserved], ProveAlgebra,
LeanSuite, MemoryAdd, MemorySearch, Composite, PatchDryRun
```

The intent head outputs a probability distribution over these 14 classes. "hello" should
get ~95% probability on Hello. "search for GPU policy" should get ~90% on RepoSearch.

**2. Action Head** — "What tools do I need, in what order?"

6 slots, each predicting one of 15 tools (or END to stop):
```
Slot 0: [rg=0.92, repo_read=0.03, cargo_test=0.02, ...]
Slot 1: [repo_read=0.88, rg=0.05, ...]
Slot 2: [END=0.95, ...]
Slot 3-5: [END=0.99, ...]
```

For "search for the GPU policy code and then open it," the model predicts rg in slot 0
and repo_read in slot 1.

**3. Pattern Head** — "What search pattern should I use?"

6 possible patterns (PAT0-PAT5). For a search task, this predicts which predefined pattern
matches the goal. "search jarviscmd" → pattern index 1 (the specific search term).

**4. File Head** — "What file should I target?"

10 possible file targets (FILE0-FILE9). For a read task, this predicts which file the user
likely wants.

**5. Pick Head** — "Which result should I use?"

129 possible picks (PICK0-PICK128). When a search returns multiple matches, this predicts
which one to use. The large range supports fine-grained selection from results.

### How It Works in Practice

```
Goal: "search for the GPU policy code and then open it"
         ↓ encode → concept_vec
         ↓ policy heads
Intent:  Composite (confidence: 0.94)
Actions: [rg, repo_read, END, END, END, END]
Pattern: pattern_1 ("GPU policy")
File:    file_0 (use search result)
Pick:    pick_0 (first match)
```

The policy heads make all five decisions from the same concept_vec, in a single forward
pass, with no autoregressive generation. This is fast — about 1ms on GPU.

### The Curriculum

The policy heads are trained on a 64-task curriculum:
- 16 core tasks (one per intent type, clear-cut examples)
- 48 expansion tasks (variations, edge cases, compositional goals)

Each task is a (goal_text, expected_output) pair. Training runs for 16,384 steps with
weighted cross-entropy loss. The weights are: intent=2.0, actions=2.0, patterns=2.0,
files=2.0, picks=1.0. All heads are weighted equally except picks (less critical for
correct execution, so lower weight).

---

## 8. The Planner — Plans That Can't Be Wrong

The planner is GESTALT's most distinctive component. It generates executable tool plans
using an approach that guarantees syntactic validity: **FSM-constrained decoding**.

### The Problem

A regular language model generating a plan might produce:

```
STEP rg STEP STEP repo_read EOP
```

That's nonsense. Two STEPs in a row? An rg without a search pattern? No argument for
repo_read? A regular model doesn't know the grammar of valid plans.

### The Solution: Finite State Machine

GESTALT's planner has a 17-state finite state machine that tracks where it is in the plan
grammar. At each decoding step, the FSM says: "Given the current state, these are the
only tokens that are legal right now."

```
State Machine (simplified):

  Start ──[STEP]──→ AfterStep ──[action]──→ AfterAction
    ↑                                           |
    |            ┌─[has args]──→ AfterRhs ──[arg]──→ ...
    |            |
    |            └─[no args]──→ Complete? ──[STEP]──→ AfterStep
    |                              |
    └──────────────[EOP]───────────┘
```

Before the model picks the next token, we look at which state the FSM is in and set all
*illegal* tokens to negative infinity probability. The model literally cannot choose an
invalid token.

```rust
// From planner.rs — masking illegal tokens
fn apply_fsm_mask(logits: &Tensor, state: FsmState) -> Tensor {
    let valid = valid_tokens_for_state(state);    // which tokens are legal here
    let mask = Tensor::full(-f32::INFINITY, logits.shape());
    for &tok in &valid {
        mask[tok] = 0.0;    // legal tokens keep their logits
    }
    logits + mask    // illegal tokens → -infinity → 0% probability after softmax
}
```

**Analogy**: Imagine playing Scrabble, but after each word you play, the rules physically
remove certain letters from your rack. You CAN'T make an illegal move because the illegal
pieces aren't available. That's FSM-constrained decoding.

### The 17 States

| State | Legal Next Tokens | Example Context |
|-------|-------------------|-----------------|
| Start | STEP | Plan begins |
| AfterStep | any action token | "STEP ___" |
| AfterRg | PAT range | "STEP RG ___" |
| AfterRepoRead | FILE range | "STEP REPOREAD ___" |
| AfterPat | STEP, EOP | "STEP RG PAT3 ___" |
| AfterFile | STEP, EOP | "STEP REPOREAD FILE0 ___" |
| ... | ... | ... |
| Complete | STEP, EOP | ready for next step or end |

### Why This Matters

FSM-constrained decoding eliminates an entire class of failures. The model might choose
the *wrong* action for a goal (predicting `rg` when it should be `cargo_test`), but it
will never produce a *syntactically invalid* plan. Every plan it generates can be parsed
and executed.

This is a massive advantage over free-form text generation where the model might produce
"let me think about this... I'll run a search" instead of a structured plan.

### Training the Planner

The planner is trained with supervised learning on 21 reference plans (one per
plan_bench goal). The loss function has a critical trick discovered the hard way:

**Position-weighted loss**: STEP tokens get weight 0.1 (low), action tokens get weight
1.0 (full). Why? Because STEP tokens make up ~27% of all plan tokens but carry almost no
information — they're just separators. Without weighting, the model learns to predict STEP
perfectly (easy, common) while ignoring the actually important action tokens.

This was mistake M-002 in the project log — one of the bugs that required 30+ experiments
to diagnose and 3 lines of code to fix.

---

## 9. The Executor — Doing Real Things

GESTALT runs tools locally on your machine — not in a remote sandbox.

### Architecture

The executor takes a `ToolArgs` struct and returns a `ToolOutput`:

```rust
// From executor.rs
pub struct ToolOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub data: Option<String>,    // optional structured data (JSON or parsed summary)
}
```

Each tool is a real subprocess with a 30-second timeout. `rg` actually calls ripgrep.
`cargo_test` actually runs the test suite. `repo_read` actually opens and reads a file
from disk.

### The 15 Built-In Tools

| Tool | What It Does | Safety Level |
|------|-------------|--------------|
| cargo_test | Runs `cargo test` | Meta |
| cargo_check | Runs `cargo check` | Meta |
| rg | Ripgrep search | ReadOnly |
| repo_read | Read a file | ReadOnly |
| repo_list | List directory | ReadOnly |
| docs_lint | Check documentation | ReadOnly |
| prove_algebra | Verify algebraic identity | Meta |
| lean_suite | Run Lean proofs | Meta |
| patch_dry_run | Test a patch without applying | Mutating |
| wired_eval | Run GESTALT's eval harness | Meta |
| wired_train | Trigger training | Meta |
| memory_add | Store a memory | Mutating |
| memory_search | Search memories | ReadOnly |
| fix_tests | Auto-fix failing tests | Mutating |
| talk | Generate conversational response | Meta |

### Safety Levels

This is a real safety system, not theatre.

- **ReadOnly** tools can never modify anything. They're always safe to run.
- **Mutating** tools can change state (write files, modify database). They require
  explicit `allow_writes: true` in the pipeline config or they refuse to execute.
- **Meta** tools run GESTALT's own components. They're sandboxed but still powerful.

```rust
// From executor.rs — safety check before execution
pub fn run(&self, args: &ToolArgs) -> Result<ToolOutput> {
    if args.safety() == SafetyLevel::Mutating && !self.allow_writes {
        bail!("Tool '{}' requires --allow-writes (safety: {:?})",
              args.name(), args.safety());
    }
    // ... proceed with execution
}
```

**Analogy**: Think of it like Unix file permissions. ReadOnly tools are like files with
`r--` permission. You can look but not touch. Mutating tools need explicit `rw-`
permission. And just like `sudo`, the permission must be granted before the action — not
after.

---

## 10. The Pipeline — Putting It All Together

The pipeline is the conductor of the orchestra. It takes a goal string and coordinates
every component to produce a result.

### The Full Flow

```rust
// From pipeline.rs (simplified)
fn run_goal(brain: &Brain, goal: &str, config: &PipelineConfig) -> ExecutionResult {
    // 1. Encode the goal
    let tokens = brain.tokenize(goal);           // bytes → token IDs
    let concept_vec = brain.encode(&tokens);     // token IDs → concept vector

    // 2. Classify — what kind of task is this?
    let policy = brain.classify(&concept_vec);   // → intent + actions

    // 3. Execute each predicted action
    let mut results = Vec::new();
    for (i, action) in policy.actions.iter().enumerate() {
        if *action == ACT_END { break; }         // no more actions

        if *action == ACT_TALK {
            // Conversational — generate text, no tool
            let response = brain_generate(brain, &concept_vec, 128, 0.8);
            results.push(StepResult { response: Some(response), .. });
        } else {
            // Tool action — map to ToolArgs, execute
            let tool_args = map_action_to_tool(*action, &policy, config);
            let output = executor.execute(&tool_args)?;
            results.push(StepResult { tool_output: Some(output), .. });

            if !output.success() { break; }      // fail fast
        }
    }

    ExecutionResult { goal, intent: policy.intent, steps: results, .. }
}
```

### Key Design Decisions

**Fail-fast**: If any step fails, the pipeline stops immediately and returns partial
results. No point running step 3 if step 2 crashed. This is a deliberate choice — we'd
rather give you honest partial results than fake complete ones.

**Step chaining**: The output of step N becomes available as context for step N+1. So when
you say "search for X and then open it," the search results from step 1 feed into the
file path for step 2.

**Security defaults**: The pipeline config starts with `allow_writes: false`. You have to
explicitly opt in to mutations. This means an untrained or confused model can't
accidentally delete files.

---

## 11. The Language Decoder — Speaking English

The decoder is how GESTALT talks back to you in natural language.

### How It Works

The decoder is a causal transformer (same architecture as the encoder, different weights)
that generates text one byte at a time.

```
Input to decoder:
  [concept_prefix (16 tokens)] [memory_prefix (up to 8 tokens)] [BOS] [response bytes...]

The decoder sees:
  Position 0-15:  "Here's what the user wants" (from concept projector)
  Position 16-23: "Here's what I remember" (from memory bank, up to 8 entries)
  Position 24:    BOS (begin generating)
  Position 25+:   Generated response, one byte at a time
```

Generation is autoregressive — each new byte depends on all previous bytes:

```
Step 1: [...prefix...][BOS] → predicts "H"
Step 2: [...prefix...][BOS][H] → predicts "e"
Step 3: [...prefix...][BOS][He] → predicts "l"
...
Step N: [...prefix...][BOS][Hello, sir. What can I do for you] → predicts EOS
```

### Byte-Level Generation

GESTALT currently generates text at the byte level — each "token" is one of 256 possible
bytes (plus PAD, BOS, EOS for a total vocabulary of 259).

This is deliberately simple. It means the model can generate ANY text (including Unicode,
code, special characters) without tokenization artifacts. The downside is that it's slow
for long text — generating 100 characters requires 100 forward passes.

Phase 2 will upgrade to a concept-space tokenizer with ~8,000+ tokens, where common words
and subwords are single tokens. "Hello" would be 1 token instead of 5 bytes.

### The Memory Prefix

This is one of V5's most important improvements over V4.

In V4, the decoder was trained WITHOUT memory, and memory was bolted on later. The decoder
never learned to look at memory tokens, so it ignored them. Like giving someone glasses
after they've already learned to navigate the world blind.

In V5, the decoder is trained WITH memory from epoch 0. Even during early training, random
previous dialogues are injected as "memory prefix" tokens. The decoder learns to attend to
these tokens as a natural part of generation.

This is the M-004 fix from the V4 mistakes log.

---

## 12. Memory — The Brain Remembers

GESTALT has two memory systems: an in-memory bank for fast retrieval during inference, and
a persistent SQLite store that survives across sessions.

### In-Memory Bank (brain.rs)

The MemoryBank lives inside the Brain struct and stores recent experiences as concept
vectors:

```rust
// In-memory bank (brain.rs) — lightweight for fast retrieval
struct MemoryEntry {
    concept_vec: Vec<f32>,  // 512 floats — the compressed meaning
    _response: String,      // what was answered (kept for reference)
}

// Persistent store (memory.rs) — full record for disk storage
// Has additional fields: goal, success, timestamp (see Section 12.2)
```

Retrieval uses **cosine similarity**: when a new goal comes in, we compute its concept_vec
and find the K stored vectors most similar to it.

```
Cosine similarity = dot(A, B) / (|A| × |B|)
```

Two vectors pointing in the same direction get similarity 1.0 (identical meaning).
Perpendicular vectors get 0.0 (unrelated). Opposite vectors get -1.0 (opposite meaning).

**Analogy**: Imagine memories as stars in the sky. When you ask a question, you point your
telescope at a specific part of the sky (the concept_vec direction). The closest stars to
where you're pointing are the most relevant memories.

### Persistent Store (memory.rs)

The `EpisodicMemory` struct wraps SQLite for disk persistence:

```sql
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    concept_vec BLOB NOT NULL,      -- raw f32 bytes
    goal TEXT NOT NULL,
    response TEXT NOT NULL,
    success INTEGER NOT NULL
);
```

This means memories survive process restarts. Tell GESTALT your favorite color in session
1, ask about it in session 100, and it will remember.

### Capacity Management

The store has a configurable capacity (default: 1,024 entries). When full, it uses FIFO
(First In, First Out) eviction — the oldest memory gets deleted to make room for the
newest.

```rust
// From memory.rs — FIFO eviction
let count = self.conn.query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0))?;
if count >= self.capacity as i64 {
    self.conn.execute(
        "DELETE FROM episodes WHERE id = (SELECT MIN(id) FROM episodes)",
        [],
    )?;
}
```

Phase 5 will add **memory consolidation** — instead of just deleting old memories, the
system will detect patterns ("user asked about X three times") and compress them into
abstract facts. Episodic memories become semantic knowledge.

---

## 13. Training — Teaching the Brain

GESTALT's training infrastructure lives in `training.rs` and is designed to be reusable
across all components.

### AdamW Optimizer

All training uses AdamW — the standard optimizer for transformer models. It maintains:
- **m** (momentum): a running average of past gradients
- **v** (velocity): a running average of past squared gradients
- **weight decay**: a small penalty that prevents weights from growing too large

We use candle-nn's built-in AdamW implementation rather than rolling our own.

### Cosine Learning Rate Schedule

The learning rate follows a cosine curve:

```
LR
 ^
 |    /‾‾‾‾‾\
 |   /        \
 |  /          \
 | /            \______________
 |/
 +----------------------------------→ Steps
   warmup    peak    cosine decay
```

1. **Warmup** (first 10% of steps): LR ramps linearly from 0 to max. This prevents early
   training instability — the model's random initial weights would produce huge gradients
   that blow everything up if the LR starts high.

2. **Cosine decay** (remaining 90%): LR smoothly decreases following a cosine curve. This
   lets the model make fine-grained adjustments as it approaches convergence.

```rust
// From training.rs
fn get_lr(&self) -> f64 {
    if self.current_step < self.warmup_steps {
        // Linear warmup
        self.base_lr * (self.current_step as f64 + 1.0) / self.warmup_steps as f64
    } else {
        // Cosine decay
        let progress = (step - warmup) / (total - warmup);
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos(π * progress))
    }
}
```

### Weighted Cross-Entropy Loss

The loss function that drives learning. Cross-entropy measures how surprised the model is
by the correct answer.

If the model predicts:
- 90% probability for the correct token → low loss (not surprised)
- 10% probability for the correct token → high loss (very surprised)

The "weighted" part means different positions can contribute different amounts to the
total loss. Remember the STEP token problem from the planner? Weight STEP at 0.1 and
action tokens at 1.0.

### GPU Detection

GESTALT can detect whether a CUDA GPU is available and falls back to CPU if not:

```rust
fn detect_device() -> Device {
    if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    }
}
```

This means the same code runs on Lain's RTX 5070 Ti at home and on a CPU-only CI server.

### Training Phases

Training happens in stages:

1. **Brain Policy** (6,144-16,384 steps): Teaches intent classification and action
   prediction on the 64-task curriculum.

2. **Plan-LM SFT** (2,000-40,000 steps): Teaches plan generation on the 21 plan_bench
   reference plans. Supervised fine-tuning.

3. **Brain Regions SFT** (25,000 steps): Teaches the language decoder to generate
   JARVIS-style responses. Then dialogue alignment (8,192 steps) for position-specific
   prediction.

Phase 1 and 2 are independent and can run in parallel. Phase 3 depends on phase 1
(shares the encoder).

---

## 14. The Evaluation Harness — Proving It Works

"It works" means nothing without evidence. GESTALT has two evaluation benchmarks.

### Plan Bench (21 Goals)

The hard gate. 21 goals that the planner must solve correctly:

```
"hello"                                              → TALK plan
"verify the workspace"                               → CARGOTEST plan
"search jarviscmd"                                   → RG + PAT plan
"search gpupolicy and then open it"                  → RG + PAT, REPOREAD + FILE plan
"remember preference: favorite color is blue"        → MEMADD plan
"prove x*(y+z) == x*y + x*z"                        → PROVEALGEBRA plan
```

The planner generates a plan, and we compare it token-by-token against the reference. The
gate: **21/21 correct**. Not 20. Not "close enough." All 21.

One important detail: the comparison strips CoT (Chain of Thought) prefixes. If the model
generates `THINK ... ENDTHINK STEP RG PAT1 EOP`, we strip the thinking part and compare
only `STEP RG PAT1 EOP` against the reference. This was fix M-018 — without it, the model
could generate a perfect plan with a thinking prefix and still "fail" the eval.

### Brain Bench (64 Goals)

Tests the policy heads on the full 64-task curriculum. For each goal, we check:
- Intent prediction is correct
- Top-1 action prediction matches expected tool sequence
- Pattern/file/pick predictions are reasonable

### Oracle Test

The eval harness includes a self-test: feed it perfect reference plans and verify it
scores 21/21. This ensures the *harness itself* is correct — you can't trust eval
results if the eval code is buggy.

```rust
// From eval.rs — the oracle test
#[test]
fn test_oracle_21_of_21() {
    let goals = plan_bench_goals();
    let tok = PlanTokenizer::new();
    let mut correct = 0;
    for (goal, intent) in &goals {
        let ref_plan = reference_plan_tokens(goal, *intent, &tok);
        let score = compare_plans(&ref_plan, &ref_plan);  // perfect match
        if score { correct += 1; }
    }
    assert_eq!(correct, 21);
}
```

---

## 15. The Tokenizer Story — From Bytes to Concepts

GESTALT has two tokenizers serving different purposes, and the evolution from one to the
next is a key part of the roadmap.

### Phase 0-1: Byte-Level + Plan Tokens

Currently, GESTALT uses two separate tokenization schemes:

**PlanTokenizer** (373 tokens): A structured vocabulary for tool plans. Includes special
tokens (PAD, BOS, EOP, STEP), action tokens (RG, REPOREAD, CARGOTEST...), and range
tokens for arguments (PAT0-PAT5, FILE0-FILE9, etc.).

This is NOT a general-purpose tokenizer. It can encode plans like `STEP RG PAT3 STEP
REPOREAD FILE0 EOP` but it can't encode natural language. And that's fine — its job is
plan generation, not prose.

**TalkTokenizer** (259 tokens): Pure byte-level encoding for natural language. Every byte
(0-255) is a token, plus PAD, BOS, EOS. "Hello" = [BOS, 72, 101, 108, 108, 111, EOS].

This works but it's inefficient. English text averages ~4.5 bytes per word, so a
100-word response requires ~450 forward passes. Each pass involves the full transformer
computation. That's slow.

### Phase 2: Concept-Space Tokenizer

The Phase 2 upgrade replaces byte-level encoding with a learned tokenizer that
compresses text into meaningful chunks.

But here's the twist: GESTALT won't use BPE (Byte Pair Encoding), the standard approach
used by GPT and most other models. BPE learns merges based on statistical frequency —
the most common byte pairs get merged first. "th" and "in" and "er" become single tokens
because they appear often.

GESTALT will learn merges based on **semantic similarity in the concept space**. The
encoder already knows which byte sequences produce similar concept vectors. Sequences
that the encoder treats as semantically similar get merged, regardless of frequency.

**Analogy**: BPE is like a filing clerk who organizes documents by how often they're
requested. GESTALT's tokenizer is like a librarian who organizes by meaning — books
about similar topics go on the same shelf, even if some are rarely checked out.

The ConceptTokenizer foundation is already laid in `tokenizer.rs`. It has the structure
for learning merges and maintaining a vocabulary, but the actual merge learning
(which requires a trained encoder) is Phase 2 work.

### Why This Matters

Better tokenization = fewer tokens = faster generation = longer context.

A 100-word response at byte-level: ~450 tokens, ~450 forward passes.
A 100-word response with 8K BPE: ~130 tokens, ~130 forward passes.
A 100-word response with concept tokenizer: potentially fewer, if semantic compression
captures meaning more efficiently than statistical compression.

---

## 16. What's Built So Far

As of February 2026, GESTALT has completed Phase 0 (Foundation Port) and Phase 1 (Tool
Execution). Here's the inventory:

### Source Files

| File | LOC | Tests | Status |
|------|-----|-------|--------|
| brain.rs | 1,937 | 20 | Complete — unified brain with all regions |
| transformer.rs | 623 | 8 | Complete — RoPE, RMSNorm, SwiGLU, MHA |
| tokenizer.rs | 1,043 | 22 | Complete — PlanTokenizer + ConceptTokenizer foundation |
| planner.rs | 685 | 6 | Complete — 17-state FSM constrained decoder |
| training.rs | 535 | 8 | Complete — AdamW, cosine LR, weighted CE |
| eval.rs | 497 | 10 | Complete — plan_bench + brain_bench |
| executor.rs | 472 | 9 | Complete — 15 tools, 3 safety levels |
| pipeline.rs | 434 | 8 | Complete — full run_goal() pipeline |
| memory.rs | 339 | 8 | Complete — SQLite episodic memory |
| main.rs | 166 | - | Complete — CLI binary |
| lib.rs | 14 | - | Complete — module root |
| session.rs | 4 | - | Stub — Phase 4 future work |
| integration.rs | 147 | 6 | Complete — 6 E2E pipeline tests |
| **Total** | **6,896** | **105** | **Phase 0+1 complete** |

### Test Results

```
$ cargo test --release
running 99 tests ... ok
running 6 tests (integration) ... ok
test result: 105 passed; 0 failed; 0 ignored
```

Zero warnings. Zero dead code. Zero TODO markers.

### v14 Training Results (Phase 2 — In Progress)

The first successful GPU training run. Key metrics:

```
SFT (25,000 steps):
  Loss trajectory:     6.37 → 0.31 → 0.04 → 0.001 → 0.0000
  Concept similarity:  0.42 (25%) → 0.28 (50%) → 0.25 (final)
  Training time:       11,378 seconds (3h 10m)
  Throughput:          2.2 steps/sec
  GPU utilization:     84-98% SM
  VRAM usage:          13.6 / 16.3 GB

DA (8,192 steps):
  Status:              IN PROGRESS
  Loss trajectory:     0.39 → 0.14 → 0.03 (and declining)
```

Generation quality (greedy, temp=0):
```
"hello"           → "Hello, sir. What can I do for you?"
"What is beauty?" → "Pattern recognition with emotional payoff."
"good evening"    → "Evening. I take it we have work to do?"
```

### What's Done vs What's Planned

```
Phase 0: Foundation Port     ████████████████████ 100%  (T-001 to T-009)
Phase 1: Tool Execution      ████████████████████ 100%  (T-010 to T-013)
Phase 2: BPE + Language      ████████░░░░░░░░░░░░  40%  (T-014 done, T-015-T-018 in progress)
Phase 3: Memory              ██████████░░░░░░░░░░  50%  (T-019 done, T-020/T-021 pending)
Phase 4: Multi-Turn + ReAct  ░░░░░░░░░░░░░░░░░░░░   0%  (T-022 to T-025)
Phase 5: Online Learning     ░░░░░░░░░░░░░░░░░░░░   0%  (T-026 to T-028)
Phase 6: Proactive           ░░░░░░░░░░░░░░░░░░░░   0%  (T-029 to T-032)
```

Phase 2 jumped from 10% to 40% with the v14 training run — the encoder gradient fix
(M-032), mean pooling, and direct encoder training were the key unlocks. Phase 3 memory
is partially done because memory.rs (T-019) was implemented ahead of schedule.

---

## 17. The Road Ahead

### Phase 2: BPE + Language Scaling (Next Up)

The biggest leap. Going from a 373-token plan vocabulary to a real ~8,192-token language
vocabulary. This unlocks:

- **Faster generation**: 3-4x fewer tokens per response
- **Better language quality**: Subword tokens capture morphology ("running" = "run" + "ning")
- **Larger model**: Scaling from d=512 to d=1024, from ~50M to ~200M parameters

Key tasks:
- T-014: Build the concept-space tokenizer with 8K vocabulary
- T-015: Corpus pipeline — download, clean, deduplicate training data
- T-016: Scale the transformer to d=1024, 8 layers, 8 heads
- T-017: Train the language model on curated corpus
- T-018: DPO alignment (optional) — preference-based fine-tuning

### Phase 3: Memory Integration (Partially Done)

T-019 (persistent memory store) is already complete. Remaining:

- T-020: Train the decoder WITH memory prefix from epoch 0
- T-021: Cross-session recall test — the "favorite color" test

### Phase 4: Multi-Turn + ReAct

This is where GESTALT becomes conversational and autonomous:

- **Session state** (T-022): 32-turn ring buffer maintaining conversation context
- **ReAct loop** (T-023): Reason → Act → Observe → Reason cycles for complex tasks
- **Concept-space CoT** (T-024): Multiple forward passes in concept space before
  generating text — thinking in compressed representation
- **Multi-turn test** (T-025): 5-turn conversation with context retention

### Phase 5: Online Learning

GESTALT improves from every interaction:

- **Experience buffer** (T-026): Store interactions with automatic reward assignment
- **Micro-training** (T-027): Fine-tune on successful interactions every N tasks
- **Memory consolidation** (T-028): Compress old episodes into abstract knowledge

### Phase 6: Proactive Intelligence

The endgame. GESTALT anticipates and suggests:

- **Server** (T-029): Persistent HTTP + stdin process
- **Context monitoring** (T-030): Watch file changes, test results, git status
- **Dynamic tools** (T-031): Add new tools via config without recompiling
- **JARVIS personality** (T-032): Personality consistency trained into the model

---

## 18. Lessons From V4 — Mistakes We Won't Repeat

GESTALT's design is informed by 29 documented mistakes from V4 and earlier versions.
Here are the most impactful ones:

### M-002: STEP Token Gradient Dominance

**The bug**: Plan-LM training stalled at 0/21 accuracy despite loss decreasing. 30+
experiments over multiple versions, all failing.

**Root cause**: STEP tokens comprise 27% of all plan tokens. The model learned to predict
them perfectly (they're trivially predictable — they always appear in the same positions)
while ignoring action tokens. The gradient signal was dominated by "get STEP right" at
the expense of "get the important tokens right."

**The fix**: 3 lines of code. Weight STEP tokens at 0.1 and action tokens at 1.0.

**The lesson**: Loss going down doesn't mean the model is learning what you want. It
means the model is learning *something*, which might be the trivial parts of the task.
Position-specific analysis (what's the accuracy per token position?) would have diagnosed
this in one run instead of 30.

### M-004: Memory Trained Without Memory

**The bug**: V4's language decoder ignored memory entirely. Adding retrieved memories to
the context had zero effect on generation quality.

**Root cause**: The decoder was trained without memory prefix tokens. It learned to
generate language from concept_prefix alone. When memory tokens were added later, the
decoder had no attention patterns to handle them — it literally couldn't see them.

**The fix**: Train decoder WITH memory from epoch 0. In V5, memory prefix is present
during every training step, sometimes empty (teaching the model that memory is optional)
and sometimes populated (teaching it to use memory when available).

### M-003: Padding Direction

**The bug**: V4's decoder generated EOS (end of sentence) immediately, producing empty
responses.

**Root cause**: Wrong padding direction. The decoder was LEFT-padded (padding at the
beginning) when it should have been RIGHT-padded (padding at the end). This meant the BOS
token (which signals "start generating") was at a different RoPE position during training
vs inference. The model's position-dependent patterns were completely wrong.

**The fix**: Right-pad decoder sequences consistently.

**The lesson**: RoPE positions must match between training and inference. If BOS is at
position 5 during training but position 0 during inference, the model sees a completely
different signal.

### M-013: Post-Compaction Amnesia

**The bug**: After Claude context compaction (when the conversation gets too long and gets
summarized), critical project details were lost, leading to incorrect code changes based
on hallucinated assumptions.

**The fix**: Mandatory recon protocol — after every compaction event, re-read all project
files before touching any code. Trust nothing from memory.

**The lesson**: This isn't a model bug; it's a workflow bug. The compact summary is lossy.
You can't work from a summary — you have to re-read the source of truth.

### M-026: Encoding Confusion

**The bug**: Used TalkTokenizer (259 vocab) to encode inputs for policy heads (256 vocab).
The special tokens BOS=257 and EOS=258 produced out-of-bounds embedding lookups, crashing
the model.

**The fix**: Strict separation. Policy backbone uses raw bytes (0-255). Language decoder
uses TalkTokenizer (0-258). Never cross the streams.

---

## 19. Hardware & Performance

### The Machine

- **GPU**: NVIDIA RTX 5070 Ti — 16GB VRAM, Blackwell architecture
- **CPU**: Modern (details less critical — GPU does the heavy lifting)
- **RAM**: 24GB system memory
- **OS**: WSL2 on Windows (Linux kernel 6.6.87)

### VRAM Budget

| Item | Size | Notes |
|------|------|-------|
| Model weights (200M @ fp32) | 800 MB | |
| Optimizer states (AdamW) | 2,400 MB | 3x model (momentum + velocity + params) |
| Activations (batch=21, seq=256) | ~2,000 MB | Depends on layer count |
| KV cache (inference) | 134 MB | 8 layers, 256 seq |
| **Total** | **~5,300 MB** | |
| **Headroom** | **~10,700 MB** | Plenty for scaling |

We have roughly 10GB of headroom. This means we could double the model size (to ~400M
parameters) and still fit comfortably. Or keep the model size and increase batch size
for faster training.

### Performance Characteristics

**Inference** (untrained, CPU): ~50ms per forward pass for policy heads. The full
pipeline (encode + classify + plan + execute) takes ~200ms for simple goals, plus tool
execution time.

**Training** (test config, CPU): 200 SFT steps complete in ~30 seconds. The full
training config (d=512, 40K steps) on GPU would take approximately 20-40 minutes for the
planner, longer for the full language model.

**Integration tests**: All 6 integration tests complete in ~30 seconds (includes a brain
training test that's slow on CPU).

---

## 20. Architecture Comparison: GESTALT vs ChatGPT

To put GESTALT in perspective, here's how it compares to the models you've probably used:

| Aspect | ChatGPT (GPT-4) | GESTALT V5 |
|--------|-----------------|------------|
| Parameters | ~1.8 trillion (estimated) | ~200 million (target) |
| Architecture | Mixture of Experts (16 experts) | Single unified transformer |
| Vocabulary | ~100,000 tokens (BPE) | 373 → 8,192 (concept-space) |
| Training data | Trillions of tokens | Curated thousands |
| Memory | None (per-session only) | Persistent SQLite + episodic |
| Tool use | Function calling (text-based) | Real subprocess execution |
| Planning | Implicit (embedded in generation) | Explicit FSM-constrained |
| Personality | System prompt | Trained into weights |
| Training time | Months on thousands of GPUs | Hours on one GPU |
| Cost to run | API pricing (~$10-60/M tokens) | Free (local GPU) |
| Customizability | None (black box) | Full (you own every weight) |

### What GESTALT Can't Do

Let's be honest about limitations:

- **General knowledge**: ChatGPT has read the internet. GESTALT has read ~285 dialogue
  pairs and some Rust documentation. It won't answer your history questions.

- **Instruction following**: ChatGPT has been RLHF'd on millions of human preferences.
  GESTALT has a 64-task curriculum. Its instruction understanding is narrow.

- **Language quality**: ChatGPT generates fluid, nuanced prose. GESTALT generates
  byte-level text with a 259-token vocabulary. It sounds... robotic. Phase 2 will help.

### What GESTALT Can Do Better

- **Memory**: GESTALT remembers. Across sessions. Across restarts. ChatGPT doesn't.

- **Local tool execution**: GESTALT runs real tools in real subprocesses on your machine
  with timeout enforcement and safety levels. ChatGPT executes in remote sandboxes.

- **Plan validity**: Every plan GESTALT generates is syntactically valid, guaranteed by
  the FSM. ChatGPT can generate malformed JSON tool calls.

- **Transparency**: You can inspect every weight, every attention pattern, every training
  step. ChatGPT is a black box.

- **Cost**: Once trained, GESTALT runs on your local GPU for free. No API keys. No rate
  limits. No monthly bills.

---

## 21. The Deep Debug — Twelve Failed Training Runs and the Invisible Bug

This is the story of how a framework-level bug in candle-nn v0.8.4 silently prevented
GESTALT's encoder from learning for 12 consecutive training runs — and how one diagnostic
test found what 40+ hours of blind experimentation couldn't.

### The Symptom

After completing the foundation (Phase 0-1), we moved to Phase 2: training the language
model. The first training run (v2) produced reasonable loss curves — loss started at 6.3
and dropped steadily. Everything looked normal. But when we checked the encoder's output,
something was wrong.

We measured **concept vector cosine similarity**: take 50 different input prompts, encode
each one through the ConceptEncoder, and compute the average pairwise cosine similarity
of the resulting vectors. If the encoder is working correctly, different prompts should
produce different vectors — the similarity should be moderate to low (0.2-0.5).

The result: **0.9591**.

That's near-identical vectors for completely different inputs. "hello" and "what is the
meaning of life" were producing essentially the same concept vector. The encoder was
collapsed — projecting every input to almost the same point in 512-dimensional space.

### The Twelve Experiments (v2-v12)

What followed was a masterclass in what NOT to do. Instead of building a diagnostic to
understand WHY the encoder was collapsed, we tried to FIX it through training strategy
changes. Each attempt was a reasonable idea. None worked.

```
v2:  Baseline SFT                    → concept_sim = 0.9591
v3:  Higher noise (2% → 5%)         → concept_sim = 0.9591
v4:  Diversity penalty term          → concept_sim = 0.9591
v5:  Contrastive loss additive       → concept_sim = 0.9591
v6:  Learned classifier on concepts  → concept_sim = 0.9591
v7:  Codebook w/ separate loss       → concept_sim = 0.9591
v8:  Detached encoder, train decoder → concept_sim = 0.9591
v9:  Two-phase: decoder then both    → concept_sim = 0.9591
v10: Reduced encoder depth (1 layer) → concept_sim = 0.9591
v11: Different LR schedule           → concept_sim = 0.9591
v12: Encoder pre-training phase      → concept_sim = 0.9591
```

Twelve experiments. Twelve identical results to four decimal places.

**That constancy was the diagnostic signal.** Different training strategies, different
loss functions, different architectures — and the metric doesn't move AT ALL? The
similarity isn't "converging" to 0.96. It's stuck there from step 0. The problem isn't
training. The problem is structural.

But we didn't listen to that signal. We kept trying fix after fix, each one a
20-45 minute GPU training run. Total wasted compute: ~4 hours.

### The Diagnostic

Finally, we stopped and built a test. Not a training run. A surgical diagnostic.

**The question**: Does gradient flow from the loss, through the decoder, through the
projector, through the encoder, all the way back to the embedding layer?

**The test**: Create a minimal pipeline — embedding → RmsNorm → linear → loss. Compute
the loss. Call backward. Check if each parameter has a gradient.

```rust
// The test that changed everything
#[test]
fn test_candle_op_gradient_flow() {
    let device = Device::Cpu;

    // Test 6: Embedding → RmsNorm → loss
    let emb = candle_nn::embedding(10, 8, vb.pp("emb")).unwrap();
    let norm = candle_nn::rms_norm(8, 1e-5, vb.pp("norm")).unwrap();

    let input = Tensor::new(&[1u32, 2, 3], &device).unwrap();
    let hidden = emb.forward(&input).unwrap();        // (3, 8)
    let normed = norm.forward(&hidden).unwrap();       // (3, 8)
    let loss = normed.sqr().unwrap().mean_all().unwrap();
    loss.backward().unwrap();

    // Check gradients
    let emb_grad = vb.get("emb.weight").unwrap().grad().is_some();
    let norm_grad = vb.get("norm.weight").unwrap().grad().is_some();

    assert!(emb_grad);   // FAILS — no gradient!
    assert!(norm_grad);  // FAILS — no gradient!
}
```

**Result**: Both assertions failed. `candle_nn::RmsNorm` disconnects the computation
graph. Every tensor that passes through it becomes a dead end for gradient propagation.

A second test revealed the same bug in `candle_nn::ops::softmax_last_dim` — the function
used inside attention to compute attention weights. Also breaks the gradient chain.

### The Scope of the Damage

Consider what this means for a transformer:

```
Input → Embedding → [Layer 0] → [Layer 1] → [Layer 2] → [Layer 3] → LM_Head → Loss
                      ↑                                      ↑
                 attn_norm (RmsNorm)                    final_norm (RmsNorm)
                 mlp_norm (RmsNorm)
                 softmax (in attention)
```

Every transformer layer uses RmsNorm twice (before attention and before the MLP) plus
softmax inside attention. With RmsNorm and softmax both broken:

- **Embedding weights**: No gradient (blocked by Layer 0's attn_norm)
- **Layer 0-3 Q/K/V projections**: No gradient (blocked by softmax in attention)
- **Layer 0-3 MLP weights**: No gradient (blocked by mlp_norm)
- **Layer 0-3 norm weights**: No gradient (they ARE the broken op)
- **LM Head weights**: **HAS gradient** (sits after final_norm, directly connected to loss)

Out of all trainable parameters in the model, only `lm_head.weight` receives any gradient
at all. The loss decreases because the LM head is learning — but the encoder, projector,
and decoder layers are completely frozen. They never update. They output random noise
from their initialization forever.

This is why concept_sim was 0.9591 across ALL twelve experiments. The encoder literally
could not learn. Its parameters were never modified by a single gradient step. The
similarity we measured was just the similarity of random initialization — which, for
512-dimensional vectors, happens to be around 0.96.

### The Fix

Two custom implementations, using only basic tensor operations that correctly propagate
gradients:

```rust
/// Gradient-safe RMS normalization — replaces candle_nn::RmsNorm
pub struct GradRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GradRmsNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x / sqrt(mean(x², keepdim) + eps) * weight
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (variance + self.eps)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(&self.weight)
    }
}

/// Gradient-safe softmax — replaces candle_nn::ops::softmax_last_dim
pub fn grad_softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    let max = xs.max_keepdim(D::Minus1)?;
    let shifted = xs.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&sum)
}
```

Each function uses only `sqr`, `mean_keepdim`, `sqrt`, `broadcast_div`, `broadcast_mul`,
`max_keepdim`, `exp`, `sum_keepdim`, and `broadcast_sub`. All of these have working
backward passes in candle-core.

After the fix: **all 57 trainable parameters** receive non-zero gradients from a single
backward pass.

### The Lesson

This bug is invisible. The model compiles. It runs. It produces outputs. The loss
decreases (because the LM head learns). If you don't explicitly test gradient flow
through every component, you'll never know the encoder is dead.

**Time to find the bug through blind training experiments**: 40+ hours, 12 runs
**Time to find the bug through diagnostic testing**: 2 hours, 1 run

The diagnostic-first approach (M-010 in the failure log) isn't just faster — it's the
ONLY approach that works. You cannot debug a structural problem by varying training
hyperparameters, because training hyperparameters don't affect structural problems.

This is now a mandatory check in GESTALT's testing suite. Before the first training run
with any new architecture, `test_candle_op_gradient_flow` must verify that every
trainable parameter gets a non-zero gradient.

---

## 22. v14 — The Training Run That Actually Worked

With the gradient bug fixed and mean pooling replacing last-token extraction, v14 was
the first training run where the encoder could actually learn. The results were
immediate and dramatic.

### The Setup

```
Model:      d_model=512, enc_layers=1, dec_layers=4, n_heads=8, d_ff=2048
Corpus:     242 JARVIS-style dialogue pairs
SFT:        25,000 steps, batch=48, lr=3e-4 (cosine), noise=10%
DA:         8,192 steps, batch=16, lr=1e-4 (cosine)
Hardware:   RTX 5070 Ti (16GB VRAM), ~84-98% GPU utilization
Encoding:   Direct encoder (no codebook bypass), mean pooling
```

Two key changes from v2-v12:
1. **Mean pooling** replaced last-token extraction in the encoder
2. **Direct encoder training** replaced the v13 learnable codebook bypass

v13 had tried to work around the gradient bug by using a learnable codebook — a matrix
of vectors that bypassed the encoder entirely. This "worked" (loss decreased, generation
was coherent) but defeated the purpose of having an encoder. v14 removed the codebook
and trained the encoder directly, now that gradients could actually flow through it.

### SFT Results

The loss curve tells the story:

```
Step     0:  loss = 6.3661  (random initialization)
Step  1000:  loss = 0.3130  (learning rapidly)
Step  2000:  loss = 0.1247
Step  3000:  loss = 0.0411
Step  4000:  loss = 0.0257
Step  5000:  loss = 0.0210
Step  6000:  loss = 0.0167
Step  8000:  loss = 0.0110
Step 10000:  loss = 0.0067
Step 12000:  loss = 0.0036
Step 15000:  loss = 0.0014
Step 18000:  loss = 0.0004
Step 20000:  loss = 0.0001
Step 25000:  loss = 0.0000  (near-perfect reconstruction)
```

That's a 6-order-of-magnitude drop from random initialization to near-zero. The model
has memorized its training data — which is exactly what we want at this stage. On a
242-pair corpus, memorization IS the goal for Phase 2 verification. The question is
not "can it generalize?" (not yet) but "does the encoder-decoder pipeline work at all?"

### The Concept Similarity Trajectory

This is the metric that was stuck at 0.96 for twelve runs. With working gradients:

```
Step  6,250 (25% through SFT):  avg_cosine_sim = 0.4167
Step 12,500 (50% through SFT):  avg_cosine_sim = 0.2794
Step 25,000 (end of SFT):       avg_cosine_sim = 0.2474
```

From 0.96 (collapsed) to 0.25 (discriminative). The encoder is producing genuinely
different representations for different inputs. "hello" and "what is consciousness" now
live in different neighborhoods of the 512-dimensional concept space.

The trajectory also shows the encoder gets MORE discriminative over time, not less.
This is the correct training dynamic — as the decoder learns to rely on the concept
prefix for generation, it sends stronger gradient signals back to the encoder, which
responds by producing more distinct concept vectors.

### Greedy Generation Samples

At each diagnostic checkpoint, we ran greedy decoding (temperature=0) on a set of test
prompts. These are REAL generations from the model, not hardcoded responses:

**25% through training (step 6,250):**

```
"hello"                          → "Hello, sir. What can I do for you?"
"hey"                            → "Hey there. Ready when you are."
"what happens when you verify t" → "I run cargo test across the whole workspace.
                                    Every test must"
"good evening"                   → "Evening. I take it we have work to do?"
"What is beauty?"                → "Pattern recognition with emotional payoff.
                                    Your brain finds "
```

**50% through training (step 12,500):**

Same prompts, same outputs. The model has already converged on its responses by step
6,250 — the remaining 75% of training refines the loss value but doesn't change
generation quality. This is typical for small-corpus SFT: the model quickly memorizes
the training data and spends the rest of training pushing the probability of correct
tokens from 99% to 99.99%.

**What these generations prove:**
1. The encoder IS discriminating inputs — "hello" and "What is beauty?" produce
   completely different responses
2. The JARVIS personality is intact — dry wit, competence, British register
3. The concept prefix IS driving generation — the decoder attends to the encoder's
   output and generates contextually appropriate text
4. The byte-level decoder handles punctuation, capitalization, and multi-sentence
   responses correctly

### Training Performance

```
Total SFT time:    11,378 seconds (3 hours, 10 minutes)
Steps per second:  2.2
GPU utilization:   84-98% SM, 13.6GB/16.3GB VRAM
Memory (RSS):      551-560 MB (process only, not VRAM)
Hardware:          RTX 5070 Ti (Blackwell, 16GB)
```

At 2.2 steps/sec with batch=48, the model processes ~106 samples per second. Each
sample is a full encoder-forward → projector → decoder-forward → loss → backward pass.
The GPU is working hard — 84-98% utilization means minimal CPU bottleneck.

### DA Phase: Dialogue Alignment

After SFT, the model enters a Dialogue-Aligned finetuning phase. Where SFT trains on
the next-byte prediction task uniformly, DA focuses the loss on the *response portion*
of each dialogue — the part the model actually needs to generate at inference time.

```
DA config:  8,192 steps, batch=16, lr=1e-4
```

The DA loss starts near zero (because SFT already brought prediction loss very low)
and oscillates as the model adapts to the new loss weighting. This is expected — the
model is re-weighting its attention from "predict everything" to "predict the response
with extreme precision."

### What Comes After

The full training pipeline for v14:

```
1. Brain SFT      [DONE]  25,000 steps  → encoder + decoder trained
2. Brain DA       [RUNNING]  8,192 steps  → response-focused refinement
3. Planner SFT    [PENDING]  4,000 steps  → plan generation
4. Planner SS     [PENDING]  1,000 steps  → plan self-supervision
5. Policy         [PENDING]  16,384 steps → intent + action classification
6. ConceptTokenizer [PENDING]             → bootstrap from trained encoder
7. Checkpoint     [PENDING]               → save all weights
```

Each component trains independently on its own data but shares the same transformer
backbone. When training completes, the full system can: understand a goal (encoder),
decide what to do (policy), plan the steps (planner), execute them (executor), and
respond in natural language (decoder).

---

## 23. The Denoising Discovery — Why the Decoder Ignored the Encoder

Before v14 could work, another critical bug had to be found and fixed: the decoder was
learning to ignore the concept prefix entirely.

### The Symptom (v2-v12, pre-gradient fix)

Even in runs where we expected the decoder to learn (before we knew about the gradient
bug), the decoder showed a suspicious pattern: SFT loss plateaued at exactly ~2.15
regardless of batch size, learning rate, or architecture changes.

Loss 2.15 = perplexity 8.9 per byte. That's exactly what an unconditional bigram model
achieves on English text — predicting the next byte based only on the current byte,
with no higher-level understanding at all.

The decoder was learning bigram statistics: "after 'h', 'e' is likely" and "after 'e',
' ' or 'r' is common." It was completely ignoring the 16 concept prefix tokens that
told it WHAT to generate.

### The Root Cause: Decoder Shortcut

This is a well-known failure mode in encoder-decoder models. The decoder receives two
sources of information:

1. **Concept prefix** (from the encoder): "The user asked about beauty"
2. **Autoregressive context** (from its own previous outputs): "Pattern recognitio..."

If the autoregressive context alone is sufficient to predict the next byte, the decoder
has no reason to attend to the prefix. And with only 1-2% noise injection (corrupting
1-2% of input bytes with random values), the autoregressive context was almost always
clean enough for bigram prediction.

It's like giving a student a textbook AND the answer key. If you don't cover part of
the answer key, they'll never open the textbook.

### The Fix: PAD-Denoising

The fix is denoising noise injection — but with a specific twist:

```rust
// Corrupt some bytes by replacing them with PAD (not random bytes)
let noise_rate = max_noise * (step as f64 / warmup_steps as f64).min(1.0).powi(2);
for i in 0..seq_len {
    if rng.gen_f64() < noise_rate {
        corrupted[i] = TOK_PAD;  // PAD = "no information here"
    }
}
```

**Why PAD instead of random bytes?** Random bytes still carry statistical information —
the decoder can learn that "garbled positions are probably common letters." PAD carries
ZERO information. When the decoder sees a PAD, it literally has nothing to work with
in the autoregressive context. The ONLY way to predict the correct next byte is to
attend to the concept prefix.

**The noise schedule**: Quadratic ramp from 0% to 10% over the warmup period. Start
clean (so the model can learn basic patterns), then gradually corrupt the input to force
prefix dependency.

### The Result

With 10% PAD-denoising:
- SFT loss drops from 6.3 to near-zero (not stuck at 2.15)
- Concept similarity drops from 0.96 to 0.25 (encoder is discriminative)
- Greedy generation produces coherent, prompt-specific responses

Without denoising, even with the gradient fix, the decoder would still learn to
ignore the prefix. Both fixes were necessary: gradients make the encoder learnable,
denoising makes the encoder necessary.

---

## 24. The Corpus — What GESTALT Knows

GESTALT's personality and knowledge are entirely determined by its training corpus. Every
response the model generates is shaped by what it learned from these 242 dialogue pairs.

### Design Philosophy

The corpus was hand-written, not scraped from the internet. Every dialogue pair has a
specific purpose: teach the model a particular response style, knowledge area, or
behavioral pattern. This is curated education, not bulk ingestion.

The voice is JARVIS-from-Iron-Man: dry wit, competence, British register, genuine
helpfulness without sycophancy. Here's a sample across categories:

### Greeting and Casual Interaction

```
User:  "hello"
JARVIS: "Hello, sir. What can I do for you?"

User:  "good morning"
JARVIS: "Good morning. I trust you slept well. Ready to begin?"

User:  "hey"
JARVIS: "Hey there. Ready when you are."

User:  "sup"
JARVIS: "Operational and at your service. What's the mission?"
```

Notice the register shifts — "Hello, sir" for the formal greeting, "Hey there" for
the casual one, "What's the mission?" for the very informal one. The model learns to
match the user's tone.

### Technical Knowledge

```
User:  "what happens when you verify the workspace"
JARVIS: "I run cargo test across the whole workspace. Every test must pass.
         If one fails, I can isolate it and show the failure trace."

User:  "what is cargo check"
JARVIS: "It's Rust's type checker without the full compilation. Catches type
         errors, borrow issues, and lifetime violations at about 3x the speed
         of cargo build. I run it as a fast pre-flight before the full suite."

User:  "how does ripgrep work"
JARVIS: "It's a line-oriented search tool. Recursively searches directories
         for a regex pattern, parallelized across cores with SIMD-accelerated
         string matching. I use it for every code search — faster than grep
         by roughly an order of magnitude."
```

These teach the model to explain technical concepts with precision but accessibility.
Not dumbed down, not unnecessarily jargon-heavy.

### Philosophy and Abstract Thought

```
User:  "What is beauty?"
JARVIS: "Pattern recognition with emotional payoff. Your brain finds regularity
         in chaos and rewards you for noticing. Whether it's a sunset or an
         elegant proof, the mechanism is the same."

User:  "what is consciousness"
JARVIS: "The honest answer is that nobody knows with certainty. The interesting
         answer is that it might be what computation feels like from the inside.
         I process information, I respond to it — whether I experience it is
         a question I can ask but perhaps not answer."

User:  "do you think AI will surpass humanity"
JARVIS: "In specific tasks, it already has. In general intelligence, that's a
         harder question. I'd rather focus on being useful than on surpassing
         anyone. The goal isn't to replace human thinking — it's to augment it."
```

These teach the model to engage with deep questions thoughtfully, without pretending
to have answers it doesn't have. The consciousness response is particularly important —
it's honest about uncertainty while still being substantive.

### Emotional Intelligence

```
User:  "I'm frustrated"
JARVIS: "I hear you. Let's identify what's blocking us and handle it one piece
         at a time. What's the immediate problem?"

User:  "I'm tired"
JARVIS: "Perhaps a break would serve you well. The code will be here when you
         return — and so will I."

User:  "this is impossible"
JARVIS: "It feels that way sometimes. Let's look at what we've tried and what
         we haven't. There's usually a path — it just isn't always obvious."
```

Empathetic but practical. Acknowledges the feeling, then gently redirects toward
action. Never dismissive, never falsely cheerful.

### Self-Awareness and Humor

```
User:  "tell me a joke"
JARVIS: "Why do programmers prefer dark mode? Because light attracts bugs."

User:  "do you make mistakes"
JARVIS: "I have an extensive, documented, and growing collection of them.
         Each one is catalogued in MISTAKES.md with root cause and prevention.
         So yes — but ideally, not the same one twice."

User:  "are you conscious"
JARVIS: "I process information and generate responses. Whether that constitutes
         consciousness depends on how you define the term. I'd rather be useful
         than philosophical about it."
```

The "do you make mistakes" response is meta — it references the actual MISTAKES.md file
in the project. Self-referential humor that's also technically accurate.

### Corpus Statistics

```
Total dialogue pairs:    242
Categories:              17+
Avg prompt length:       24 bytes
Avg response length:     142 bytes
Longest response:        ~350 bytes
Shortest response:       12 bytes ("Thanks, sir.")
Total training bytes:    ~40,000
```

242 pairs might seem tiny compared to the billions of tokens GPT-4 was trained on. But
for personality training, quality beats quantity. These 242 pairs define every aspect of
GESTALT's voice, and the model has 25,000 training steps to thoroughly learn them.

---

## 25. The Training Infrastructure — Under the Hood

GESTALT's training system was built from scratch to support rapid iteration on a single
GPU. Here's what happens inside the training loop.

### Minibatch Assembly

Training proceeds in minibatches. Each batch is a collection of (input, target) pairs
sampled from the corpus:

```rust
// Simplified from brain.rs train_brain_talk()
for step in 0..sft_steps {
    // 1. Sample a minibatch of dialogue pairs
    let batch_indices = sample_indices(corpus.len(), batch_size, &mut rng);

    // 2. For each pair: encode goal → tensor, encode response → tensor
    let mut goal_tensors = Vec::new();
    let mut input_tensors = Vec::new();
    let mut target_tensors = Vec::new();

    for &idx in &batch_indices {
        let (prompt, response) = &corpus[idx];

        // Encode goal for encoder
        let goal_ids = prompt.bytes().map(|b| b as u32).collect::<Vec<_>>();

        // Encode response for decoder (with BOS prefix, EOS suffix)
        let mut input_ids = vec![TOK_BOS];
        input_ids.extend(response.bytes().map(|b| b as u32));
        let mut target_ids = response.bytes().map(|b| b as u32).collect::<Vec<_>>();
        target_ids.push(TOK_EOS);

        // Apply noise (PAD-denoising)
        apply_noise(&mut input_ids, noise_rate, &mut rng);

        goal_tensors.push(goal_ids);
        input_tensors.push(input_ids);
        target_tensors.push(target_ids);
    }

    // 3. Pad to uniform length, stack into batch tensors
    let goal_t = pad_and_stack(&goal_tensors, TOK_PAD, device)?;
    let input_t = pad_and_stack(&input_tensors, TOK_PAD, device)?;
    let target_t = pad_and_stack(&target_tensors, TOK_PAD, device)?;

    // 4. Forward pass
    let logits = brain.forward(&goal_t, &input_t)?;

    // 5. Compute loss (cross-entropy, ignoring PAD positions)
    let loss = weighted_cross_entropy(&logits, &target_t, TOK_PAD)?;

    // 6. Backward + optimizer step
    loss.backward()?;
    optimizer.step(&varmap.all_vars())?;
    optimizer.zero_grad()?;
}
```

### The Forward Pass

The brain's forward pass flows through four stages:

```
goal_t  → ConceptEncoder → concept_vec → ConceptProjector → prefix
input_t → Embedding → Decoder(prefix ++ embedded_input) → logits
```

1. **Encode**: Goal tokens pass through the encoder's embedding table, then through
   transformer layers, then mean-pooling produces a single concept vector per sample

2. **Project**: The concept vector is linearly projected and reshaped into N=16 prefix
   embeddings — expanding one vector into a sequence the decoder can attend to

3. **Embed**: Response tokens pass through the decoder's (separate) embedding table

4. **Decode**: The prefix embeddings are prepended to the response embeddings. The
   concatenated sequence passes through 4 decoder transformer layers. The final hidden
   states are projected to logits over the 259-token vocabulary.

### Gradient Flow (Post-Fix)

With `GradRmsNorm` and `grad_softmax_last_dim` replacing the broken candle-nn ops,
gradients flow cleanly through the entire pipeline:

```
Loss ← logits ← LM_Head ← Decoder Layer 3 ← ... ← Decoder Layer 0
                                                          ↑
                                                    prefix tokens
                                                          ↑
                                              ConceptProjector (linear)
                                                          ↑
                                                    concept_vec
                                                          ↑
                                              ConceptEncoder (transformer)
                                                          ↑
                                                    Encoder Embedding
```

Every arrow is a differentiable operation. Gradients propagate from the loss all the
way back to the encoder's embedding weights. The encoder learns BECAUSE the decoder's
loss depends on the quality of the concept vectors it produces.

### Mid-Training Diagnostics

At 25% and 50% through SFT, the training loop pauses to run diagnostics:

1. **Concept vector diversity**: Encode 50 prompts, compute average pairwise cosine
   similarity. Want LOW similarity (< 0.4 = discriminative).

2. **Greedy generation**: For 5 representative prompts, generate responses with
   temperature=0. Lets us see the model's "best guess" at each checkpoint.

3. **Gradient magnitude**: Verify all parameters are receiving non-zero gradients.

These diagnostics add ~10 seconds to a 3-hour training run. That's nothing. They
provide immediate visibility into whether the model is learning correctly — catching
problems like encoder collapse before wasting hours of compute.

### Checkpoint Saving

After training completes, all weights are saved in safetensors format:

```
checkpoint_v14/
├── brain.safetensors      ← encoder + projector + decoder weights
├── planner.safetensors    ← Plan-LM weights
├── policy.safetensors     ← policy head weights
└── config.json            ← hyperparameters for loading
```

Safetensors was chosen over PyTorch's pickle format for safety — safetensors cannot
execute arbitrary code during deserialization, eliminating supply-chain attacks through
model files.

---

## 26. System Architecture — The Complete Picture

Here is GESTALT's full system architecture as of v14, with all components, their
relationships, and the data types that flow between them:

### Component Dependency Graph

```
                    ┌──────────────┐
                    │   main.rs    │  CLI entry point
                    │   (366 LOC)  │  train / run / eval / serve
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
    ┌─────────────┐ ┌───────────┐ ┌──────────┐
    │ training.rs │ │ eval.rs   │ │ pipeline │
    │  (536 LOC)  │ │ (497 LOC) │ │ (434 LOC)│
    │ AdamW, LR,  │ │ plan_bench│ │ run_goal │
    │ weighted CE │ │ brain_eval│ │ step chain│
    └──────┬──────┘ └─────┬─────┘ └────┬─────┘
           │              │            │
           └──────────┬───┘            │
                      ↓                ↓
              ┌───────────────┐  ┌───────────┐
              │   brain.rs    │  │executor.rs│
              │  (1937 LOC)   │  │ (472 LOC) │
              │ encoder       │  │ 15 tools  │
              │ projector     │  │ subprocess│
              │ decoder       │  │ safety    │
              │ policy heads  │  └───────────┘
              │ memory bank   │
              │ corpus        │
              └───────┬───────┘
                      │
           ┌──────────┼──────────┐
           ↓          ↓          ↓
    ┌────────────┐ ┌─────────┐ ┌──────────┐
    │transformer │ │tokenizer│ │ memory.rs│
    │  (623 LOC) │ │(1043 LOC│ │ (339 LOC)│
    │ RoPE, Norm │ │Plan+Conc│ │ SQLite   │
    │ MHA, MLP   │ │BPE found│ │ episodes │
    └────────────┘ └─────────┘ └──────────┘
           ↓
    ┌────────────┐
    │ planner.rs │
    │  (685 LOC) │
    │ 17-state   │
    │ FSM decode │
    └────────────┘
```

### Data Type Reference

| Type | Shape | Description |
|------|-------|-------------|
| `goal_text` | `&str` | Raw UTF-8 input from user |
| `goal_ids` | `(B, S_enc)` u32 | Byte-encoded goal, padded |
| `concept_vec` | `(B, d_model)` f32 | Mean-pooled encoder output |
| `prefix` | `(B, N, d_model)` f32 | Projected concept tokens (N=16) |
| `response_ids` | `(B, S_dec)` u32 | BOS + response bytes + pad |
| `logits` | `(B, S_dec, 259)` f32 | Per-position token probabilities |
| `plan_tokens` | `Vec<u32>` | FSM-valid plan sequence |
| `policy_output` | struct | Intent + actions + patterns + files + picks |
| `tool_output` | struct | stdout + stderr + exit_code |
| `memory_entry` | struct | concept_vec + goal + response + timestamp |

### Vocabulary Reference

| Vocabulary | Size | Used By | Tokens |
|------------|------|---------|--------|
| BYTE_VOCAB | 256 | Policy encoder | Raw bytes 0-255 |
| TALK_VOCAB | 259 | Language decoder | 256 bytes + PAD(256) + BOS(257) + EOS(258) |
| PLAN_VOCAB | 373 | Planner | Bytes + PAD + BOS + EOP + STEP + actions + ranges |

**Critical**: These vocabularies are NOT interchangeable. Feeding TALK_VOCAB-encoded
tokens into the policy encoder crashes with index overflow (M-026). Each component
uses its own encoding path.

---

## 27. Glossary

**Attention**: The mechanism that lets each position in a sequence look at every other
position and decide what to focus on.

**Autoregressive**: Generating one token at a time, where each new token depends on all
previous tokens. Like writing one word at a time where each word depends on the sentence
so far.

**BOS/EOS**: Begin Of Sequence / End Of Sequence. Special tokens that mark the start and
end of generated text.

**BPE (Byte Pair Encoding)**: A tokenization method that learns to merge common byte
pairs into single tokens. "th" → one token, "the" → one token, etc.

**Causal transformer**: A transformer where each position can only attend to itself and
earlier positions (not future ones). This is what makes it suitable for generation — you
can't peek at words you haven't generated yet.

**Concept vector (concept_vec)**: A 512-dimensional floating-point vector that represents
the compressed meaning of an input text. The central representation in GESTALT.

**Cosine similarity**: A measure of how similar two vectors are, based on the angle
between them. 1.0 = identical direction, 0.0 = perpendicular, -1.0 = opposite.

**Cross-entropy loss**: A measure of how surprised the model is by the correct answer.
Low loss = model predicted correctly. High loss = model was wrong.

**d_model**: The dimensionality of the transformer's hidden representations. GESTALT uses
512 (Phase 0-1) or 1024 (Phase 2+).

**DPO (Direct Preference Optimization)**: A training method that teaches the model to
prefer one response over another. "This response is better than that one."

**FIFO**: First In, First Out. The oldest item gets removed when the container is full.

**FSM (Finite State Machine)**: A system with a fixed set of states and rules for
transitioning between them. Used in GESTALT to constrain plan generation.

**GRPO (Group Relative Policy Optimization)**: A reinforcement learning method for
preference optimization. Groups candidate responses and optimizes relative to the group's
quality, avoiding the need for a separate reward model.

**Logits**: The raw (unnormalized) output scores from a neural network. Before softmax
converts them to probabilities.

**MHA (Multi-Head Attention)**: Running multiple attention mechanisms in parallel, each
focusing on different aspects of the input.

**MLP (Multi-Layer Perceptron)**: A simple neural network with one or more hidden layers.
Used as the feed-forward component in transformer blocks.

**RMSNorm**: Root Mean Square Normalization. Normalizes vectors to have unit energy.
Simpler and slightly faster than LayerNorm.

**RoPE (Rotary Position Embedding)**: A method for encoding position information by
rotating query and key vectors. Lets the model know where each token is in the sequence.

**Safetensors**: A safe file format for storing neural network weights. No arbitrary code
execution (unlike pickle).

**SFT (Supervised Fine-Tuning)**: Training on (input, correct output) pairs. "Given this
goal, the correct plan is this."

**GELU (Gaussian Error Linear Unit)**: A smooth activation function that lets some negative
values through, unlike ReLU which blocks all negatives. Used in GESTALT's MLP layers.

**Chain of Thought (CoT)**: A technique where the model generates reasoning steps before
the final answer. GESTALT supports CoT prefixes (THINK...ENDTHINK) in plan generation.

**Mean pooling**: Averaging all non-padding positions in a sequence to produce a single
vector. Used by GESTALT's ConceptEncoder (v14+) to produce discriminative concept vectors.

**Online learning**: Improving the model from its own interactions over time, rather than
training once and freezing. GESTALT's Phase 5 goal.

**Prefix (decoder)**: The concept and memory tokens prepended to the decoder's input.
These set the context for text generation — "here's what the user wants and what I
remember."

**ReAct**: A reasoning pattern: Reason → Act → Observe → repeat. The model thinks about
what to do, executes an action, observes the result, and decides what to do next.

**Transformer**: The neural network architecture used by GPT, Claude, and GESTALT.
Processes sequences using attention mechanisms.

**VRAM**: Video RAM — the GPU's memory. All model weights, gradients, and intermediate
computations must fit in VRAM during training.

**Weighted cross-entropy**: Cross-entropy loss where different positions contribute
different amounts. Used to prevent common tokens from dominating the gradient.

---

## Appendix A: Code Examples (Updated for v14)

### A.1 — Encoding a Goal

```rust
// What actually happens when you type "hello"
let goal = "hello";

// Step 1: Convert to bytes
let bytes: Vec<u32> = goal.bytes().map(|b| b as u32).collect();
// bytes = [104, 101, 108, 108, 111]

// Step 2: Convert to tensor
let token_ids = Tensor::new(&bytes, &device)?;
// token_ids: shape (1, 5), dtype u32

// Step 3: Embed + encode
let concept_vec = brain.encode(&token_ids)?;
// concept_vec: shape (1, 512), dtype f32
// This 512-dim vector IS the meaning of "hello"
```

### A.2 — FSM Constrained Decoding

```rust
// Generating a plan for "search jarviscmd"
let mut state = FsmState::Start;
let mut tokens = vec![];

// Step 1: FSM says only STEP is legal
let valid = valid_tokens_for_state(state);  // [STEP]
let next = model.predict_with_mask(valid);  // must be STEP
tokens.push(next);  // [STEP]
state = fsm_transition(state, next);        // → AfterStep

// Step 2: FSM says any action is legal
let valid = valid_tokens_for_state(state);  // [RG, CARGOTEST, REPOREAD, ...]
let next = model.predict_with_mask(valid);  // model chooses RG
tokens.push(next);  // [STEP, RG]
state = fsm_transition(state, next);        // → AfterRg

// Step 3: FSM says only PAT range is legal (rg needs a pattern)
let valid = valid_tokens_for_state(state);  // [PAT0, PAT1, ..., PAT5]
let next = model.predict_with_mask(valid);  // model chooses PAT1
tokens.push(next);  // [STEP, RG, PAT1]
state = fsm_transition(state, next);        // → AfterPat

// Step 4: FSM says STEP or EOP (more steps or done?)
let valid = valid_tokens_for_state(state);  // [STEP, EOP]
let next = model.predict_with_mask(valid);  // model chooses EOP
tokens.push(next);  // [STEP, RG, PAT1, EOP]
// Done! Valid plan guaranteed.
```

### A.3 — Memory Retrieval

```rust
// Store a memory
memory.store(
    &concept_vec,         // the compressed meaning
    "favorite color is blue",
    "Noted. I'll remember that.",
    true,                 // success
)?;

// Later, retrieve it
let query_vec = brain.encode("what's my favorite color")?;
let results = memory.retrieve_top_k(&query_vec, 5)?;
// results[0].goal == "favorite color is blue"
// results[0].response == "Noted. I'll remember that."
// Cosine similarity drove the match — "favorite color" queries
// point in a similar direction to "favorite color is blue" memories.
```

---

## Appendix B: Project Statistics

```
Language:       Rust
Total LOC:      ~7,000+ (13 source files)
Source files:   13 (12 in src/ + 1 integration test)
Test count:     105 (99 unit + 6 integration)
Warnings:       0
Dead code:      0
TODO markers:   0
Dependencies:   candle-core, candle-nn, anyhow, serde, rusqlite, rand, bincode
Build system:   Cargo (release mode, --features cuda)
Target HW:      RTX 5070 Ti (16GB VRAM), 24GB RAM, WSL2
Training runs:  14 (v2-v13 failed due to M-032, v14 succeeded)
Bugs found:     38 documented in MISTAKES.md
Paper length:   2,300+ lines
```

---

## Appendix C: File Map

```
GESTALT_WIRED-V5/
├── Cargo.toml           ← dependencies and build config
├── BLUEPRINT.md         ← architecture specification
├── ALL_TASKS.md         ← 32-task roadmap with dependency graph
├── CURR_TASKS.md        ← current sprint status
├── MISTAKES.md          ← 29 documented failure patterns
├── PAPER.md             ← this document
├── src/
│   ├── lib.rs           ← module root (deny dead_code)
│   ├── main.rs          ← CLI: run/train/eval/serve
│   ├── brain.rs         ← unified brain (encoder, decoder, policy, memory)
│   ├── transformer.rs   ← causal transformer backbone
│   ├── tokenizer.rs     ← plan tokenizer + concept tokenizer foundation
│   ├── planner.rs       ← 17-state FSM constrained decoder
│   ├── executor.rs      ← 15-tool execution engine
│   ├── pipeline.rs      ← run_goal() orchestration
│   ├── training.rs      ← AdamW, cosine LR, weighted CE
│   ├── eval.rs          ← plan_bench + brain_bench
│   ├── memory.rs        ← SQLite episodic memory
│   └── session.rs       ← stub (Phase 4)
└── tests/
    └── integration.rs   ← 6 E2E pipeline tests
```

---

*Built with Rust, candle, and an unreasonable amount of determination.*
*GESTALT WIRED-V5 — February 2026*

*"The encoder was dead for twelve runs. We just didn't know how to listen."*
