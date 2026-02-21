# GESTALT: Building an AI Brain From Scratch

**A technical guide to WIRED-V5 — what it is, how it works, and why every decision was made.**

*February 2026.*

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [Why Build Your Own AI?](#2-why-build-your-own-ai)
3. [Why Rust? Why From Scratch?](#3-why-rust-why-from-scratch)
4. [The Big Picture](#4-the-big-picture)
5. [The Concept Bottleneck — The Core Idea](#5-the-concept-bottleneck)
6. [The Transformer — The Building Block](#6-the-transformer)
7. [The Brain — Where Everything Meets](#7-the-brain)
8. [The Policy Heads — Making Decisions](#8-the-policy-heads)
9. [The Planner — Plans That Can't Be Wrong](#9-the-planner)
10. [The Executor — Running Real Tools](#10-the-executor)
11. [The Language Decoder — How It Speaks](#11-the-language-decoder)
12. [The Tokenizer Story — From Bytes to Words](#12-the-tokenizer-story)
13. [The Corpus Pipeline — Feeding the Brain](#13-the-corpus-pipeline)
14. [Memory — The Brain Remembers](#14-memory)
15. [Training — Teaching the Brain](#15-training)
16. [The Training Story — v2 to v23](#16-the-training-story)
17. [The Full Pipeline — A Complete Walkthrough](#17-the-full-pipeline)
18. [The Gallery — How We Measure Quality](#18-the-gallery)
19. [Gradient Accumulation — Scaling Up](#19-gradient-accumulation)
20. [Where It Stands Now](#20-where-it-stands-now)
21. [The Road Ahead](#21-the-road-ahead)
22. [GESTALT vs ChatGPT](#22-gestalt-vs-chatgpt)
23. [Design Philosophy](#23-design-philosophy)
24. [Glossary](#24-glossary)

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

GESTALT's personality target is JARVIS from Iron Man — specifically the MCU version mixed with TARS from Interstellar. Dry wit, competent, helpful, never annoying. The language model is trained on curated JARVIS-style dialogue so it sounds like a British butler who also happens to be a world-class engineer.

This isn't cosmetic — personality consistency is a real engineering problem. When you talk to ChatGPT, it has no persistent voice. Ask it the same question twice and you'll get two different tones, two different personalities. GESTALT trains personality *into the weights*. The JARVIS voice isn't a system prompt that can be jailbroken — it's the only language the model knows how to speak.

The goal isn't to build a chatbot. It's to build a genuine AI assistant that lives on your machine, knows your codebase, remembers your preferences, and speaks with a consistent voice that gets better over time. An AI that earns the name JARVIS.

### The Scale of This Thing

Let's be honest about what we're up against. GPT-4 is estimated at ~1.8 trillion parameters, trained on trillions of tokens across thousands of GPUs for months. GESTALT is ~50 million parameters, trained on 21,000 dialogue pairs on a single GPU in 90 minutes.

That's not a fair fight, and we're not pretending it is.

But here's the thing: GPT-4 forgets you the moment the conversation ends. GESTALT remembers. GPT-4 runs on someone else's server. GESTALT runs on your machine. GPT-4 can't explain why it made a decision. GESTALT lets you inspect every weight, every attention pattern, every gradient.

The question isn't "can this beat ChatGPT?" It's "can this become something ChatGPT fundamentally can't be?"

---

## 2. Why Build Your Own AI?

### What's Wrong With Existing AI

ChatGPT, Claude, and Copilot are phenomenally capable. But they have fundamental limits:

**No persistent memory.** Every conversation starts from zero. Tell Claude your favorite color today, ask tomorrow — blank stare. There are workarounds (conversation history, RAG systems), but they're bolted on. The model itself doesn't remember.

**No local tool execution.** They run code in remote sandboxes, not on YOUR machine. GESTALT runs tools in real subprocesses with access to your actual files, your actual test suite, your actual git repo.

**No learning from experience.** Use ChatGPT for 1,000 tasks and it hasn't improved at all on task 1,001. The weights are frozen. GESTALT's architecture is designed to get better with every interaction through online micro-training.

**Massive and opaque.** GPT-4 is estimated at ~1.8 trillion parameters. You can't inspect it, modify it, or understand why it made a decision. If it gives you a wrong answer, you can't look at the attention patterns and say "ah, it focused on the wrong part of my question." GESTALT targets ~200M parameters — small enough to understand end-to-end.

**No ownership.** Your conversations with ChatGPT are on OpenAI's servers. Your prompts, your code, your company's proprietary information — all flowing through someone else's infrastructure. GESTALT is yours. The model file sits on your disk. The weights are in your VRAM. Nothing leaves your machine unless you explicitly send it.

### What This Project Actually Teaches

Building a neural network from scratch is the best way to understand how AI actually works. Not "a transformer uses attention" — actually understand it. Why does attention need Q, K, and V? What happens when you forget positional encoding? Why does the learning rate need a warmup period?

Every conceptual question has a concrete answer in this codebase:

- "How does a transformer work?" → Read `transformer.rs` (650 lines)
- "How does training work?" → Read `training.rs` (550 lines)
- "How does tokenization work?" → Read `tokenizer.rs` (1,100 lines)
- "What's a concept vector?" → Look at `brain.rs:encode()` (one function call)

No abstraction layers. No framework magic. Just math.

### The V4 Story

GESTALT is V5 of the WIRED project. V4 worked — it could classify intents, generate plans, and execute tools. But it had three architectural problems that V5 fixes:

**Brain split.** V4 had a separate "policy brain" (decides what to do) and "language brain" (generates text). They didn't share any understanding. Like having one person who can think but can't speak, and another who can speak but can't think. V5 merges them into one brain.

**Memory bolted on.** V4's decoder was trained without memory, then memory was added later. The decoder never learned to *use* memories. Like learning to cook for 10 years and then someone gives you a recipe book — you've already developed all your habits without it. When we turned memory on in V4, the decoder actively ignored the memory tokens. They were noise to it. V5 trains with memory from day one.

**Tokenizer ceiling.** V4 used a 373-token vocabulary — enough for structured plans, nowhere near enough for natural language. English has ~170,000 words in common use. Even with byte-level generation (256 tokens for raw bytes), producing coherent text requires the model to learn spelling, grammar, and meaning simultaneously. V5 scales to a concept-space tokenizer with learned merges.

---

## 3. Why Rust? Why From Scratch?

### The Language Choice

Most AI research happens in Python. PyTorch, TensorFlow, JAX — the entire ecosystem is Python-first. So why Rust?

**Memory safety at compile time.** Neural network training involves massive tensor operations, GPU memory management, and complex data pipelines. In Python, a subtle memory leak means your training crashes at hour 47 of an 80-hour run. In Rust, the borrow checker catches these at compile time. If it compiles, the memory management is correct.

**No GIL.** Python's Global Interpreter Lock means true parallelism requires multiprocessing (slow, heavy, error-prone). Rust gives you real threads, real async, real parallelism. This matters when you're orchestrating GPU training, tool execution, memory retrieval, and HTTP serving simultaneously.

**Performance.** The tokenizer bootstrap went from 83 minutes in a naive implementation to 2 seconds after optimization. Not because of algorithmic improvements — because Rust lets you write cache-friendly, zero-allocation code without fighting the language.

**Deployment simplicity.** GESTALT compiles to a single binary. `./gestalt serve` and you have an AI running locally. No `pip install`, no dependency hell, no "works on my machine." One binary, every dependency statically linked.

### The Framework: candle

We use **candle** (by Hugging Face) as the tensor computation library. It's Rust-native, supports CUDA, and gives us autograd (automatic differentiation) without the overhead of PyTorch.

candle is young. This has pros and cons.

**Pros:**
- Lean. No 2GB framework download. No Python interop layer.
- Direct CUDA control. Custom kernels in PTX assembly when needed.
- Deterministic. Same input → same output. No hidden state from previous runs.

**Cons:**
- Bugs. `candle_nn::RmsNorm` has a broken backward pass. So does `softmax_last_dim`. These two bugs cost us 12 training runs and 40+ hours before we found them (see [The Bug Graveyard](#17-the-bug-graveyard)).
- Limited ecosystem. No pre-built data loaders, no distributed training, no mixed precision (yet).
- f32 only for training. No fp16, no bf16, no int8 quantization. This means models that would fit in 4GB on PyTorch take 16GB on candle.

### Why Not Just Use PyTorch?

Because then we'd understand PyTorch, not neural networks.

When you call `torch.nn.TransformerDecoder()`, you get a working decoder. But do you know what happens inside it? Do you know why it needs a causal mask? What happens if you remove RMSNorm from one layer? Why the feed-forward dimension is 4× the model dimension?

Building from scratch means every design decision is conscious. We chose d_ff = 4 × d_model because we read the "Attention Is All You Need" paper and implemented the recommendation. We use GELU instead of ReLU because we tested both and measured the difference. We use 8 attention heads because our experiments showed diminishing returns beyond 8 for our model size.

You can read the paper. Or you can write the code and discover — viscerally — why each choice exists.

---

## 4. The Big Picture

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

- **The ConceptEncoder** is the waiter who listens to your order and writes it down on a single notecard (the concept vector). No matter how long your order is — "I'll have the carbonara, but could you use pancetta instead of guanciale, and can I get a side salad with balsamic but not too much, and actually can you make the pasta al dente" — the waiter distills it to one notecard.

- **The PolicyHeads** are the head chef reading the notecard and deciding: "This is a pasta order — we need the pasta station and the sauce station." Five decisions in parallel, from the same notecard.

- **The Planner** is the sous chef writing the exact recipe: "Step 1: boil water. Step 2: add pasta. Step 3: make sauce." And here's the key — the sous chef can ONLY write valid recipes. They physically can't write "Step 1: boil water. Step 2: Step 3: boil." The grammar rules are baked into their pen.

- **The Executor** is the line cook actually doing the work — turning on the stove, boiling the water. Real flames, real pots, real food. Not a simulation.

- **The LanguageDecoder** is the waiter again, translating the kitchen's work back into "Here's your carbonara, made with pancetta as you requested."

- **Memory** is the restaurant's notebook: "Table 4 is allergic to shellfish. Table 7 always orders extra bread. The Tuesday regular likes their steak rare." Consulted every time that customer returns, even months later.

Every region reads from the same notecard. That shared understanding is what makes it a *gestalt*, not a Rube Goldberg machine.

### Data Flow in Numbers

Here's what actually flows through the pipeline when you type "search for the GPU policy code":

```
Input:    "search for the GPU policy code"               (31 characters)
          ↓ ConceptTokenizer
Tokens:   [115, 101, 97, 114, ...]                       (12 concept tokens)
          ↓ ConceptEncoder (1 encoder layer, 8 heads)
Hidden:   (1, 12, 512) float32                           (24,576 numbers)
          ↓ mean pool across non-PAD positions
Concept:  (1, 512) float32                               (512 numbers — THE bottleneck)
          ↓ ConceptProjector
Prefix:   (1, 16, 512) float32                           (8,192 numbers — 16 "concept tokens")
          ↓ PolicyHeads (5 linear layers, parallel)
Intent:   RepoSearch (confidence: 0.94)                  (16-way classification)
Actions:  [rg, END, PAD, PAD, PAD, PAD]                  (6 tool slots)
          ↓ Planner (FSM-constrained decoder)
Plan:     [STEP, RG, PAT1, EOP]                          (4 plan tokens)
          ↓ Executor (real subprocess)
Output:   "src/policy.rs:42: fn gpu_policy() {"           (ripgrep stdout)
          ↓ LanguageDecoder (4 decoder layers)
Response: "Found the GPU policy in src/policy.rs..."     (generated text)
```

Total computation: ~6ms on GPU. The bottleneck is the tool execution (subprocess spawn + I/O), not the neural network.

---

## 5. The Concept Bottleneck

This is the single most important idea in the architecture.

### What It Is

When you type "search for the GPU policy code," the ConceptEncoder reads every byte, passes it through transformer layers, and **mean-pools** across all non-padding positions to produce a single vector of 512 floating-point numbers.

```
"search for the GPU policy code"
    ↓
[115, 101, 97, 114, 99, 104, ...]    ← raw bytes (or concept tokens)
    ↓ ConceptEncoder (transformer)
    ↓ mean pool non-PAD positions
[0.23, -1.07, 0.84, ..., -0.31]      ← concept_vec (512 numbers)
```

That 512-dimensional vector **IS** the meaning. Not a summary. Not a hash. A learned compression where similar meanings map to nearby points in vector space.

### Why Force a Bottleneck?

Imagine explaining a coding task to a colleague. You could read them the entire 2,000-line file. Or you could say "the GPU policy enforcer in the policy module." Both convey the same *intent*, but the second one strips away everything except what matters.

The bottleneck forces the encoder to learn that compression. It has 30+ bytes of input but only 512 numbers of output. It must learn which parts of "search for the GPU policy code" matter (intent: search, target: GPU policy) and which are noise (the words "for" and "the").

This is different from how GPT works. GPT keeps the FULL sequence of tokens flowing through the model — every word remains as a separate vector throughout all layers. GESTALT compresses everything into 512 numbers, and then everything downstream works from that compression.

Why? Because a concept vector can be:
- **Stored** as a memory key (2,048 bytes on disk)
- **Compared** to other concepts via cosine similarity (one dot product)
- **Retrieved** from a memory bank (K comparisons)
- **Projected** into different formats (prefix for decoder, key for memory, input for policy)

One representation, many uses. That's the gestalt principle.

### How It Expands

512 numbers are great for decisions (the policy heads) but not enough for writing a paragraph. So the ConceptProjector expands:

```
concept_vec: (1, 512)     ← compressed meaning
    ↓ linear layer + reshape
prefix: (1, 16, 512)      ← 16 "concept tokens" for the decoder
```

One vector becomes 16 tokens. The decoder reads these tokens as a prefix — they set the context for what it's about to say.

Think of the concept_vec as a sticky note that says "search + GPU policy." The prefix is an expanded brief that says "The user wants to search the codebase for GPU-related policy code, probably a Rust file, and they want to see the contents." Same information, more surface area for the decoder to attend to.

### Why Mean Pooling?

This is one of our best debugging stories (see [The Training Story](#16-the-training-story) for the full version).

An earlier version (v2-v13) used **last-token extraction** — taking the hidden state at the final sequence position. This failed catastrophically because our sequences were left-padded, meaning every input ends with the same padding/EOS pattern. Different inputs produced nearly identical concept vectors:

```
cosine_similarity("hello", "what is the meaning of life") = 0.9591
```

That's essentially identical. The encoder was "seeing" the padding tokens, not the content. Mean pooling over non-PAD positions gives every content token's representation equal weight:

```
After mean pooling:
cosine_similarity("hello", "what is the meaning of life") = 0.25
```

That's discriminative — the vectors are genuinely different for different meanings.

### Concept Space Visualization

If you could visualize the 512-dimensional concept space (you can't, but humor me), you'd see clusters:

```
                    [technical questions]
                       ↗  ↗  ↗
    [greetings] ← ← ← center → → → [emotional support]
       ↙  ↙                             ↘  ↘
  [tool commands]                   [philosophical queries]
```

"hello" and "good morning" are close together. "search for X" and "find the file" are close together. "what is beauty?" and "what is the meaning of life?" are in the same neighborhood.

The model didn't learn these clusters from labels. It learned them from the training signal: inputs that should produce similar outputs need similar concept vectors. The structure emerges from the training objective.

---

## 6. The Transformer

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
2. **Feed-Forward Network (MLP)** — processes each position independently, adding non-linear transformations

And critically: **residual connections**. The output of each sub-component is *added* to the input, not replaced. This means information can flow through the entire network unmodified if the layers decide not to change it. It's like a highway with optional exits — the information CAN take the exit and get transformed, but it doesn't have to.

### Attention in Plain English

For each position, the transformer computes three things from the same input:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I have to offer?"
- **Value (V)**: "What information do I carry?"

The attention score between positions is how well a Query matches a Key. High match → more attention → more influence on the output.

Here's a concrete example. Consider the sentence "The cat sat on the mat":

```
Position 3 ("on") computes:
  Q: "I'm a preposition, I need to know WHAT is being sat on"

  Checks against every other position:
    K[0] ("The"):  low match — articles don't answer "what?"
    K[1] ("cat"):  medium match — possible object
    K[2] ("sat"):  low match — verb, not an object
    K[4] ("the"):  low match — article
    K[5] ("mat"):  HIGH match — this is the object I'm looking for!

  Result: position 3 attends mostly to position 5
  Its representation becomes: "on" + context_from("mat") → "on [something]"
```

This is simplified — the actual computation is learned, not programmed. But the mechanism is real: each position dynamically selects which other positions to attend to.

### Multi-Head: Parallel Perspectives

GESTALT uses **8 attention heads** in parallel. Each head can focus on something different:

```
Head 0: tracks grammatical relationships ("on" → "mat")
Head 1: tracks subject-verb agreement ("cat" → "sat")
Head 2: tracks entity references ("The" → "cat")
Head 3: tracks semantic similarity (tool names, intent keywords)
Head 4-7: other learned patterns we can't easily name
```

Each head has its own Q, K, V projections. Their outputs are concatenated and projected back to the model dimension. It's like having 8 translators who each focus on a different aspect of the text, then combine their notes.

### Position: RoPE

Transformers don't inherently know where tokens are. "The cat sat" and "sat cat the" look the same without position info. RoPE (Rotary Position Embedding) solves this by rotating the Q and K vectors based on position.

The math is elegant: for each pair of dimensions, apply a rotation proportional to the position:

```
Q_rotated[2i]   = Q[2i] * cos(pos * θᵢ) - Q[2i+1] * sin(pos * θᵢ)
Q_rotated[2i+1] = Q[2i] * sin(pos * θᵢ) + Q[2i+1] * cos(pos * θᵢ)
```

Where θᵢ = 1/10000^(2i/d). Lower dimensions rotate faster (capture local position), higher dimensions rotate slower (capture global position).

The result: when computing attention between position 3 and position 5, the rotations naturally encode that they're 2 positions apart. Nearby tokens get similar rotations (strong attention), distant tokens get different rotations (weaker attention). And unlike absolute position embeddings, RoPE generalizes to sequence lengths the model hasn't seen during training.

### RMSNorm and GELU

**RMSNorm** normalizes each vector to prevent numbers from growing too large or too small through layers. Without normalization, values compound through layers — a value of 1.01 becomes 1.01^32 ≈ 1.37 after 32 operations, while 0.99 becomes 0.99^32 ≈ 0.72. Over a 4-layer transformer with 8 operations per layer, this drift is catastrophic.

RMSNorm is simpler than the more common LayerNorm — it normalizes by root-mean-square without centering:

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) × weight
```

Just a volume knob keeping the signal clean.

**GELU** (Gaussian Error Linear Unit) is the activation function in the MLP. It's a smooth gate that mostly lets positive values through and mostly blocks negative values, with a soft transition around zero:

```
GELU(x) ≈ x × 0.5 × (1 + tanh(sqrt(2/π) × (x + 0.044715x³)))
```

Why not just ReLU (max(0, x))? Because GELU's smooth curve allows small negative values through, giving the model more expressiveness. The difference matters more at small model sizes — GESTALT is small enough that every bit of expressiveness counts.

### Important: candle-nn Bugs

Two critical bugs in candle-nn v0.8.4 broke GESTALT for 12 training runs (see [The Bug Graveyard](#17-the-bug-graveyard) for the full story):

- `candle_nn::RmsNorm` **silently kills the gradient graph**. Tensors passing through it become dead ends for backpropagation. The model compiles, runs, and reports decreasing loss — but only the final layer actually learns.

- `candle_nn::ops::softmax_last_dim` **same problem**. Breaks the gradient chain inside attention.

Both were replaced with custom implementations (`GradRmsNorm` and `grad_softmax_last_dim` in transformer.rs) using only basic tensor ops that correctly propagate gradients. This experience is described in detail in The Training Story.

### GESTALT's Configurations

| Parameter | Test | Default (d=512) | Target (d=1024) |
|-----------|------|-----------------|------------------|
| d_model   | 64   | 512             | 1024             |
| n_layers  | 1-2  | 4               | 8                |
| n_heads   | 2-4  | 8               | 8                |
| d_ff      | 128  | 2048            | 4096             |
| max_seq   | 128  | 256             | 512              |

"Test" is for unit tests — tiny enough to run in milliseconds on CPU. We run 127 tests on every change, so these need to be fast. "Default" is what we train on now. "Target" is where we're heading.

---

## 7. The Brain

The `Brain` struct is GESTALT's central nervous system. It lives in `brain.rs` (~2,000 lines — the largest file in the project, and for good reason).

### Why One Brain?

In V4, two separate models processed the same input into different vector spaces. Like having two translators who both read your email but translate into different languages — they can't share notes. If the policy brain learns that "check the build" means "run cargo test," that knowledge doesn't help the language brain generate a response about test results.

V5 has ONE encoder. The concept_vec it produces feeds:
- Policy heads (deciding intent and actions)
- The planner (as a prefix for plan generation)
- The language decoder (as a prefix for text generation)
- Memory (as a key for storage and retrieval)

One understanding. Multiple uses. When the encoder learns what "check the build" means, EVERY downstream component benefits.

### The Struct

```rust
Brain {
    encoder: WiredTransformer,          // text → concept_vec
    projector: Linear,                  // concept_vec → 16 prefix tokens
    language_decoder: WiredTransformer,  // prefix → response text
    policy_intent: Linear,              // intent classification
    policy_actions: Linear,             // tool sequence prediction
    policy_patterns: Linear,            // search patterns
    policy_files: Linear,               // file targets
    policy_picks: Linear,               // result selection
    memory_projector: Linear,           // memory → decoder prefix
    memory_bank: MemoryBank,            // in-memory episodic store
}
```

Every field here is a neural network component with learnable weights. The total parameter count at d=512 is ~50 million. At d=1024 it'll be ~200 million. Every one of those parameters gets a gradient during training. Every one contributes to the brain's understanding.

### The Forward Pass

When a goal comes in, the Brain processes it in sequence:

```
1. Tokenize: goal → token_ids
2. Embed: token_ids → embeddings (1, seq_len, d_model)
3. Encode: embeddings → hidden states → mean pool → concept_vec (1, d_model)
4. Project: concept_vec → prefix (1, 16, d_model)
5. Branch:
   a. Policy: concept_vec → 5 classification heads (parallel)
   b. Memory: concept_vec → cosine search → top-K → memory prefix
   c. Decode: [prefix; memory_prefix; BOS] → autoregressive generation
```

Steps 5a, 5b, and 5c can run in parallel — they all read from the same concept_vec. In practice, we run them sequentially because they're fast enough and it keeps the code simple.

---

## 8. The Policy Heads

When the brain receives a goal, the first thing it does is classify: what *kind* of task is this? Five linear heads answer five questions from the same concept_vec, in a single forward pass (~1ms on GPU).

### The Five Questions

**Intent Head (16 classes):** "What category of task is this?"

```
Hello        → greeting, needs a friendly response
RunTests     → cargo test, needs tool execution
CargoCheck   → type checking, needs tool execution
RepoSearch   → find something in the code
RepoRead     → read a specific file
Composite    → multi-step task (search THEN read)
Memory       → store or recall information
... and 8 more
```

**Action Head (6 slots × 15 tools + END):** "Which tools, in which order?"

```
"search for the GPU policy code and then open it":
  Slot 0: rg          (search tool)
  Slot 1: repo_read   (file reader)
  Slot 2: END         (no more tools)
  Slot 3-5: PAD       (unused)
```

**Pattern Head:** Which search pattern to use (for ripgrep queries).
**File Head:** Which file to target (from a fixed catalog).
**Pick Head:** Which search result to select (first match, best match, etc.).

All five decisions happen simultaneously from the same concept_vec. One forward pass, five answers. Think of it as five specialists reading the same sticky note and each writing their answer on a different part of the form.

### The Curriculum

The policy is trained on 64 tasks: 16 core (one per intent) + 48 variations (paraphrases, edge cases).

```
Core tasks (one per intent):
  "hello"                              → Hello, talk
  "run the tests"                      → RunTests, cargo_test
  "search jarviscmd"                   → RepoSearch, rg
  "search jarviscmd and then open it"  → Composite, rg + repo_read
  ...

Variations:
  "hi there"                           → Hello (paraphrase)
  "hey"                                → Hello (informal)
  "check for compilation errors"       → CargoCheck (paraphrase)
  "do a type check"                    → CargoCheck (different phrasing)
  ...
```

Training runs for 16,384 steps with weighted cross-entropy. The policy converges quickly — 64 tasks with 256-dimensional embeddings is a very learnable problem. We consistently hit 64/64 accuracy.

---

## 9. The Planner

The planner generates executable tool plans with a guarantee that no other system has: **every plan is syntactically valid**, always.

### The Problem

A regular language model might produce:
```
STEP rg STEP STEP repo_read EOP
```
That's nonsense. Two STEPs in a row? An rg without a search pattern? Regular models don't know the grammar of valid plans. They produce sequences that *look* like plans but can't actually be parsed or executed.

This is a real problem. ChatGPT using function calling regularly produces malformed JSON. Claude sometimes generates tool calls with wrong parameter names. These systems treat tool use as a language problem — generating text that happens to look like a tool invocation. GESTALT treats it as a grammar problem.

### The Solution: FSM-Constrained Decoding

GESTALT's planner has a **17-state finite state machine** tracking where it is in the plan grammar. At each step, the FSM says: "Given the current state, these are the only tokens that are legal right now." All illegal tokens get set to -infinity probability.

```
State Machine (simplified):

  Start ──[STEP]──→ AfterStep ──[action]──→ AfterAction
                                                 |
                    ┌─[has args]──→ AfterRhs ──[arg]──→ ...
                    └─[no args]──→ Complete? ──[STEP or EOP]──→ ...
```

The full state machine has 17 states:

```
 0. Start           → only STEP is legal
 1. AfterStep       → only action tokens (rg, repo_read, etc.) are legal
 2. AfterAction     → depends on which action: some need args, some don't
 3. AfterRhs        → argument value expected (PAT0-PAT5, FILE0-FILE9, etc.)
 4. AfterPat        → STEP or EOP (plan complete or more steps)
 5. AfterFile       → STEP, EOP, or FROM (chain from previous step)
 ...
16. Error           → unreachable if FSM is correct (assertion failure)
```

Think of it like Scrabble where after each word, the rules physically remove certain letters from your rack. You CAN'T make an illegal move because the illegal pieces aren't available.

### Why It Matters

The model might choose the *wrong* action (predicting `rg` when it should be `cargo_test`), but it will never produce a *syntactically invalid* plan. Every plan can be parsed and executed. This eliminates an entire class of failures.

In V3, this was the difference between a system that worked and a system that crashed. 21/21 plans parse correctly — every single time.

### The STEP Token Problem

This is one of our favorite debugging stories (M-002 in the failure log).

STEP tokens make up ~27% of plan tokens but carry almost no information — they're just separators, like commas in a list. Without weighting, the model learns to predict STEP perfectly while ignoring the important action tokens:

```
Before weighting:
  STEP prediction accuracy: 99.8%
  Action prediction accuracy: 0.7%    ← model gives up on these
  Overall loss: looks fine (mostly STEPs are correct)

After weighting (STEP=0.1, action=1.0):
  STEP prediction accuracy: 99.5%    ← barely changed
  Action prediction accuracy: 94.2%   ← now it actually learns
```

Three lines of code. The fix was trivial. Finding it took 30+ experiments because we were looking in the wrong place — tuning learning rates, architectures, training schedules — when the problem was that the loss function was incentivizing the wrong behavior.

---

## 10. The Executor

GESTALT runs tools locally — not in a remote sandbox. This is one of the things that makes it fundamentally different from API-based AI assistants.

### Real Subprocesses

Each of the 15 built-in tools is a real subprocess with a 30-second timeout:

| Tool | What It Does | Safety |
|------|-------------|--------|
| rg | Ripgrep search | ReadOnly |
| repo_read | Read a file | ReadOnly |
| repo_list | List directory | ReadOnly |
| cargo_test | Run tests | Meta |
| cargo_check | Type check | Meta |
| cargo_clippy | Lint check | Meta |
| memory_add | Store memory | Mutating |
| memory_search | Search memories | ReadOnly |
| patch_dry_run | Test a patch | Mutating |
| patch_apply | Apply a patch | Mutating |
| talk | Generate response | Meta |
| ... | 4 more | Various |

### Safety Levels

**Safety levels** are real, not theatre.

```rust
enum SafetyLevel {
    ReadOnly,   // Can't modify anything. Ever.
    Meta,       // Runs analysis but could have side effects (cargo test might write artifacts)
    Mutating,   // Actually changes files. Requires explicit allow_writes: true.
}
```

Like Unix file permissions — you have to grant `rw-` before the action, not after. A Mutating tool called without `allow_writes: true` will refuse to execute. Not warn. Not ask. Refuse.

### Timeout and Isolation

Every tool runs in its own subprocess. If it hangs, the 30-second timeout kills it. The main brain process is never at risk.

```
Brain Process (long-running, holds all neural network weights in VRAM)
    |
    ├── subprocess: rg "jarviscmd" ./src   (timeout: 30s)
    |       → stdout: "src/main.rs:42: fn jarviscmd"
    |       → exit_code: 0
    |
    ├── subprocess: cargo test             (timeout: 30s)
    |       → stdout: "test result: 127 passed; 0 failed"
    |       → exit_code: 0
    |
    └── subprocess: repo_read src/main.rs  (timeout: 30s)
            → stdout: file contents
            → exit_code: 0
```

If `cargo test` takes 45 seconds, it gets killed. The brain logs "tool timed out" and continues with the next step. No hanging, no deadlocks, no infinite loops.

---

## 11. The Language Decoder

The decoder is how GESTALT talks back to you. It's a causal transformer (same architecture as the encoder, different weights) that generates text one token at a time.

### How It Works

The decoder sees a composite input: concept tokens (what the user wants), memory tokens (what the brain remembers), and the response so far:

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

At each step, the decoder produces a probability distribution over all tokens in the vocabulary. "What's the most likely next token?" The sampling strategy (temperature, top-K, top-P) determines how we pick from that distribution.

### From Bytes to Concept Tokens

The decoder originally generated at **byte level** — each token was one of 256 bytes plus PAD, BOS, EOS (259 total). This works for any text (Unicode, code, special characters) but is slow — 100 characters requires 100 forward passes.

Phase 2.5 upgraded to the **ConceptTokenizer** (see [Section 12](#12-the-tokenizer-story)), where common byte sequences are merged into single tokens. "Hello" becomes ~2-3 tokens instead of 5 bytes. Faster generation, better learning.

### The Memory Prefix: V5's Key Innovation

In V4, the decoder was trained WITHOUT memory, then memory was bolted on after training. The decoder never learned to look at memory tokens — it ignored them entirely.

Here's why this fails: neural networks learn to attend to inputs that help predict the correct output. If memory tokens are present during training, the model learns "when the memory says 'user likes blue,' I should respond about blue." But if the model is trained for 25,000 steps WITHOUT memory, it learns to predict responses using ONLY the concept prefix and autoregressive context. Then when you bolt on memory, it's noise — the model has no reason to look at those tokens.

In V5, the decoder trains WITH memory from epoch 0. During training, we randomly sample 0 to K previous dialogues and inject them as "memory prefix" tokens. The model learns that these tokens carry useful information because they correlate with the expected response.

This was bug M-004 in V4. The fix cost zero code complexity — just feed the memory prefix during training, not just during inference.

### Sampling: Temperature, Top-K, Top-P, and Repetition Penalty

Raw generation produces a probability distribution. The sampling strategy determines how we pick from it.

**Temperature (default 0.7):** Scales the logits before softmax. Temperature 0.0 = always pick the most probable token (greedy). Temperature 1.0 = sample proportionally to probability. Temperature 0.7 is a sweet spot — some randomness for variety, but still mostly picking the best options.

```
Before temperature:  [0.5, 0.3, 0.15, 0.05]     (raw probabilities)
After temp=0.5:      [0.65, 0.24, 0.09, 0.02]    (sharper, more confident)
After temp=1.5:      [0.38, 0.29, 0.21, 0.12]    (flatter, more random)
```

**Top-K (default 40):** Only consider the K most probable tokens. Everything below rank K gets masked to zero probability. Prevents low-probability garbage from being sampled.

**Top-P / Nucleus Sampling (default 0.9):** Only consider tokens whose cumulative probability reaches P. This adapts to the model's confidence:

```
Confident prediction (model knows the answer):
  Token probs: [0.85, 0.08, 0.04, 0.02, 0.01, ...]
  At p=0.9: only top 2 tokens considered (0.85 + 0.08 = 0.93 ≥ 0.9)

Uncertain prediction (multiple valid continuations):
  Token probs: [0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, ...]
  At p=0.9: top 12 tokens considered (many options are viable)
```

This is smarter than fixed top-K because it automatically narrows the beam when the model is confident and widens it when uncertain.

**Repetition Penalty (windowed, last 50 tokens):** Penalizes tokens that appeared recently. Without this, the model gets stuck in loops: "the the the the the." The penalty multiplies the probability of repeated tokens by a factor < 1.

The window is important — earlier versions penalized across the entire generation, which meant common words like "the" and "is" got increasingly penalized as the response grew. With a 50-token window, only very recent repetitions are penalized.

---

## 12. The Tokenizer Story

This is one of GESTALT's most important evolution stories — and a key discovery about what makes language models learn effectively.

### Phase 0-1: Byte Level (259 tokens)

Originally, every character was its own token. "Hello" = 5 tokens: [H, e, l, l, o]. Simple, universal, but:

1. **Slow**: 100 characters = 100 forward passes.
2. **Hard to learn**: the model has to figure out that H+e+l+l+o is a greeting. It's learning spelling, grammar, vocabulary, and meaning simultaneously.
3. **Sequence length limited**: with max_seq=256, you can only generate 256 bytes (~256 characters). Long responses get cut off.

But it has one killer advantage: it works for EVERYTHING. Unicode emoji, code with special characters, filenames with weird extensions — all just bytes.

### Phase 2: The ConceptTokenizer (BPE)

The ConceptTokenizer learns **BPE merges** from the training corpus. BPE (Byte Pair Encoding) is the same algorithm used by GPT, but at much smaller scale:

```
Starting vocabulary: 256 bytes + PAD + BOS + EOS = 259 tokens

Merge 1: Find most common byte pair in corpus
         "t" + "h" occurs 847,293 times → new token "th" (ID 259)

Merge 2: " " + "th" occurs 203,117 times → new token " th" (ID 260)

Merge 3: "th" + "e" occurs 191,854 times → new token "the" (ID 261)

...

Merge 200: (some pair) → new token (ID 458)

Final vocabulary: 459 tokens (259 base + 200 learned merges)
```

With 200 merges, "Hello" might become 2-3 tokens instead of 5. Common phrases compress even more. "what can you do" goes from 15 bytes to 8 tokens — nearly 2x compression.

### The Key Discovery: Fewer Merges = Faster Learning

This was a surprise. We expected more merges (bigger vocabulary) to help because each token carries more information. Instead:

```
Grid search at 300 training steps (val_loss @ step 200):
  50 merges  (306 vocab):  val_loss = 2.79   ← best at short training
  100 merges (356 vocab):  val_loss = 3.13
  200 merges (459 vocab):  val_loss = 3.56
  500 merges (759 vocab):  val_loss = 4.08
  1000 merges (1259 vocab): val_loss = 4.70
  2000 merges (2259 vocab): val_loss = 5.53  ← worst
```

**Why?** More tokens = larger embedding table = more parameters to learn from the same data. Each token in the vocabulary gets its own 512-dimensional embedding vector. With 459 tokens, that's 459 × 512 = 235,008 embedding parameters, and each token appears often enough in the training data (21K pairs) to learn a good embedding. With 2,259 tokens, that's 1,156,608 embedding parameters, and rare tokens appear too few times to learn meaningful embeddings.

It's like learning a language. If someone tells you 200 new words with 100 example sentences each, you'll learn them well. If someone tells you 2,000 new words with 10 example sentences each, you'll barely remember any of them.

**Sweet spot: 200 merges** (459 tokens, ~2x compression ratio). Below 150 approaches byte-level. Above 500, the model can't learn enough per token. The curve is flat between 180-220 merges — diminishing returns around 200.

### Grid Search Infrastructure

To find this sweet spot, we built grid search infrastructure with environment variables that override the compiled-in config:

```bash
# Quick 300-step sweep (takes ~2 min per setting)
GESTALT_MERGES=50   GESTALT_SFT_STEPS=300 ./target/release/gestalt train --config default
GESTALT_MERGES=100  GESTALT_SFT_STEPS=300 ./target/release/gestalt train --config default
GESTALT_MERGES=200  GESTALT_SFT_STEPS=300 ./target/release/gestalt train --config default
GESTALT_MERGES=500  GESTALT_SFT_STEPS=300 ./target/release/gestalt train --config default
GESTALT_MERGES=1000 GESTALT_SFT_STEPS=300 ./target/release/gestalt train --config default
GESTALT_MERGES=2000 GESTALT_SFT_STEPS=300 ./target/release/gestalt train --config default
```

This was crucial — being able to sweep hyperparameters WITHOUT recompiling saved hours per experiment.

### Tokenizer Bootstrap: From 83 Minutes to 2 Seconds

The original tokenizer bootstrap was painfully slow. Here's why:

The "concept-aware" scoring tried to measure semantic similarity between byte pairs by running them through the neural encoder:

```
For each candidate merge (byte_a + byte_b):
  For each occurrence in the corpus:
    encoded_a = encoder.forward(byte_a)
    encoded_ab = encoder.forward(byte_a + byte_b)
    score += cosine_similarity(encoded_a, encoded_ab)
```

With 3,000 training pairs and 8.6 million n-gram occurrences, this meant 8.6 million encoder forward passes. At ~1ms each → 83+ minutes.

The discovery: **this was completely pointless**. The context-free encoder produces the same vector for the same bytes regardless of surrounding context. `cosine_similarity(encode("th"), encode("th"))` = 1.0, always. The scoring reduced to: `score = frequency × compression`. Pure BPE, no neural component needed.

New tokenizer bootstrap: 2 seconds. Same merge quality. The "concept-aware" scoring is kept as dead code for when we build a context-dependent encoder (future work).

---

## 13. The Corpus Pipeline

Training data is the fuel. GESTALT's corpus pipeline is a 1,050-line Python script (`scripts/build_corpus.py`) that transforms public datasets into JARVIS-style dialogue pairs.

### The Sources

Three public datasets form the base:

**Dolly (Databricks):** ~15K instruction-following pairs. High quality but formal tone. Good for factual and technical Q&A.

**OASST2 (Open Assistant):** ~10K conversational turns. More natural, more varied, but sometimes too casual or too long.

**Alpaca (Stanford):** ~52K diverse instruction pairs. Generated by GPT-3.5, so there's a distinctive "AI assistant" voice we need to filter.

### The Processing Pipeline

```
Raw data (~77K pairs across 3 sources)
    ↓ Download and parse (JSONL, Parquet)
    ↓ Extract (input, response) pairs
    ↓ Deduplication (exact + near-duplicate detection)
    ↓ Length filtering (too short → useless, too long → training noise)
    ↓ Quality filtering (remove garbled, non-English, code-only)
    ↓ JARVIS voice augmentation (rephrase responses in JARVIS style)
    ↓ Category tagging (greeting, technical, philosophical, etc.)
Result: 21,786 unique dialogue pairs (v19/v22 corpus)
```

### The v23 Expansion: More Isn't Always Better

For v23, we expanded aggressively — adding SlimOrca, UltraChat, and WizardLM datasets. The corpus ballooned from 21K to 92K pairs (4.2x expansion).

The results were instructive — and sobering.

**v22 (21K pairs, 30K steps, val_loss≈1.9):**
```
"good morning"       → "Morning! How'd you sleep?"
"The build broke"    → "Step one: read the error message. The whole thing, not just the last line."
"tell me about yourself" → "I'm a language model with opinions, a dry sense of humor..."
```

**v23 (92K pairs, 11K steps, val_loss≈1.88):**
```
"good morning"       → (long ramble about surfaces and codes)
"The build broke"    → (something about Chicago plants)
"what is a transformer" → "A low-free problem is a constant system..."
```

v23 had BETTER loss (1.88 vs 1.9) but WORSE output. The grammar was almost perfect — no broken tokens, no spelling errors. But the responses had no personality. They sounded like a generic AI assistant, not JARVIS.

**The lesson: corpus quality > corpus quantity.** The 92K corpus was dominated by generic instruction-following data. SlimOrca teaches the model to sound like ChatGPT. UltraChat teaches it to sound like a textbook. WizardLM teaches it to sound like a professor. None of them teach it to sound like JARVIS.

21K pairs with JARVIS voice > 92K pairs with committee voice.

This is now rule M-044 in our failure log: **when scaling data, maintain personality density.** The next expansion will filter much more aggressively — keep only pairs that match JARVIS's voice profile, even if that means a smaller corpus.

### The Gold Corpus: Original JARVIS Pairs

The earliest and most important data is the "gold corpus" — ~242 hand-curated JARVIS dialogue pairs:

```
"hello"
  → "Hello, sir. What can I do for you?"

"What is beauty?"
  → "Pattern recognition with emotional payoff. Your brain rewards you for finding
     structure in chaos — a sunset, a proof, a face. Beauty is the reward signal."

"good evening"
  → "Evening. I take it we have work to do?"

"What do you think about Neon Genesis Evangelion?"
  → "A mecha show that's actually about depression, parental abandonment, and the
     hedgehog's dilemma. Hideaki Anno poured his clinical depression into 26 episodes
     and accidentally created a cultural touchstone."
```

These set the personality target. The rest of the corpus should sound like more of THIS, not less.

---

## 14. Memory

GESTALT has two memory systems: fast in-memory retrieval during inference, and persistent SQLite storage across sessions.

### Why Memory Matters

This is where GESTALT diverges most fundamentally from existing AI assistants. ChatGPT, Claude, Gemini — none of them remember you between sessions. Every conversation starts from zero. The workarounds (conversation history, RAG, user profiles) are all external to the model.

GESTALT's memory is internal. The concept_vec used to store a memory is the same concept_vec used during inference. When you tell GESTALT your favorite color is blue, it stores the concept vector of that conversation. When you ask about your favorite color later, the query concept vector is similar to the stored one — cosine similarity matches them. No external lookup table. No keyword matching. Pure neural similarity.

### In-Memory Bank

The MemoryBank stores recent experiences as concept vectors. When a new goal comes in, we compute its concept_vec and find the K stored vectors most similar to it using **cosine similarity**:

```
cosine_sim = dot(A, B) / (|A| × |B|)
```

Two vectors pointing the same direction = similarity 1.0 (identical meaning). Perpendicular = 0.0 (unrelated). Opposite = -1.0.

Think of memories as stars in the sky. When you ask a question, you point your telescope at the concept_vec direction. The closest stars are the most relevant memories.

### How Memory Flows Through the Decoder

Retrieved memories aren't just metadata. They're projected through a learned `memory_projector` and prepended to the decoder's prefix:

```
Without memory:
  Decoder input: [concept_prefix(16 tokens)][BOS][response...]

With 3 memories:
  Decoder input: [concept_prefix(16 tokens)][mem_0][mem_1][mem_2][BOS][response...]
```

Each memory token is a 512-dimensional vector that the decoder's attention mechanism can attend to. When generating "Your favorite color is blue," the decoder can attend to the memory token from the previous conversation where you said "I like blue."

### Persistent Store (SQLite)

The `EpisodicMemory` wraps SQLite for disk persistence:

```sql
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    concept_vec BLOB,    -- 512 × 4 = 2,048 bytes of raw f32
    goal TEXT,           -- what the user asked
    response TEXT,       -- what GESTALT said
    success INTEGER      -- did the task succeed?
);
```

Tell GESTALT your favorite color in session 1, kill the process, restart it a week later, ask about it in session 100, and it will remember. The concept_vec is stored as raw f32 bytes — no serialization overhead, no precision loss.

### Capacity: 1,024 entries, FIFO eviction

When full, the oldest memory is deleted. This is the simplest eviction policy and works for now. Phase 5 will add **consolidation** — detecting patterns across episodes:

```
Episode 1: "What's my favorite color?" → "Blue" (success)
Episode 17: "Do I like blue?" → "Yes, you mentioned it" (success)
Episode 42: "What colors do I like?" → "Blue, definitely" (success)

Consolidation: abstract fact: "User's favorite color = blue" (stored once, not three times)
```

### Cross-Session Persistence: The Integration Test

This is how we verify memory actually survives process restarts:

```rust
#[test]
fn test_cross_session_memory_persistence() {
    // Session 1: store memories
    let db_path = temp_db();
    {
        let mut mem = EpisodicMemory::open(&db_path).unwrap();
        mem.store("favorite color is blue", concept_vec_blue, "Noted!", true);
        mem.store("I like rust programming", concept_vec_rust, "Good taste.", true);
    }  // EpisodicMemory dropped, connection closed

    // Session 2: retrieve from fresh connection
    {
        let mem = EpisodicMemory::open(&db_path).unwrap();
        let results = mem.retrieve_recent(10);
        assert_eq!(results.len(), 2);
        assert!(results[0].goal.contains("favorite color"));
    }
}
```

All 8 integration tests pass, including this one. Memory persists.

---

## 15. Training

### The Three Training Phases

Training proceeds in stages, each building on the previous:

```
Phase 1: Brain SFT      (50K steps, ~90 min)    Encoder + decoder on dialogue pairs
Phase 2: Planner SFT    (4K-40K steps, ~15 min) Plan generation on reference plans
Phase 3: Policy          (16K steps, ~5 min)     Intent + action classification
                                                 ─────────────
                                           Total: ~110 min on one GPU
```

Brain SFT is the bottleneck — 50,000 steps at 2.2 steps/second on the RTX 5070 Ti. This is where the model learns to generate language.

### The Optimizer: AdamW

Standard transformer optimizer. Maintains two running averages per parameter:

```
m = 0.9 × m + 0.1 × gradient              (momentum: smoothed direction)
v = 0.999 × v + 0.001 × gradient²          (velocity: smoothed magnitude)
update = m / (sqrt(v) + ε)                  (direction / magnitude = normalized step)
param -= lr × (update + wd × param)         (step + weight decay)
```

The momentum means the optimizer keeps going in a consistent direction even when individual gradients are noisy. The velocity means it adapts the step size per parameter — parameters with large, consistent gradients get smaller steps (they're already confident). Parameters with small, sporadic gradients get larger steps (they need more exploration).

Weight decay (wd = 0.01) prevents weights from growing too large. Without it, the model can overfit by memorizing training data with extreme weight values. Weight decay gently pulls all weights toward zero, acting as a regularizer.

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

**Warmup (first 10%):** LR ramps linearly from 0 to max (3e-4). Why? Because at initialization, weights are random. Random weights produce random gradients. Large random gradients with high learning rate = chaos. The warmup period lets the model find a reasonable region of parameter space before taking big steps.

**Cosine decay (remaining 90%):** LR smoothly decreases following a cosine curve:

```
lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × progress))
```

This is better than linear decay because it spends more time near the peak LR (where the model learns fastest) and only slows down near the end (where fine adjustments matter).

### Weighted Cross-Entropy Loss

The loss measures how surprised the model is by the correct answer. For each position in the sequence:

```
loss = -log(P(correct_token))

If P(correct_token) = 0.9  → loss = 0.105  (model is confident, low loss)
If P(correct_token) = 0.01 → loss = 4.605  (model is surprised, high loss)
```

"Weighted" means different positions contribute differently to the total loss:
- **PAD tokens**: weight 0.0 (ignore padding entirely)
- **BOS tokens**: weight 0.0 (the start token is always the same)
- **Content tokens**: weight 1.0 (this is what we're training on)
- **In the planner**: STEP = 0.1, action = 1.0 (focus on the important tokens)

### Early Stopping

Monitors validation loss every N steps. If it hasn't improved for `patience` consecutive checks, training stops and the best checkpoint is kept.

```
Step  1000: val_loss = 3.58  → new best, save checkpoint
Step  2000: val_loss = 2.80  → new best, save checkpoint
Step  3000: val_loss = 2.33  → new best, save checkpoint
Step  4000: val_loss = 2.12  → new best, save checkpoint
Step  5000: val_loss = 2.08  → patience 1/5 (improved but barely)
Step  6000: val_loss = 2.09  → patience 2/5 (got worse!)
Step  7000: val_loss = 2.10  → patience 3/5
Step  8000: val_loss = 2.11  → patience 4/5
Step  9000: val_loss = 2.15  → patience 5/5 → STOP
Best checkpoint: step 4000 (val_loss = 2.12)
```

This prevents overfitting — training too long causes the model to memorize training data while getting worse on new inputs. The best checkpoint from step 4000 generalizes better than the overfit model from step 9000.

### Dropout (0.1): Planned Incompetence

During training, randomly zeroes 10% of values in each layer. This seems counterproductive — why deliberately sabotage your model?

Because it forces redundancy. Without dropout, the model can rely on a few "star" neurons to carry all the information. If those neurons get slightly wrong values, the whole output collapses. With dropout, every neuron must contribute — no single neuron is irreplaceable.

The effect on training life is dramatic:

```
Without dropout: overfits at ~6K steps (v19)
With dropout:    extends to 13K+ steps before overfitting (v20)
```

More useful training steps = more learning = better generalization.

### PAD-Denoising: Forcing the Decoder to Listen

This is one of our most important training tricks, born from frustration (M-029).

Without noise, the decoder can predict the next byte from autoregressive context alone. If the training data says "What is love? → Love is a neurochemical...", the decoder learns the bigram statistics: after "Love is a", the next byte is probably "n." It never needs to look at the concept prefix — the local context is sufficient.

The result: loss plateaus at exactly 2.15 (the bigram baseline). The encoder isn't learning because the decoder doesn't need it.

The fix: replace random bytes with PAD tokens during training. PAD carries ZERO information — it's the absence of signal. When 10% of the response bytes are PAD, the decoder can't rely on local context alone. The only way to predict the missing bytes correctly is to attend to the concept prefix.

```
Without denoising:
  Input:    [prefix] [L][o][v][e][ ][i][s][ ][a][ ][n]...
  Model:    predicts "n" from "[a][ ]" context alone → ignores prefix

With 10% PAD denoising:
  Input:    [prefix] [L][PAD][v][e][ ][PAD][s][ ][PAD][ ][n]...
  Model:    can't predict byte at position 1 from "[L]" alone → must attend to prefix
```

Noise schedule: quadratic ramp from 0% to 10%. Start clean so the model learns basics, then gradually corrupt to force prefix dependency.

---

## 16. The Training Story — v2 to v23

This is the most important section. It chronicles every major training run, what broke, what we learned, and how each failure led to the next breakthrough. Understanding this story is understanding how GESTALT actually works.

### The Invisible Bug: v2-v12 (The Dark Ages)

The first training run (v2) looked fine. Loss started at 6.3 and dropped steadily. But when we checked the encoder's output, something was wrong.

We measured **concept vector similarity**: encode 50 different prompts and compute average pairwise cosine similarity. If the encoder works, different prompts should produce different vectors (similarity 0.2-0.5).

Result: **0.9591**.

"hello" and "what is the meaning of life" were producing essentially the same concept vector. The encoder was collapsed — every input mapped to nearly the same point in concept space.

What followed was twelve experiments trying to fix this through training strategy changes:

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

Twelve experiments. Twelve identical results to four decimal places. ~4 hours of GPU time. Progressively more creative training strategies. Not a single digit of change.

**That constancy was the diagnostic signal.** Different strategies, different losses, different architectures — and the metric doesn't move AT ALL. The similarity isn't "converging" to 0.96. It's stuck there from step 0. The problem isn't training. It's structural.

But we didn't see it. We kept thinking "maybe v6's contrastive loss needs more weight" or "maybe v10 needs fewer layers." Classic case of looking in the wrong room for your keys — the problem was in the framework, not in our code.

### The Diagnostic That Changed Everything

We stopped guessing and built a test. One question: does gradient flow from the loss, through the decoder, back through the encoder?

```rust
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

`candle_nn::RmsNorm` **disconnects the computation graph**. Every tensor that passes through it becomes a dead end for backpropagation. The gradient can't flow backward through the normalization layer.

In a transformer, RmsNorm appears **twice per layer** (before attention and before MLP) plus once at the end (final_norm). So: every gradient that needs to flow through ANY transformer layer gets stopped at the first normalization. The only parameter that receives gradients is `lm_head.weight` — the final projection from hidden states to logits — because it sits AFTER the last RmsNorm, directly connected to the loss.

Same bug in `softmax_last_dim` — used in every attention computation. Even if we fixed RmsNorm, the softmax would block Q and K gradients.

The encoder literally could not learn. Its 0.9591 similarity was just random initialization noise propagated through the same dead computation.

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

After the fix: **all 57 trainable parameters** receive non-zero gradients. The test passes. The encoder can finally learn.

**Time to find through blind experiments: 40+ hours, 12 runs.**
**Time to find through diagnostic testing: 2 hours, 1 test.**

The lesson burned into the project's DNA: **after 2 failed experiments, STOP and build a diagnostic.** One diagnostic run reveals more than 30 blind experiments.

### v14: First Light (242 pairs)

With working gradients + mean pooling, the first real training run:

```
SFT 25,000 steps on 242 JARVIS dialogue pairs:
  Loss:       6.37 → 0.04 → 0.001 → 0.0000 (near-perfect)
  Similarity: 0.96 → 0.42 → 0.28 → 0.25 (discriminative!)
  GPU:        84-98% utilization, 13.6/16 GB VRAM, 2.2 steps/sec
  Time:       3 hours 10 minutes
```

Greedy generation:
```
"hello"            → "Hello, sir. What can I do for you?"
"What is beauty?"  → "Pattern recognition with emotional payoff."
"good evening"     → "Evening. I take it we have work to do?"
```

The encoder was finally learning. Different prompts produced genuinely different concept vectors. The JARVIS personality came through clearly. After 12 failed runs, these outputs felt miraculous.

**But** — this was memorization of 242 pairs. The model had ~3 concept dimensions per training example (512 dims / 242 pairs ≈ 2.1). It could store every example as a unique point in concept space and reproduce it verbatim. Novel prompts that didn't match a training example produced gibberish.

### The DA Disaster (M-039)

After v14's SFT success, we tried a Dialogue-Aligned (DA) fine-tuning phase. DA picks a random position in a response and trains the model to predict just that one byte given the prefix up to that point.

The theory: DA should improve the model's ability to continue from any point in a sentence, not just from the beginning.

The reality: DA destroyed the autoregressive flow. After 8,192 DA steps:

```
Before DA:
  "hello" → "Hello, sir. What can I do for you?"

After DA:
  "hello" → "Rello, sir. What can I do for you?"
```

The first byte was wrong. Longer prompts garbled after ~20 bytes. DA had taught the model to predict isolated bytes but broken the continuous flow that SFT established.

The fix: `da_steps = 0` by default. SFT-only is the proven path. DA needs fundamental rethinking — either much lower learning rate (1e-5 vs 1e-4), full-sequence training instead of isolated positions, or mixed SFT+DA loss.

### v18: Scaling Headaches (1,749 pairs)

We built the corpus pipeline and expanded from 242 to 1,749 pairs. v18 results:

```
val_loss = 2.02 at step 2K, early-stopped at 12K
Gallery: coherent outputs are verbatim training data, novel prompts = gibberish
```

Still memorization, but now with 1,749 memories instead of 242. The model had ~0.3 concept dimensions per example (512/1749) — not quite enough to memorize everything perfectly, so it memorized the most common patterns and garbled the rest.

### v19-v20: The Vocabulary and Dropout Discoveries (21K pairs)

The corpus expanded to 21,786 pairs. Now the real experiments began.

**v19 (merges=2000, no dropout):** Disaster. Best val=3.55 at step 6K. Gallery: 0% coherent. The 2,259-token vocabulary was too large — each token appeared too rarely to learn a good embedding. And without dropout, the model overfit by step 6K and couldn't generalize at all.

**v20 (merges=2000, dropout=0.1):** Better but not good. Best val=3.26 at step 8K. Gallery: ~30% coherent. The dropout extended useful training life from 6K to 13K+ steps. Good openings ("Hmm, let me think...") but sentences dissolved into word salad after 15 tokens.

These two runs established two critical facts:
1. **Vocabulary size matters enormously.** Too many tokens = too little training signal per token.
2. **Dropout is essential for generalization.** Without it, the model memorizes training data and falls off a cliff when tested on novel inputs.

### v22: The Breakthrough (merges=200, 30K steps)

After the grid search revealed that fewer merges help, we ran the definitive Phase 2.5 experiment:

```
Config: merges=200, dropout=0.1, batch=48, 30K steps
Corpus: 21,786 pairs, 459-token vocabulary
Time: ~4 hours on RTX 5070 Ti
```

The results were the best we'd ever seen:

**Gallery highlights — what works (~40-45% coherent openings):**

```
"good morning"
  → "Morning! How'd you sleep?"
    ✓ Perfect. Natural, conversational, in-character.

"The build broke and I don't know why"
  → "Step one: read the error message. The whole thing, not just the last line."
    ✓ Excellent. Practical, specific, personality intact.

"tell me about yourself"
  → "I'm a language model with opinions, a dry sense of humor, and a generator
     conversation."
    ✓ Strong opening. JARVIS voice captured. Degrades after ~15 tokens.

"What do you think about Neon Genesis Evangelion?"
  → "A mecha show that's actually about depression, parental abandonment, and the
     hedgehog's dilemma..."
    ✓ Remarkable. Captured the actual critical analysis of NGE.

"I'm scared of failing"
  → "Good. Fear of failure means you care about the..."
    ✓ Emotionally intelligent opening.

"debug this function"
  → "First, read the error. Then isolate the..."
    ✓ JARVIS debugging methodology.
```

**What breaks — the ~15 token wall:**

After ~15 tokens, outputs degenerate into word salad. The model interpolates between training examples mid-sentence:

```
"what is a transformer"
  → "A novel that's referring to the computer science. We can be seen as a bad linear
     fast and it's insurance to former."

"What is beauty?"
  → "Hmm, let me think... it's when something hits are the bad. It's the decision
     notice, but fear - that's rightly proved that ating with line is just
     optimization..."
```

**The pattern**: First ~10-15 tokens are often correct and stylistically on-point. After that, the model runs out of "memorized opening" and starts interpolating between training examples. Common words and phrases from different dialogues get mixed together into semantic soup.

This is the **capacity bottleneck**. A 512-dimensional model with 21K pairs learns openings — the first few tokens of common response patterns — but doesn't have enough capacity to model full response sequences. It generalizes greeting patterns but not paragraph structure.

### v23: The Corpus Quality Lesson (92K pairs)

Thinking "more data = better model," we expanded to 92K pairs (SlimOrca, UltraChat, WizardLM). The result was humbling.

**v23 at 11K steps, val_loss=1.88 (better than v22's 1.9):**
```
"hello" → "I'm not explaining what you want to do that?"
"good morning" → (long ramble about surfaces and codes)
"what is a transformer" → "A low-free problem is a constant system..."
```

Grammatically improved. Semantically destroyed. The model had learned to generate fluent generic-AI-assistant text instead of JARVIS personality. Lower loss, worse output.

**Key insight: val_loss measures prediction accuracy on the TRAINING distribution, not response quality.** A model that perfectly predicts generic AI responses will have low loss on a corpus full of generic AI text — but it won't sound like JARVIS.

**Corpus quality > corpus quantity.** 21K pairs with personality > 92K pairs with committee voice. The next expansion will be quality-filtered.

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
| v23 (11K) | 92,273 | 459 | 0.1 | 11K | 1.88 | Fluent but personality-flat |

---

## 17. The Full Pipeline — A Complete Walkthrough

Let's trace exactly what happens when you type a goal into GESTALT. Not the theory — the actual tensor operations, function calls, and data transformations. This is how the system works right now, today.

### Step 1: You Type a Goal

```
"search for the GPU policy code and then open it"
```

This is a UTF-8 string. 49 bytes. The pipeline receives it as a `&str` in Rust.

### Step 2: Tokenization

The ConceptTokenizer converts the string into token IDs. With 200 BPE merges (459 tokens total), common byte sequences are merged:

```rust
let tokens = tokenizer.encode("search for the GPU policy code and then open it");
// Result: [342, 115, 101, 97, 114, 99, 104, ...]  (12 tokens instead of 49 bytes)
// Compression ratio: 49/12 = 4.08x
```

Each token ID indexes into a 512-dimensional embedding table. The special tokens PAD (0), BOS (1), and EOS (2) are reserved. The embedding table is learned during training — each token gets a unique 512-dimensional vector that encodes its meaning and relationships.

### Step 3: Encoding

The token embeddings pass through the ConceptEncoder — a 1-layer transformer with 8 attention heads:

```
Token embeddings:     (1, 12, 512)     — 12 tokens, each 512-dim
    ↓ Self-attention (8 heads, RoPE positional encoding)
    ↓ RMSNorm + residual connection
    ↓ Feed-forward MLP (512 → 2048 → 512)
    ↓ RMSNorm + residual connection
Contextualized:       (1, 12, 512)     — 12 tokens, now context-aware
    ↓ Mean pool across non-PAD positions
Concept vector:       (1, 512)          — THE bottleneck
```

That 512-dimensional vector IS the meaning of "search for the GPU policy code and then open it." Similar goals produce similar vectors. Different goals produce different vectors. Cosine similarity between "hello" and "search for code" is ~0.25 — genuinely different representations.

### Step 4: Projection

The ConceptProjector expands the 512-dim vector into 16 prefix tokens for the decoder:

```
Concept vector:       (1, 512)          — compressed meaning
    ↓ Linear layer (512 → 8192)
    ↓ Reshape
Prefix tokens:        (1, 16, 512)      — 16 "concept tokens"
```

These prefix tokens serve as the context for everything downstream — the decoder reads them to understand what it should say, the policy heads read the concept vector to decide what to do.

### Step 5: Policy Classification

Five linear heads classify the goal simultaneously:

```
Concept vector → Intent head    → Composite (idx 10, confidence 0.94)
Concept vector → Action head    → [rg, repo_read, END, PAD, PAD, PAD]
Concept vector → Pattern head   → [PAT1, -, -, -, -, -]
Concept vector → File head      → [-, FILE0, -, -, -, -]
Concept vector → Pick head      → [PICK0, -, -, -, -, -]
```

Five decisions from one forward pass. ~1ms on GPU.

### Step 6: Plan Generation

The FSM-constrained planner generates an executable plan:

```
Input: goal tokens + PLAN_SEP
FSM state: Start → only STEP is legal
    ↓ Decoder predicts STEP (forced)
FSM state: AfterStep → only action tokens legal
    ↓ Decoder predicts RG (model's choice, from distribution)
FSM state: AfterAction(rg) → rg needs a pattern argument
    ↓ Decoder predicts PAT1
FSM state: Complete → STEP or EOP legal
    ↓ Decoder predicts STEP
    ↓ Decoder predicts REPOREAD
    ↓ Decoder predicts FILE0
    ↓ Decoder predicts EOP

Plan: [STEP, RG, PAT1, STEP, REPOREAD, FILE0, EOP]
```

Every token in this plan was chosen by the model's probability distribution, but the FSM ensured that only syntactically legal tokens could be selected. The plan ALWAYS parses. Always.

### Step 7: Plan Compilation

The plan tokens are compiled into executable ToolSteps:

```rust
let steps = compile_plan(&plan_tokens);
// Result:
// Step 0: ToolStep { tool: Rg, args: { pattern: "GPU policy" } }
// Step 1: ToolStep { tool: RepoRead, args: { file: step0.result.first_match } }
```

Step 1 references step 0's output — it will read whatever file ripgrep found. This chaining is automatic.

### Step 8: Tool Execution

Each step spawns a real subprocess:

```
Step 0: Rg
    spawn: rg "GPU policy" ./src --json
    timeout: 30 seconds
    safety: ReadOnly
    stdout: "src/policy.rs:42: fn gpu_policy() {"
    exit_code: 0 ✓

Step 1: RepoRead
    spawn: cat src/policy.rs | head -100
    timeout: 30 seconds
    safety: ReadOnly
    stdout: (file contents, ~2KB)
    exit_code: 0 ✓
```

Real processes. Real files. Real output. Not a simulation.

### Step 9: Memory Storage

The experience is stored for future retrieval:

```rust
brain.memory.store(
    concept_vec,                    // 512 floats → 2,048 bytes
    "search for GPU policy code",   // the goal
    "Found in src/policy.rs:42",    // the response
    true,                           // success
);
```

Next time someone asks about GPU policy, this memory will be retrieved via cosine similarity and injected as prefix context for the decoder.

### Step 10: Response Generation

If the intent is conversational, the decoder generates a response:

```
Decoder input:
  [concept_prefix (16 tokens)][memory_prefix (0-8 tokens)][BOS]

Step 1: predict "F"    (from prefix context: this was a search task)
Step 2: predict "o"
Step 3: predict "un"   (merged token)
Step 4: predict "d"
...
Step N: predict EOS

Response: "Found the GPU policy in src/policy.rs, line 42."
```

The entire pipeline — tokenization, encoding, classification, planning, execution, memory, generation — runs in under 2 seconds. The neural network parts take ~50ms. The subprocess execution takes the rest.

---

## 18. The Gallery — How We Measure Quality

### What the Gallery Is

The gallery is GESTALT's comprehensive evaluation suite. 166 prompts across 15 categories, each designed to probe a different capability:

```
Category          Prompts  Tests
────────────────  ───────  ────────────────────────────────────
Greetings         12       "hello", "good morning", "hey there"
Identity          15       "who are you", "tell me about yourself"
Technical         18       "what is a transformer", "explain TCP"
Philosophy        12       "what is beauty", "meaning of life"
Emotional         14       "I'm scared of failing", "I feel stuck"
Debugging         10       "the build broke", "tests are failing"
Humor             8        "tell me a joke", "what's funny"
SelfReflection    10       "are you conscious", "do you dream"
DeepThought       12       "what is time", "nature of reality"
EngineeringWisdom 8        "best practices", "when to refactor"
Profound          10       "what defines us", "what matters"
PopCulture        10       "thoughts on NGE", "favorite anime"
OOD_Creative      10       Novel prompts unseen in training
OOD_Knowledge     10       Factual questions (out of domain)
Temperature       7        Same prompt at 0.0, 0.3, 0.5, 0.7, 1.0
```

### How Gallery Is Run

```bash
# Full gallery with default config
./target/release/gestalt gallery --config default

# Gallery with specific merge count
GESTALT_MERGES=200 ./target/release/gestalt gallery --config default
```

The gallery loads the best checkpoint (`brain_checkpoint.safetensors`), rebuilds the tokenizer from the training corpus, and generates a response for each of the 166 prompts at temperature 0.7 with top-K=40 and top-P=0.9.

### How We Score

Each response is manually evaluated on four dimensions:

**Coherence** — Does the response form grammatically correct, semantically meaningful sentences? Broken tokens, random words, or incomplete sentences score 0. Fluent English scores 1.

**Relevance** — Does the response actually address the prompt? "hello" should produce a greeting, not a technical explanation. On-topic scores 1.

**Personality** — Does it sound like JARVIS? Dry wit, competence, warmth without being saccharine. Generic "I'm an AI assistant" scores 0. "Evening. I take it we have work to do?" scores 1.

**Sustainability** — How many tokens before the response degrades? Some responses start strong and collapse into word salad after 15 tokens. Full-length coherent responses are the goal.

### Current Results: v22 Hero

The v22 hero checkpoint (merges=200, 30K steps, 21K curated pairs) is our best model:

```
Overall:        ~40-45% coherent openings, ~10-15% fully coherent responses
Greetings:      ~70% coherent (strongest category)
Technical:      ~30% coherent openings, rapid degradation
Philosophy:     ~25% coherent (interesting attempts, then word salad)
Emotional:      ~40% (good empathetic openings)
OOD_Creative:   ~15% (novel prompts are hardest)
OOD_Knowledge:  ~5% (factual questions are out of domain)
```

The pattern is clear: the model has learned response PATTERNS (how to start a greeting, how to open a technical explanation) but not response SUSTENANCE (how to maintain coherent text for a full paragraph). This is the capacity bottleneck — d=512 with 21K pairs learns fragments but not full responses.

### Gallery as Diagnostic Tool

The gallery isn't just an evaluation — it's a diagnostic. By comparing gallery results across training runs, we can see exactly what changed:

```
v19 (merges=2000): 0% coherent → vocabulary too large
v20 (merges=2000): 30% coherent → dropout helps, but merges still too many
v22 (merges=200):  45% coherent → right vocabulary size
v23 (92K corpus):  grammar good, personality gone → corpus quality matters
```

Each gallery run tells a story about what the model learned and what it didn't. The gallery is the ground truth — loss curves can deceive, but generated text doesn't lie.

---

## 19. Gradient Accumulation — Scaling Up

### The Problem: d=1024 OOMs on 16GB VRAM

The current model is d=512 (~50M parameters). The target is d=1024 (~200M parameters). At d=1024, even a batch size of 2 exceeds the 16GB VRAM on the RTX 5070 Ti. candle doesn't support fp16 training, gradient checkpointing, or memory-efficient attention — it's f32 all the way.

```
d=512:  ~5.3GB VRAM at batch=48  ✓ Fits easily
d=1024: ~18GB VRAM at batch=2   ✗ OOM on 16GB card
```

### The Solution: Gradient Accumulation

Instead of computing the gradient from one large batch, compute it from N small batches and average:

```
Without accumulation (batch=32):
  Forward pass on 32 examples
  Backward pass → gradients
  Optimizer step

With accumulation (micro_batch=2, accum_steps=16, effective_batch=32):
  For i in 0..16:
    Forward pass on 2 examples
    Backward pass → gradients for this micro-batch
    Accumulate: total_grads += micro_grads / 16
  Optimizer step (using accumulated gradients)
```

Mathematically equivalent. The gradients are the same (averaged over the same 32 examples). But memory usage is that of batch=2 instead of batch=32.

### The Implementation Challenge

candle's `GradStore::new()` is **private** — you can't construct a gradient store from scratch. You can only get one from `tensor.backward()`. But `GradStore::insert()` and `GradStore::get()` are public. So:

```rust
pub fn accumulate_and_step(&mut self, loss: &Tensor) -> Result<Option<usize>> {
    // Scale loss by 1/N for gradient averaging
    let scaled = if self.config.grad_accum_steps > 1 {
        (loss / self.config.grad_accum_steps as f64)?
    } else {
        loss.clone()
    };

    let grads = scaled.backward()?;

    // First micro-batch: store as-is (this gives us a GradStore to work with)
    // Subsequent micro-batches: add to existing
    match &mut self.accum_grads {
        None => { self.accum_grads = Some(grads); }
        Some(accum) => {
            for var in self.varmap.all_vars() {
                let t = var.as_tensor();
                if let Some(new_g) = grads.get(t) {
                    let sum = if let Some(existing) = accum.get(t) {
                        (existing + new_g)?
                    } else {
                        new_g.clone()
                    };
                    accum.insert(t, sum);
                }
            }
        }
    }

    self.accum_count += 1;

    if self.accum_count >= self.config.grad_accum_steps {
        // Step optimizer with accumulated (averaged) gradients
        if let Some(grads) = self.accum_grads.take() {
            self.optimizer.step(&grads)?;
        }
        self.accum_count = 0;
        self.step_count += 1;
        self.scheduler.step();
        return Ok(Some(self.step_count));
    }

    Ok(None)  // Not a full step yet
}
```

### Important: The Previous Implementation Was Wrong

The original `accumulate_and_step` called `optimizer.backward_step()` per micro-batch. This means N optimizer steps with scaled loss — NOT gradient accumulation. The optimizer updates weights N times instead of once, with each update based on only 1/N of the batch. Mathematically different, practically worse.

The correct implementation calls `loss.backward()` to get raw gradients, accumulates them across micro-batches, then calls `optimizer.step(&grads)` exactly ONCE with the combined gradients.

### Configuration

```rust
BrainConfig {
    grad_accum_steps: 1,   // default: no accumulation
    // phase2:
    grad_accum_steps: 16,  // effective batch = 2 × 16 = 32
}
```

Override with `GESTALT_ACCUM_STEPS=16` environment variable. All 127 tests pass with accumulation enabled.

---

## 20. Where It Stands Now

### Source Files (February 2026)

| File | ~LOC | Tests | What It Does |
|------|------|-------|-------------|
| brain.rs | 2,000 | 20 | Unified brain: encoder, decoder, policy, memory, generation |
| transformer.rs | 650 | 8 | Causal transformer with gradient-safe RMSNorm and softmax |
| tokenizer.rs | 1,100 | 22 | PlanTokenizer (373) + ConceptTokenizer (adaptive BPE) |
| planner.rs | 700 | 6 | 17-state FSM constrained plan decoder |
| training.rs | 550 | 10 | AdamW, cosine LR, weighted CE, early stopping, grad accum |
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

Every test runs, every test passes. This includes gradient flow tests, tokenizer encode/decode round-trips, plan FSM validation, memory persistence, cross-session recall, and end-to-end pipeline execution.

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

### Current Training Run

v23 is running: d=512, 92K pairs, batch=48, 50K steps. GPU at 99% utilization. Even though we know the 92K corpus has personality issues, training it to completion gives us a data point: does longer training (50K vs 11K steps) sharpen the personality, or is it permanently diluted?

### Hardware

- **GPU**: NVIDIA RTX 5070 Ti — 16GB VRAM, Blackwell architecture
- **RAM**: 24GB system memory (28GB WSL allocation)
- **OS**: WSL2 on Windows (Linux kernel 6.6.87)
- **VRAM usage**: ~5.3GB during training (model + optimizer + activations), ~10.7GB headroom
- **Throughput**: 2.2 steps/sec at d=512, batch=48

---

## 21. The Road Ahead

### Near Term: Scale Up (d=1024)

The biggest single improvement available. Going from d=512 to d=1024 means:
- 4x more capacity per concept vector (1024 dimensions vs 512)
- 8 encoder + 8 decoder layers (vs 1+4 now)
- ~200M parameters total (vs ~50M now)
- Should sustain coherence well past the current 15-token wall

Gradient accumulation is implemented and tested. The next step is a d=1024 training run with `micro_batch=2, accum_steps=16` (effective batch=32).

### Near Term: Quality-Filtered Corpus

The v23 experiment proved that more data doesn't mean better output. The next corpus expansion will:

1. Start with the 21K curated pairs (v22's corpus — proven personality)
2. Filter aggressively: each new pair must match the JARVIS voice profile
3. Target 50K pairs with consistent personality, not 100K with diluted personality
4. Add more of what works: emotional intelligence, technical wit, cultural references

### Phase 3: Memory-Augmented Training

Train the decoder with memory prefix containing randomly sampled prior dialogues. The `memory_projector` weights (currently random) will learn to project retrieved memories into useful context.

This is the M-004 fix in action — the architecture is ready, the training pipeline has the hooks, we just need to pull the trigger on a training run.

### Phase 4: Multi-Turn + ReAct

Make GESTALT conversational and autonomous:

- **Session state**: 32-turn ring buffer for conversation context. Already implemented in `session.rs`.
- **ReAct loop**: Reason → Act → Observe → Reason for complex, multi-step tasks. The model doesn't just execute a plan once — it observes the results and decides what to do next.
- **Multi-turn context**: Maintain coherence across conversation turns. "Open that file" should know what "that" refers to from the previous turn.

### Phase 5: Online Learning

This is where GESTALT becomes fundamentally different from static models:

- **Experience buffer**: Store every interaction with automatic success/failure tagging
- **Micro-training**: After every N interactions, fine-tune on successful ones (100-500 steps)
- **Memory consolidation**: Compress repetitive episodes into abstract knowledge
- **GRPO**: Group Relative Policy Optimization on plan quality — learn from which plans succeed and which fail

The result: a model that gets better at YOUR tasks, on YOUR codebase, over time.

### Phase 6: Proactive Intelligence

The endgame. GESTALT anticipates and suggests:

- **Persistent HTTP server**: `./gestalt serve` — a daemon that watches your development environment
- **File watching**: Monitor file changes, test results, git status
- **Proactive suggestions**: "Build failed because of the type change in line 47 — want me to fix the callers?"
- **Dynamic tool registration**: Add new tools at runtime without recompiling
- **JARVIS personality consistency**: Trained into the weights through Phase 5's online learning

This is the phase where GESTALT stops being a tool you invoke and becomes an assistant that's aware of your work.

---

## 22. GESTALT vs ChatGPT

| Aspect | ChatGPT (GPT-4) | GESTALT V5 |
|--------|-----------------|------------|
| Parameters | ~1.8 trillion | ~50M (current), ~200M (target) |
| Architecture | Mixture of Experts | Single unified transformer |
| Vocabulary | ~100K tokens (BPE) | 459 tokens (concept-space BPE) |
| Training data | Trillions of tokens | 21K curated pairs |
| Memory | None (per-session) | Persistent SQLite + episodic |
| Tool use | Function calling (text) | Real subprocess execution |
| Planning | Implicit (in-context) | Explicit FSM-constrained |
| Plan validity | Best effort | Guaranteed syntactically valid |
| Personality | System prompt (fragile) | Trained into weights |
| Training time | Months on thousands of GPUs | ~90 minutes on one GPU |
| Cost to run | API pricing ($20/month+) | Free (local GPU) |
| Customizability | Black box | Full (own every weight) |
| Privacy | Data leaves your machine | Everything stays local |
| Learning | Frozen after RLHF | Continuous online learning (Phase 5) |

### What GESTALT Can't Do

**General knowledge.** ChatGPT has read the internet. GESTALT has read 21K dialogue pairs. Ask it about the Franco-Prussian War and you'll get word salad.

**Instruction following.** ChatGPT has been RLHF'd on millions of human preference comparisons. GESTALT has a 64-task curriculum. Complex, multi-step instructions with nuanced constraints are beyond its current capability.

**Language quality.** ChatGPT generates fluid, coherent prose for paragraphs and pages. GESTALT generates ~10-15 coherent tokens before degrading. This is the capacity bottleneck — d=1024 should significantly improve this.

**Multi-modal understanding.** ChatGPT can process images, audio, and video. GESTALT is text-only.

### What GESTALT Can Do Better

**Memory.** Across sessions. Across restarts. Across weeks. Tell it something once, it remembers forever (until FIFO eviction). ChatGPT starts from zero every conversation.

**Local execution.** Real tools on your machine with timeout enforcement and safety levels. Not a sandboxed code interpreter — actual subprocess access to your files, your tools, your environment.

**Plan validity.** Every plan is syntactically valid, guaranteed by a 17-state FSM. ChatGPT can generate malformed JSON function calls. GESTALT cannot generate an unparseable plan.

**Transparency.** Inspect every weight, every attention pattern, every training step. When something goes wrong, you can trace the exact data flow and find the exact failure point. Try that with a 1.8T parameter model behind an API.

**Cost.** Once trained, runs on your GPU for free. No API rate limits, no token pricing, no monthly subscription. The weights are yours.

**Privacy.** Nothing leaves your machine. No telemetry, no data collection, no "we may use your conversations for training." Your code, your conversations, your data — all local.

### The Philosophical Difference

ChatGPT is a general-purpose oracle. You ask questions, it answers from its vast training data. It doesn't know you, doesn't learn from you, doesn't care what you did yesterday.

GESTALT is a specialized partner. It knows your codebase, remembers your preferences, learns from your corrections, and operates on your machine with your tools. It's not as smart in general — but it's deeply integrated with your specific workflow.

The analogy: ChatGPT is a world-class consultant you hire by the hour. GESTALT is a junior engineer who lives in your office, knows where every file is, and gets better at your specific problems every day.

---

## 23. Design Philosophy

### The Gestalt Principle

The name says everything. A gestalt is "a whole that is greater than the sum of its parts." The brain, the memory, the planner, the executor — none of them work well alone. But together, sharing a common concept space, they become something more.

This isn't just a marketing statement. It's an architectural constraint. Every component reads from the same concept_vec. Memory keys are concept_vecs. Decoder prefixes are projected concept_vecs. Policy inputs are concept_vecs. One representation, many uses.

The alternative — separate models for each task — is easier to build but fundamentally limited. A separate planner can't benefit from the language model's understanding of tool names. A separate memory system can't leverage the encoder's concept space for retrieval. Separate systems can be individually excellent but collectively incoherent.

### Language as Interface, Not Substrate

In GESTALT, language is the INPUT and OUTPUT of the system, but not the INTERNAL representation. The concept_vec (512 floating-point numbers) is the reasoning substrate. Language comes in, gets compressed to a concept, decisions happen in concept space, and language comes out.

This is fundamentally different from LLMs, where everything is language tokens all the way through. GPT-4 "thinks" in language — its internal representations are contextual token embeddings. GESTALT "thinks" in concept vectors — a compressed, learned representation that strips away the surface form of language.

Why does this matter? Because concept vectors can be:
- Compared (cosine similarity)
- Stored and retrieved (memory)
- Projected into different modalities (language, plans, policy decisions)
- Computed in one forward pass (~6ms)

Language tokens need to be processed sequentially, one by one. Concept vectors are computed all at once.

### Memory-First Design

V4's biggest mistake was training without memory and bolting it on later. V5's biggest design commitment is memory-first: every component is designed to work WITH memory from the start.

This means:
- The decoder has memory prefix slots from architecture definition
- Training includes randomly sampled memory prefixes from epoch 0
- The memory projector learns useful representations alongside everything else
- Memory retrieval is baked into the inference pipeline, not an optional add-on

The result: when we turn on persistent memory, the model is already trained to use it. No fine-tuning, no special adaptation, no "memory injection" hacks. The model was trained in a world where memories exist.

### Fail Loudly, Not Silently

GESTALT's bug history (44 documented failures) has a clear theme: the worst bugs are the silent ones. candle's RmsNorm silently killing gradients. The decoder silently ignoring the concept prefix. The training loop silently overfitting while loss looked healthy.

The design response: make failures visible.

```
- Gradient flow test: runs before first training, verifies every parameter gets gradients
- Concept diversity check: monitors pairwise cosine similarity during training
- Generation diagnostics: greedy decode every 5K steps to check coherence
- Validation split: 10% of data reserved for overfitting detection
- Early stopping: automatically halts when learning plateaus
```

The philosophy: if something can fail silently, add a check. The 10 minutes spent writing a diagnostic saves 40+ hours of blind debugging (proven by M-032).

### Own Every Weight

GESTALT is built from scratch not for the challenge, but for the understanding. When the model generates "Step one: read the error message," we can trace exactly which attention heads contributed, which encoder layers were most active, which concept dimensions were highest, and which training examples most influenced the output.

This transparency is the foundation for Phase 5 (online learning) and Phase 6 (proactive intelligence). You can't improve what you can't measure. You can't measure what you can't inspect. You can't inspect a black box.

---

## 24. Glossary

**Attention** — Mechanism letting each position look at others and decide what to focus on. Each position computes Query (what I want), Key (what I offer), Value (what I carry). High Q-K match → more V influence.

**Autoregressive** — Generating one token at a time, each depending on all previous tokens. Like typing a sentence where each letter is predicted from all previous letters.

**BOS/EOS** — Begin/End Of Sequence. Special tokens marking start and end of text. BOS tells the decoder "start generating." EOS tells it "stop generating."

**BPE (Byte Pair Encoding)** — Tokenization that learns to merge common byte pairs. "th" → one token. "the" → one token. Reduces sequence length while preserving information.

**Causal transformer** — Transformer where each position only sees itself and earlier positions (can't peek at future tokens). This is essential for generation — you can't attend to words you haven't generated yet.

**candle** — Rust-native tensor computation library by Hugging Face. Provides autograd and CUDA support. Young framework with some bugs (see M-032).

**Concept vector** — 512-dimensional vector representing the compressed meaning of input text. The central representation in GESTALT. All downstream components read from this single representation.

**Cosine similarity** — How similar two vectors are by direction. 1.0 = same direction (identical meaning). 0.0 = perpendicular (unrelated). -1.0 = opposite direction.

**Cross-entropy loss** — How surprised the model is by the correct answer. -log(P(correct)). Low = correct, high = wrong. Minimizing cross-entropy = maximizing probability of correct predictions.

**d_model** — Dimensionality of the transformer's hidden representations. Currently 512, targeting 1024. Higher = more capacity = better representations = more parameters to train.

**Denoising** — Deliberately corrupting training data (replacing tokens with PAD) to force the model to attend to the concept prefix instead of relying on local context alone.

**Dropout** — During training, randomly zero 10% of values in each layer. Forces redundancy, prevents over-reliance on single neurons, extends useful training life.

**Early stopping** — Stop training when validation loss stops improving for `patience` consecutive checks. Prevents overfitting — training too long causes memorization.

**Episodic memory** — Storing specific experiences (goal, response, success) for later retrieval. Like remembering "last time I asked about colors, the user said blue." Contrasts with semantic memory (abstract facts).

**FIFO** — First In, First Out. Oldest item removed when container is full. Simple eviction policy for memory.

**FSM (Finite State Machine)** — Fixed states with transition rules. Used to constrain plan generation so every plan is syntactically valid. 17 states in GESTALT's planner.

**GELU** — Gaussian Error Linear Unit. Smooth activation function in the MLP. Lets some negative values through (unlike ReLU which blocks all negatives). More expressive.

**Gradient accumulation** — Summing gradients across multiple micro-batches before updating weights. Allows large effective batch sizes with limited VRAM. Essential for d=1024 training.

**GradStore** — candle's internal structure for storing gradients. Maps tensor IDs to gradient tensors. Created by `loss.backward()`, consumed by `optimizer.step()`.

**Logits** — Raw unnormalized output scores before softmax converts them to probabilities. Higher logit = higher probability after softmax.

**Mean pooling** — Average all non-padding positions to produce one vector. Used by ConceptEncoder to compress variable-length input into fixed-size concept_vec.

**MHA (Multi-Head Attention)** — Multiple attention mechanisms in parallel, each with own Q/K/V projections. Each head can focus on different aspects of the input. GESTALT uses 8 heads.

**Nucleus sampling** — See Top-P.

**Overfitting** — When the model memorizes training data instead of learning general patterns. Detected by validation loss increasing while training loss decreases. Prevented by dropout, early stopping, and sufficient data.

**RMSNorm** — Normalizes vectors by root-mean-square (not mean-centered like LayerNorm). Keeps values in a stable range through layers. Custom implementation in GESTALT due to candle-nn bug.

**RoPE** — Rotary Position Embedding. Encodes position by rotating Q and K vectors. Lower dimensions rotate faster (local position), higher dimensions slower (global position). Generalizes to unseen sequence lengths.

**Safetensors** — Safe file format for neural network weights. No arbitrary code execution (unlike pickle). Used for all checkpoint saves.

**SFT (Supervised Fine-Tuning)** — Training on (input, correct output) pairs. The model sees the correct answer and adjusts weights to predict it. The primary training method for GESTALT.

**Top-K** — Only consider the K most probable next tokens during generation. Prevents low-probability garbage from being sampled.

**Top-P (nucleus sampling)** — Only consider tokens whose cumulative probability reaches P. Adapts candidate pool based on model confidence — narrow when confident, wide when uncertain.

**VRAM** — Video RAM. GPU memory. All weights, gradients, optimizer states, and activations must fit here during training. 16GB on the RTX 5070 Ti.

**Weight decay** — Slowly pulling all weights toward zero during training. Prevents extreme weight values that cause overfitting. Applied in AdamW optimizer.

---

*Built with Rust, candle, and an unreasonable amount of determination.*
*GESTALT WIRED-V5 — February 2026*

*"The encoder was dead for twelve runs. We just didn't know how to listen."*

*"val_loss 1.88 on 92K pairs sounds better than val_loss 1.9 on 21K pairs. It wasn't."*

*"After 2 failed experiments, stop and build a diagnostic. One diagnostic run > 30 blind experiments. Always."*
