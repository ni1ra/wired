# GESTALT: Building an AI Brain From Scratch in Rust

**A complete technical guide to WIRED-V5 -- what it is, how every component works, and why every design decision was made.**

*February 2026*

---

## Table of Contents

1. [Introduction -- What Is GESTALT?](#1-introduction----what-is-gestalt)
2. [Why Build From Scratch? Why Rust?](#2-why-build-from-scratch-why-rust)
3. [Architecture Overview](#3-architecture-overview)
4. [The Transformer -- The Universal Building Block](#4-the-transformer----the-universal-building-block)
5. [The Brain -- Where Everything Meets](#5-the-brain----where-everything-meets)
6. [The Tokenizer -- From Bytes to Meaning](#6-the-tokenizer----from-bytes-to-meaning)
7. [The Memory System -- Episodic Recall](#7-the-memory-system----episodic-recall)
8. [The Planner -- FSM-Constrained Decoding](#8-the-planner----fsm-constrained-decoding)
9. [The Policy Heads -- Making Decisions](#9-the-policy-heads----making-decisions)
10. [Training -- Teaching the Brain](#10-training----teaching-the-brain)
11. [What We Learned -- Findings From v18 Through v23](#11-what-we-learned----findings-from-v18-through-v23)
12. [What Comes Next](#12-what-comes-next)
13. [Glossary](#13-glossary)

---

## 1. Introduction -- What Is GESTALT?

GESTALT is an AI brain built entirely from scratch in Rust. Not a wrapper around
an existing model. Not a fine-tuned LLaMA. Every matrix multiplication, every
attention head, every gradient computation, every training loop -- written by hand
using the candle-rs tensor library for GPU acceleration.

The name comes from the German word *Gestalt*, meaning "a whole that is greater
than the sum of its parts." This captures the core philosophy: a single unified
neural network with specialized regions sharing a common understanding of the
world, rather than separate models stitched together with glue code.

### What It Does

Give GESTALT a goal in plain English:

```
"search for the GPU policy code and then open it"
```

It will:

1. **Understand** your intent by encoding the text into a concept vector -- a
   dense mathematical summary of what you mean.
2. **Classify** the action type using policy heads (search + read = composite).
3. **Plan** the steps using an FSM-constrained decoder that can only produce
   syntactically valid tool sequences.
4. **Execute** those steps in real subprocesses on your machine -- ripgrep,
   file reads, tests, git operations.
5. **Remember** the experience by storing the concept vector and outcome in
   episodic memory backed by SQLite.
6. **Respond** in natural language with a consistent JARVIS personality
   trained into the weights.

All of this happens inside one binary. No API calls. No cloud dependency.

### The JARVIS Personality

GESTALT's language model is trained on curated dialogue pairs written in the
voice of JARVIS (MCU) crossed with TARS (Interstellar). Dry wit, competence,
helpful without being sycophantic. This is not a system prompt that can be
jailbroken -- the personality is baked into the weights themselves. It is the
only voice the model knows how to produce.

### The Scale

Let us be direct about the size differential. GPT-4 is estimated at roughly
1.8 trillion parameters, trained on trillions of tokens across thousands of
GPUs for months. GESTALT is approximately 50 million parameters at its current
training scale, trained on 92,000 dialogue pairs on a single RTX 5070 Ti in
under two hours.

This is not a competition. GESTALT does things that trillion-parameter cloud
models fundamentally cannot: it runs on your machine, remembers you across
sessions, executes real tools in your environment, and lets you inspect every
weight and gradient. The question is not "can this beat ChatGPT?" but rather
"can this become something ChatGPT can never be?"

---

## 2. Why Build From Scratch? Why Rust?

### The Case for From-Scratch

Most AI projects start by downloading a pre-trained model (LLaMA, Mistral,
Phi) and fine-tuning it. This gets you impressive results quickly, but you
inherit an architecture you do not fully control and cannot deeply modify.
Want to add episodic memory as a first-class concept? Want a concept
bottleneck that forces the model to compress meaning before generating
text? Want an FSM-constrained decoder that physically cannot produce
invalid tool plans? These architectural innovations require building from
the foundation.

The educational value is immense: when something fails, you know exactly
where to look. When candle-rs's built-in RmsNorm breaks gradient flow (and
it did -- more on that later), you can replace it with three lines of basic
tensor operations because you understand what RmsNorm actually computes.

### Why Rust

Most deep learning happens in Python via PyTorch or JAX. Rust is an unusual
choice, and there are concrete reasons for it:

**Performance without a runtime.** Rust compiles to native code with no
garbage collector. The training binary uses ~98MB of RAM for a 50M parameter
model -- the rest is GPU VRAM. Python+PyTorch would use several gigabytes of
system RAM for the interpreter, framework overhead, and CUDA bindings before
a single tensor is allocated.

**candle-rs.** This is Hugging Face's Rust tensor library. It provides the
GPU compute (CUDA matrix multiplications, element-wise operations) without
PyTorch's massive dependency tree. A clean `cargo build --release --features
cuda` produces a single binary with everything statically linked.

**Memory safety as architecture.** Rust's ownership system prevents entire
classes of bugs that plague long-running AI systems: double-free, use-after-
free, data races in concurrent generation. The type system catches mistakes
at compile time rather than during a 2-hour training run.

**Single binary deployment.** `gestalt train --config default` runs the
complete training pipeline. `gestalt serve --config default` launches the
interactive server. No pip install, no conda environments, no version
conflicts between numpy and scipy. One binary.

### The candle-rs Constraint: f32 Only

candle-rs v0.8.4 does not support fp16 training. All tensors are 32-bit
floats. This means GESTALT uses twice the VRAM that a PyTorch fp16 model
would need for the same parameter count. On the RTX 5070 Ti's 16GB VRAM,
this limits the practical model size to roughly 200M parameters (800MB of
weights + 2.4GB of optimizer states + activations + KV cache).

This constraint shaped every dimension choice: d_model=512 for Phase 1
(~50M params, fits comfortably), d_model=1024 for Phase 2 (~200M params,
tight but viable with gradient accumulation).

### The candle-rs Constraint: Broken Built-in Operations

Two candle-nn v0.8.4 built-in operations have silently broken backward
passes:

1. **`candle_nn::RmsNorm`** -- breaks the autograd computation graph.
   Every tensor passing through it becomes disconnected from gradient
   tracking. Since transformers use RmsNorm twice per layer (before
   attention and before the MLP), this means zero gradients flow to
   any parameter through the transformer layers.

2. **`candle_nn::ops::softmax_last_dim`** -- same issue. Breaks the
   gradient chain in attention, preventing the query and key projections
   from receiving gradients.

These are not edge cases -- they silently prevent ANY transformer from
learning anything except the final output projection. The model compiles,
runs, produces outputs, and even reports decreasing loss (because the last
linear layer can still learn), but 95% of the parameters never update.

Twelve training runs failed before this was diagnosed. The fix was
straightforward: replace both operations with manual implementations
using only basic tensor ops (sqr, mean, sqrt, exp, div) that have
working autograd. These custom implementations -- `GradRmsNorm` and
`grad_softmax_last_dim` -- are now the backbone of every transformer
in GESTALT.

The lesson: when using any framework, verify gradient flow through every
layer before training. Do not trust built-in operations blindly.

---

## 3. Architecture Overview

GESTALT follows a **concept-bottleneck architecture**: all information from
the input must pass through a compressed "concept vector" before reaching
the output. This bottleneck forces the model to form abstract
representations of meaning, rather than routing raw tokens through to
the output.

### The Full Pipeline

```
                        goal_text (UTF-8 bytes)
                              |
                      +-------v--------+
                      | TalkTokenizer  |  byte-level: "hello" -> [BOS, 104, 101, 108, 108, 111, EOS]
                      +-------+--------+
                              |
                         token_ids: (batch, seq_len)
                              |
                      +-------v--------+
                      | ConceptEncoder |  1-layer transformer + mean pooling
                      | (transformer)  |  goal_ids -> hidden states -> concept_vec
                      +-------+--------+
                              |
                       concept_vec: (batch, d_model)    <-- THE BOTTLENECK
                              |
         +--------------------+--------------------+
         |                    |                    |
  +------v------+    +-------v--------+    +------v------+
  |  Concept    |    |  Policy Heads  |    |   Memory    |
  |  Projector  |    | (5 classifiers)|    |   Retrieval |
  +------+------+    +-------+--------+    +------+------+
         |                    |                    |
   prefix: (B,16,d)    intent + actions      mem_vecs: (B,K,d)
         |                    |                    |
         |                    |              +-----v-----+
         |                    |              |  Memory   |
         |                    |              | Projector |
         |                    |              +-----+-----+
         |                    |                    |
         |                    |              projected_mem: (B,K,d)
         |                    |                    |
   +-----v-----------+       |     CROSS-ATTENTION|
   | Language Decoder |       |         +----------+
   | (4-layer xformer)|<-----+         |
   |                  |                 |
   | self-attn: prefix+text             |
   | cross-attn: Q from hidden <--------+ K/V from memory
   |                  |
   +--------+---------+
            |
      response text
```

The key architectural distinction: **concept prefix** enters the decoder through
self-attention (prepended to the text token sequence), while **memory** enters
through dedicated cross-attention layers (Q from the decoder's hidden states,
K/V from projected memory vectors). This separation keeps the self-attention
sequence short and focused while giving memory its own dedicated attention
pathway.

### Data Flow: Tensor Shapes

Here is the exact shape of every tensor as it flows through the system.
All shapes are at fp32. Default config: d_model=512, 8 attention heads.

```
Input:
  goal_text: &str                           "hello"
  token_ids: (1, 128)                       byte-level, left-padded

ConceptEncoder:
  tok_emb:      (1, 128, 512)              token -> embedding vector
  + RoPE:       applied per attention head  positional information injected
  transformer:  (1, 128, 512)              1-layer, full causal attention
  mean_pool:    (1, 512)                   average over non-PAD positions

ConceptProjector:
  linear:       (1, 8192)                  512 -> 16 * 512
  reshape:      (1, 16, 512)              16 prefix embedding vectors

MemoryRetrieval:
  query:        (512,)                     concept_vec as flat query
  top-K cosine: K=8 nearest neighbors     from stored concept vectors
  mem_proj:     (1, 8, 512)               projected memory (for cross-attention)

LanguageDecoder:
  prefix:       (1, 16, 512)              concept prefix only (no memory in prefix)
  BOS + text:   (1, 256, 512)             autoregressive token embeddings
  self-attn:    (1, 272, 512)             prefix + text concatenated
  cross-attn:   Q=(1, 272, 512)           from decoder hidden states
                K/V=(1, 8, 512)           from projected memory
  logits:       (1, 256, 459)             one probability distribution per position
  output:       "Hello! How can I..."     sampled token by token

PolicyHeads:
  intent:       (1, 16)                   16-class intent classification
  actions:      (1, 6, 16)               6 action slots, 15 tools + END
  patterns:     (1, 6, 6)                per-slot pattern index
  files:        (1, 6, 10)               per-slot file index
  picks:        (1, 6, 129)              per-slot argument index
```

### Why a Concept Bottleneck?

The concept bottleneck is the single most important architectural decision
in GESTALT. Traditional language models pass information from input to
output through high-dimensional hidden states at every layer. GESTALT
instead forces all information through a single 512-dimensional vector.

This has three key benefits:

1. **Interpretable representations.** The concept vector is a point in a
   space where similar meanings are nearby. You can measure the cosine
   similarity between concept vectors for "hello" and "hey there" and
   verify that they are closer to each other than to "run the tests."
   This is not possible with the hidden states of a standard language
   model, which are entangled across all positions.

2. **Memory becomes natural.** Since every interaction is summarized as a
   single vector, storing and retrieving memories is just cosine similarity
   search in a vector database. No RAG pipeline, no embedding model -- the
   brain's own encoder produces the vectors.

3. **Multiple downstream tasks share one representation.** The same concept
   vector feeds the language decoder (for generating responses), the policy
   heads (for choosing actions), and the planner (for structured tool plans).
   One encoding step, multiple consumers.

The cost is information loss. A 512-dimensional vector cannot capture every
nuance of a 128-token input. This is a deliberate trade-off: we accept some
loss of detail in exchange for a clean, structured representation that the
rest of the system can reason about.

---

## 4. The Transformer -- The Universal Building Block

GESTALT uses the transformer architecture as its building block. Every
neural component in the system -- the encoder, the decoder, the planner,
the policy backbone -- is a transformer. Understanding this component is
essential to understanding everything else.

### What a Transformer Does

A transformer takes a sequence of vectors and produces a new sequence of
vectors where each output has been informed by all the others. If you feed
it the embeddings for "the cat sat on the mat," the output for "sat" will
contain information about who did the sitting (the cat) and where (on the
mat). This is called "attention" -- each position attends to all other
positions to gather context.

The key insight of transformers versus earlier architectures (RNNs, LSTMs)
is that attention is computed in parallel across all positions, rather than
sequentially left-to-right. This makes transformers dramatically faster to
train on GPUs, which excel at parallel computation.

### Architecture of a Single Transformer Block

Each transformer block has three sub-modules: self-attention, optional
cross-attention, and a feed-forward network (MLP). All use residual
connections and normalization.

```
        input: (batch, seq, d_model)          external memory (optional)
          |                                         |
   +------v------+                                  |
   |   RmsNorm   |  normalize to unit variance      |
   +------+------+                                  |
          |                                         |
   +------v------+                                  |
   |  Self-Attn  |  each position gathers info      |
   +------+------+  from all other positions         |
          |                                         |
   +------v------+                                  |
   |   + input   |  residual connection              |
   +------+------+                                  |
          |                                         |
   +------v------+                                  |
   |   Dropout   |  randomly zero elements (p=0.1)  |
   +------+------+                                  |
          |                                         |
          +--[if cross-attn enabled]---+            |
          |                            |            |
          |               +------------v--------+   |
          |               |   RmsNorm           |   |
          |               +------------+--------+   |
          |                            |            |
          |               +------------v--------+   |
          |               | Cross-Attention     |<--+
          |               | Q: from hidden      |  K/V: from memory
          |               +------------+--------+
          |                            |
          |               +------------v--------+
          |               |   + prev residual   |
          |               +------------+--------+
          |                            |
          |               +------------v--------+
          |               |   Dropout           |
          |               +------------+--------+
          |                            |
          +----------------------------+
          |
   +------v------+
   |   RmsNorm   |  normalize again
   +------+------+
          |
   +------v------+
   |     MLP     |  per-position nonlinear transformation
   +------+------+
          |
   +------v------+
   |   + prev    |  residual connection again
   +------+------+
          |
   +------v------+
   |   Dropout   |  random zeroing again
   +------+------+
          |
        output: (batch, seq, d_model)
```

The input and output have the same shape. This means transformer blocks are
stackable: you can chain 1, 4, 8, or 100 of them, each refining the
representation further. GESTALT uses 1 layer for the encoder and 4 layers
for the decoder at Phase 1 scale.

The cross-attention sub-module is optional. It is enabled on the language
decoder (which needs to attend to external memory) and disabled on the
concept encoder and policy backbone (which have no external memory input).
When disabled, the block is identical to a standard transformer block.
When enabled, each layer gets its own cross-attention parameters, allowing
different layers to attend to different aspects of the memory.

### RmsNorm: Gradient-Safe Normalization

Before attention and the MLP, the input is normalized using Root Mean Square
Layer Normalization (RmsNorm). This stabilizes training by ensuring that
vectors do not grow unboundedly large or shrink to near-zero as they pass
through many layers.

The formula is:

```
RmsNorm(x) = x / sqrt(mean(x^2) + epsilon) * gamma
```

Where:
- `x` is the input vector (the last dimension of shape `(batch, seq, d_model)`)
- `mean(x^2)` is the mean of the squared elements along the d_model dimension
- `epsilon = 1e-6` prevents division by zero
- `gamma` is a learnable scale parameter (initialized to 1.0)

This differs from the more common LayerNorm in two ways: it does not subtract
the mean (no centering), and it uses the root mean square instead of variance.
RmsNorm is slightly faster and has been shown to work equally well in practice.

**The candle bug.** As described earlier, candle-nn's built-in `RmsNorm`
implementation breaks the autograd computation graph. GESTALT uses a custom
`GradRmsNorm` that computes the exact same operation using only basic tensor
ops with working backward passes:

```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (variance + self.eps)?.sqrt()?;
    let normed = x.broadcast_div(&rms)?;
    normed.broadcast_mul(&self.weight)
}
```

Each of these operations (sqr, mean_keepdim, sqrt, broadcast_div, broadcast_mul)
has a correct autograd implementation in candle-core. By composing them
explicitly rather than relying on the higher-level wrapper, gradient flow
is preserved.

### Multi-Head Self-Attention

Attention is the mechanism by which each position in the sequence gathers
information from every other position. It uses three projections: Query (Q),
Key (K), and Value (V).

The intuition: imagine a position wants to find relevant context. It formulates
a question (the Query), every other position advertises what it knows (the Key),
and once a match is found, the relevant information (the Value) is retrieved.
It is like a soft version of a database lookup, where instead of matching
exactly, you get a weighted average based on similarity.

#### The Math

For each attention head:

```
Q = x @ W_q        shape: (batch, seq, head_dim)
K = x @ W_k        shape: (batch, seq, head_dim)
V = x @ W_v        shape: (batch, seq, head_dim)

scores = Q @ K^T / sqrt(head_dim)    shape: (batch, seq, seq)
weights = softmax(scores + mask)      shape: (batch, seq, seq)
output = weights @ V                  shape: (batch, seq, head_dim)
```

Where:
- `W_q, W_k, W_v` are learned projection matrices (no bias, following modern practice)
- `head_dim = d_model / n_heads` (512 / 8 = 64 per head)
- The division by `sqrt(head_dim)` prevents the dot products from growing too
  large, which would push softmax into saturation where gradients vanish
- `mask` is the causal mask (described below)
- `softmax` converts raw scores into a probability distribution

**Multi-head:** Instead of running one attention computation with the full
d_model=512 dimension, GESTALT runs 8 independent attention computations
("heads") with dimension 64 each, then concatenates the results. This lets
different heads learn to attend to different types of relationships -- one
head might focus on nearby tokens (local syntax), another on long-range
dependencies (semantic coherence).

```
head_1 = Attention(Q_1, K_1, V_1)    (batch, seq, 64)
head_2 = Attention(Q_2, K_2, V_2)    (batch, seq, 64)
...
head_8 = Attention(Q_8, K_8, V_8)    (batch, seq, 64)

concat = [head_1; head_2; ...; head_8]   (batch, seq, 512)
output = concat @ W_o                     (batch, seq, 512)
```

The output projection `W_o` remixes the concatenated heads back into the
d_model space.

**Gradient-safe softmax.** Just like RmsNorm, candle-nn's built-in softmax
has a broken backward pass. GESTALT uses a custom implementation:

```rust
fn grad_softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;        // numerical stability
    let shifted = x.broadcast_sub(&max)?;         // subtract max
    let exp = shifted.exp()?;                      // exponentiate
    let sum = exp.sum_keepdim(D::Minus1)?;        // normalizing constant
    exp.broadcast_div(&sum)                        // normalize to probabilities
}
```

The `max` subtraction is a standard numerical trick: softmax is invariant to
adding a constant, and subtracting the maximum prevents the exponentials from
overflowing to infinity.

#### The Causal Mask

Language models generate text left-to-right: each token predicts the next one.
During training, the model sees the entire sequence at once (for efficiency),
but position `i` must not be allowed to see tokens at positions `i+1, i+2, ...`
because those are the tokens it is trying to predict.

The causal mask enforces this constraint. It is a square matrix where position
`(i, j)` is 0 if `j <= i` (allowed to attend) and negative infinity if `j > i`
(forbidden). Adding this mask before softmax forces the forbidden positions to
have zero attention weight.

```
Causal mask for seq_len=5:

       pos 0   pos 1   pos 2   pos 3   pos 4
pos 0 [  0      -inf    -inf    -inf    -inf  ]
pos 1 [  0       0      -inf    -inf    -inf  ]
pos 2 [  0       0        0     -inf    -inf  ]
pos 3 [  0       0        0       0     -inf  ]
pos 4 [  0       0        0       0       0   ]
```

After softmax, the -inf entries become exactly 0.0, and each row sums to 1.0
over only the allowed positions.

**Precomputation.** GESTALT precomputes the causal mask once at model
initialization for the maximum sequence length (256), then slices it to the
actual sequence length during each forward pass. This avoids a per-call
allocation and a host-to-device memory transfer on every training step.

```rust
let causal_mask = build_causal_mask(cfg.max_seq_len, device)?;
// ... during forward:
let mask = causal_mask.narrow(2, 0, s)?.narrow(3, 0, s)?;
```

### Rotary Position Embeddings (RoPE)

Transformers are fundamentally position-agnostic: the attention mechanism
operates on sets, not sequences. Without positional information, the model
cannot distinguish "the cat chased the dog" from "the dog chased the cat."

GESTALT uses Rotary Position Embeddings (RoPE), which encode positions by
rotating the query and key vectors before computing attention scores. This
is elegant because it encodes *relative* positions: the attention score
between positions `i` and `j` depends only on their distance `|i - j|`,
not their absolute positions.

#### The Math

RoPE works by splitting each head's Q and K vectors into pairs of elements,
then rotating each pair by an angle that depends on the position.

First, precompute rotation frequencies:

```
theta_k = 1 / 10000^(2k / head_dim)    for k = 0, 1, ..., head_dim/2 - 1
```

This creates a geometric sequence of frequencies, from high frequency
(k=0, theta = 1.0) to low frequency (k=31, theta ~ 2.7e-5 for head_dim=64).

For position `p`, the rotation angle for dimension pair `k` is:

```
angle_{p,k} = p * theta_k
```

Then split the vector into first and second halves:

```
x = [x_1, x_2]    each of dimension head_dim/2

rotated = [x_1 * cos(angle) - x_2 * sin(angle),
           x_2 * cos(angle) + x_1 * sin(angle)]
```

This is literally a 2D rotation applied independently to each pair of
dimensions. Low-frequency dimensions rotate slowly (encoding long-range
position differences), while high-frequency dimensions rotate quickly
(encoding nearby position differences).

#### Implementation

```rust
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;      // first half
    let x2 = x.narrow(D::Minus1, half, half)?;    // second half

    let cos = cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

    let rotated_x1 = (x1 * cos - x2 * sin)?;
    let rotated_x2 = (x2 * cos + x1 * sin)?;
    Tensor::cat(&[rotated_x1, rotated_x2], D::Minus1)
}
```

The cos and sin tables are precomputed once at model initialization for all
positions up to max_seq_len (256), then sliced to the actual sequence length
during each forward pass.

#### YaRN: Context Extension Beyond Training Length

Standard RoPE is trained with a fixed maximum sequence length (256 for
GESTALT's current training). If the model encounters a longer sequence at
inference time, the rotational frequencies exceed anything seen during
training, and attention patterns break down. The model has learned to
associate certain rotation angles with certain relative distances, but
rotations beyond the training range produce meaningless position signals.

GESTALT implements YaRN (Yet another RoPE extensioN), a frequency-selective
scaling method that extends the effective context window without retraining.
The key insight is that not all RoPE dimensions are equally affected by
longer sequences.

**The three frequency bands.** Each dimension pair `k` in the RoPE table
has a corresponding wavelength:

```
wavelength_k = 2 * pi / theta_k = 2 * pi * 10000^(2k / head_dim)
```

A dimension with wavelength shorter than the training length can already
distinguish all positions within the training window -- it has completed
multiple full rotations. Stretching it would distort the fine-grained
local position information it encodes. A dimension with wavelength much
longer than the training length has barely rotated during training -- it
needs to be stretched to cover the extended range.

YaRN classifies dimensions into three bands:

```
For each dimension pair k with wavelength w_k:

  High-frequency (w_k < L_train):
    theta_yarn_k = theta_k                         (unchanged)

  Low-frequency (w_k > L_train * beta):
    theta_yarn_k = theta_k * (1 / scale_factor)    (NTK-scaled)

  Mid-frequency (L_train <= w_k <= L_train * beta):
    t = (w_k - L_train) / (L_train * (beta - 1))
    theta_yarn_k = theta_k * (1-t) + theta_ntk_k * t   (linear ramp)
```

Where:
- `L_train` is the maximum sequence length seen during training (`max_train_len`)
- `beta = 2.0` controls the width of the transition band
- `scale_factor = L_test / L_train` is the ratio of test to train sequence lengths
- `theta_ntk_k` is the NTK-scaled frequency: `1 / base_scaled^(2k / head_dim)`
- `base_scaled = 10000 * scale_factor^(d / (d-2))` where `d = head_dim`

The NTK (Neural Tangent Kernel) scaling adjusts the base frequency rather
than linearly interpolating positions. This preserves the relative spacing
between dimensions -- if dimension `k` was 10x the frequency of dimension
`k+1`, that ratio is maintained after scaling.

**The linear ramp** in the mid-frequency band provides a smooth transition.
Without it, there would be a discontinuity at the band boundary where some
dimensions suddenly jump from unscaled to fully NTK-scaled. The ramp ensures
that nearby dimensions receive similar treatment.

```rust
fn precompute_rope(seq_len: usize, head_dim: usize, max_train_len: usize, device: &Device)
    -> Result<(Tensor, Tensor, f32)>
{
    let scale = seq_len as f32 / max_train_len as f32;

    let theta: Vec<f32> = if seq_len > max_train_len {
        let beta = 2.0f32;
        let base_scaled = 10000.0 * scale.powf(head_dim as f32 / (head_dim as f32 - 2.0));

        (0..half).map(|i| {
            let freq_orig = 1.0 / 10000f32.powf(2.0 * i as f32 / head_dim as f32);
            let freq_scaled = 1.0 / base_scaled.powf(2.0 * i as f32 / head_dim as f32);
            let wavelength = 2.0 * PI / freq_orig;

            if wavelength < max_train_len as f32 {
                freq_orig                              // high-freq: unchanged
            } else if wavelength > max_train_len as f32 * beta {
                freq_scaled                            // low-freq: NTK-scaled
            } else {
                let t = (wavelength - max_train_len as f32)
                    / (max_train_len as f32 * (beta - 1.0));
                freq_orig * (1.0 - t) + freq_scaled * t  // mid-freq: ramp
            }
        }).collect()
    } else {
        // Standard RoPE when within training range
        (0..half).map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / head_dim as f32)).collect()
    };
    // ...
}
```

**When does YaRN activate?** Only when `seq_len > max_train_len`. During
training (where sequence lengths are within the trained range), standard
RoPE is used unchanged. YaRN is a pure inference-time enhancement. The
`max_train_len` field in `TransformerConfig` records the training context
window so the model knows when scaling is needed.

**Attention temperature compensation.** When extending the context window,
the entropy of attention distributions increases -- each position has more
candidates to attend to. This makes attention weights "flatter," reducing
the model's ability to focus sharply on relevant positions.

YaRN compensates by scaling the attention logits with a temperature factor:

```
attn_temp = sqrt(L_test / L_train)    when L_test > L_train
attn_temp = 1.0                        when L_test <= L_train

attention_logits = (Q @ K^T) / (sqrt(head_dim) * attn_temp)
```

This is mathematically equivalent to dividing the softmax temperature by
`sqrt(scale)`, sharpening the attention distribution to compensate for the
increased number of keys. At 2x the training length, `attn_temp = sqrt(2)
= 1.414`, which slightly sharpens attention. At 4x, `attn_temp = 2.0`,
which more aggressively concentrates attention on the most relevant keys.

The temperature factor is stored in the `WiredTransformer` struct and passed
through the attention computation:

```rust
let scale = (self.head_dim as f64).sqrt() * attn_temp as f64;
let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
let attn = (attn / scale)?;
```

#### RoPE Prefix Decoupling

GESTALT's decoder receives prefix tokens (concept projections) that are
NOT text tokens. These are dense vectors projected from the concept space
-- they do not have meaningful "positions" in a text sequence. Applying
RoPE to them would inject arbitrary positional information that has no
semantic meaning.

Consider what would happen without prefix decoupling: concept token 1 would
receive position 0's rotation, concept token 16 would receive position 15's
rotation. The model would learn that "the first concept token is always at
position 0" and "the sixteenth is at position 15," creating a rigid
positional dependency. If the number of concept tokens changes (e.g., from
16 to 32 in a future phase), these learned positional associations would
break. Worse, the concept projector would need to account for the positional
encoding it knows will be applied, coupling two components that should be
independent.

The solution: prefix tokens receive NO RoPE rotation. Only text tokens
(the actual generated response) receive positional encoding, starting at
position 0. This means:

```
Sequence:  [concept_1, concept_2, ..., concept_16, BOS, H, e, l, l, o]
RoPE:      [none,      none,      ..., none,        0,  1, 2, 3, 4, 5]
```

The implementation handles this by splitting Q and K into prefix and text
portions, applying RoPE only to the text portion, then re-concatenating:

```rust
if n_prefix > 0 && n_prefix < s {
    let q_prefix = q.narrow(2, 0, n_prefix)?;
    let q_text = q.narrow(2, n_prefix, n_text)?;
    let q_text = apply_rope(&q_text, cos, sin)?;
    (Tensor::cat(&[&q_prefix, &q_text], 2)?, ...)
}
```

This decoupling is maintained in both the training forward pass and the
cached generation path. During cached generation, the prefill step passes
`n_prefix` so that RoPE applies only to text tokens, and subsequent
single-token steps use `n_prefix=0` with `seq_offset` tracking the text
position count (not the total sequence count).

Note that with the cross-attention memory architecture, the prefix now
contains only concept tokens (not memory tokens). Memory enters through
cross-attention, which uses no RoPE at all -- memory vectors have no
sequential position, so positional encoding would be meaningless.

### Cross-Attention: Querying External Memory

In addition to self-attention (where each position attends to every other
position in the same sequence), GESTALT's decoder layers include a dedicated
**cross-attention** mechanism for querying external memory. This is the
architectural backbone of the memory system.

#### Why Cross-Attention Instead of Prefix Concatenation?

The original GESTALT design concatenated memory vectors into the prefix
alongside concept tokens. If you had 16 concept tokens and 8 memory tokens,
the decoder's self-attention operated over a 24-token prefix + text sequence.
This approach has three problems:

1. **Self-attention cost scales quadratically.** Adding K memory tokens to
   a prefix increases the self-attention cost by O((K + N)^2 - N^2) = O(K^2
   + 2KN), where N is the text length. For K=8 this is modest, but scaling
   to K=64 or K=256 memory entries would significantly increase cost.

2. **Memory competes for attention bandwidth.** In self-attention, every
   position attends to every other position. Memory tokens must compete
   with concept tokens and text tokens for the model's attention budget.
   A particularly "loud" memory entry could drown out the concept signal.

3. **Positional confusion.** Memory vectors have no inherent sequential
   order. Placing them in fixed prefix positions (e.g., positions 16-23)
   creates an artificial ordering that the model might learn to depend on.

Cross-attention solves all three problems. It gives memory its own dedicated
attention pathway, separate from the self-attention over the text sequence.

#### The Architecture

Cross-attention uses the same Q/K/V mechanism as self-attention, but with
a critical difference: **Q comes from the decoder's hidden states, while
K and V come from the external memory.**

```
Cross-attention in layer i:

  Q = LayerNorm(hidden_state) @ W_q    shape: (batch, seq, d_model)
  K = memory @ W_k                      shape: (batch, n_mem, d_model)
  V = memory @ W_v                      shape: (batch, n_mem, d_model)

  scores = Q @ K^T / sqrt(head_dim)     shape: (batch, n_heads, seq, n_mem)
  weights = softmax(scores)              shape: (batch, n_heads, seq, n_mem)
  output = weights @ V                   shape: (batch, n_heads, seq, head_dim)
```

Two key differences from self-attention:

- **No causal mask.** Every position in the decoder can attend to every
  memory entry. Memory is not sequential; there is no "future" to mask.
  All memory entries are fully visible to all decoder positions.

- **No RoPE.** Memory vectors are not text positions. They have no
  sequential ordering. Applying rotary positional encoding would inject
  meaningless position information. Cross-attention operates purely on
  content similarity between the query (what the decoder is computing)
  and the keys (what memory contains).

#### Implementation

```rust
struct CrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    fn forward(&self, x: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let (b, s, _d) = x.dims3()?;
        let m = memory.dim(1)?;  // number of memory entries

        let q = self.q_proj.forward(x)?;          // (b, s, d)
        let k = self.k_proj.forward(memory)?;      // (b, m, d)
        let v = self.v_proj.forward(memory)?;      // (b, m, d)

        // Reshape to multi-head, compute attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(...))?;   // (b, heads, s, m)
        let attn = (attn / scale)?;
        // No mask -- full attention to all memory entries
        let attn = grad_softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;                // (b, heads, s, head_dim)
        self.o_proj.forward(&out)
    }
}
```

Each decoder layer has independent cross-attention weights (`W_q`, `W_k`,
`W_v`, `W_o`), so different layers can learn to extract different aspects
from memory. Layer 1 might attend to memories that match the current topic,
while layer 4 might attend to memories that inform the response tone.

#### Placement Within the Transformer Block

Cross-attention sits between self-attention and the MLP, following the
standard encoder-decoder transformer pattern established by the original
Transformer paper:

```
input -> Self-Attention -> + residual -> Cross-Attention -> + residual -> MLP -> + residual -> output
```

Each sub-module has its own RmsNorm and dropout. The cross-attention norm
is separate from the self-attention norm and the MLP norm.

#### Graceful Degradation

When no memory is available (empty memory bank, or during early training
before memories accumulate), the cross-attention layers are simply skipped:

```rust
if let (Some(norm), Some(ca), Some(mem)) = (&self.cross_attn_norm, &self.cross_attn, memory) {
    let h = ca.forward(&norm.forward(&x)?, mem)?;
    x = (x + h)?;
}
```

The `Option<&Tensor>` pattern means the model gracefully handles the
no-memory case. The self-attention and MLP still function normally; only
the cross-attention contribution is absent. This is important for training:
the model learns to produce coherent responses with or without memory
context, rather than becoming dependent on memory being present.

#### Configuration

Cross-attention is controlled by the `use_cross_attn` field in
`TransformerConfig`. It is enabled for the language decoder and disabled
for the concept encoder and policy backbone:

```rust
TransformerConfig {
    use_cross_attn: true,   // decoder: has memory to attend to
    // ...
}
```

When `use_cross_attn` is false, the `CrossAttention` struct and its
corresponding `GradRmsNorm` are not allocated, saving both parameters and
computation for components that do not need external memory access.

### The MLP: Per-Position Transformation

After attention has gathered context from other positions, each position is
independently transformed through a two-layer MLP (feed-forward network):

```
MLP(x) = GELU(x @ W_gate + b_gate) @ W_down + b_down
```

Where:
- `W_gate` projects from d_model (512) to d_ff (2048) -- a 4x expansion
- `GELU` is the activation function (a smooth approximation of ReLU)
- `W_down` projects back from d_ff (2048) to d_model (512)

The 4x expansion to a larger intermediate dimension allows the MLP to
learn more complex transformations than would be possible at the d_model
dimension. Think of it as: attention figures out WHAT information is
relevant, then the MLP decides WHAT TO DO with that information.

GELU (Gaussian Error Linear Unit) is preferred over ReLU because it has
a smooth gradient everywhere, which helps with training stability:

```
GELU(x) = x * Phi(x)    where Phi is the CDF of the standard normal
```

In practice, it behaves like ReLU for large positive inputs (passes them
through) and smoothly attenuates negative inputs (instead of hard zeroing).

### Dropout: Preventing Memorization

During training, dropout randomly zeroes a fraction of the elements in a
tensor. This forces the model to be robust -- it cannot rely on any single
neuron, because that neuron might be disabled on the next training step.

GESTALT uses inverted dropout with rate p=0.1:

```rust
fn grad_dropout(x: &Tensor, p: f64, train: bool) -> Result<Tensor> {
    if !train || p <= 0.0 {
        return Ok(x.clone());   // no dropout during inference
    }
    let mask = Tensor::rand(0f32, 1f32, x.shape(), x.device())?
        .ge(p)?                  // keep with probability (1-p)
        .to_dtype(x.dtype())?;
    let scale = 1.0 / (1.0 - p);
    (x * mask * scale)?         // scale up survivors so expected value unchanged
}
```

The scaling by `1/(1-p)` is critical: during inference, dropout is disabled,
so all neurons are active. If we did not scale during training, the expected
magnitude of the output would be different between training and inference,
causing degraded generation quality.

Dropout is applied after both the attention and MLP sub-layers. It is only
active during training; during inference (generation), it is a no-op.

**The dropout discovery:** Training run v19 (no dropout) began overfitting
at step 6,000. Adding dropout=0.1 in v20 extended useful training to step
13,000+, a 2x improvement in training life before overfitting. This single
change improved coherence from 0% to ~30%.

### The Full Transformer Stack

A complete `WiredTransformer` consists of:

1. **Token embedding:** a lookup table mapping token IDs to d_model vectors
2. **N transformer blocks:** stacked attention + MLP layers
3. **Final RmsNorm:** normalizes the output of the last block
4. **LM head:** a linear projection from d_model to vocab_size (logits)
5. **Precomputed RoPE tables:** cos and sin for all positions
6. **Precomputed causal mask:** for the maximum sequence length

```rust
pub struct WiredTransformer {
    tok_emb: Embedding,              // token ID -> vector
    layers: Vec<TransformerBlock>,    // N stacked blocks (with optional cross-attn)
    final_norm: GradRmsNorm,         // output normalization
    lm_head: Linear,                 // hidden state -> vocab logits
    rope_cos: Tensor,                // precomputed cos table (YaRN-aware)
    rope_sin: Tensor,                // precomputed sin table (YaRN-aware)
    causal_mask: Tensor,             // precomputed causal mask
    attn_temp: f32,                  // YaRN attention temperature (1.0 at train length)
}
```

The `attn_temp` field stores the YaRN attention temperature factor. It is
computed once at model initialization: 1.0 when `max_seq_len <= max_train_len`
(normal training), or `sqrt(max_seq_len / max_train_len)` when the model
is being used for sequences longer than its training window.

The same `WiredTransformer` struct is used for the concept encoder, the
language decoder, the planner, and the policy backbone -- only the
configuration (dimensions, layer count, vocab size, cross-attention) differs.

### KV Cache: O(1) Generation

During autoregressive generation (producing text one token at a time), the
naive approach recomputes attention over the entire sequence for each new
token. For a sequence of length N, generating T tokens requires O(N*T)
attention computations -- and N grows with each token generated.

The KV cache eliminates this quadratic cost. The insight: when generating
token at position T, the Key and Value projections for all positions
0..T-1 are identical to what they were when we generated position T-1.
Only the new token's K and V are new.

GESTALT stores the K and V tensors from each layer after each generation
step. On the next step, only the new token is processed through Q/K/V
projection, and its K/V are concatenated with the cached values:

```rust
fn forward_cached(&self, x: &Tensor, ..., kv_cache: &mut Option<(Tensor, Tensor)>) {
    let q = self.q_proj.forward(x)?;   // only new token(s)
    let k = self.k_proj.forward(x)?;
    let v = self.v_proj.forward(x)?;

    // Concat with cached K/V
    let (k, v) = if let Some((past_k, past_v)) = kv_cache.take() {
        (Tensor::cat(&[&past_k, &k], 2)?,
         Tensor::cat(&[&past_v, &v], 2)?)
    } else {
        (k, v)
    };
    *kv_cache = Some((k.clone(), v.clone()));

    // Attention: new Q against all K, gather from all V
    let attn = q.matmul(&k.transpose(...))?;
    // ...
}
```

The generation loop has two phases:

1. **Prefill:** Process the entire prompt (prefix + BOS) in one forward pass,
   populating the KV cache. This is O(N^2) but only happens once.

2. **Step:** For each new token, process only that single token, attending
   to the cached K/V from all previous positions. This is O(N) per token
   instead of O(N^2).

For generating a 200-token response with a 16-token prefix, the naive
approach would require ~50,000 attention operations (quadratic). With KV
cache, it requires ~16*200 + 200*200/2 = ~23,200 (roughly 2x speedup,
and the constant factor is much smaller because each step processes only
1 token through the transformer layers instead of the full sequence).

#### Causal Mask Simplification for Cached Steps

The causal mask during cached generation is simpler than during training.
During prefill, a standard causal mask is used (position `i` attends to
positions `0..=i`). But during single-token step calls, the new token is
always the latest in the sequence -- it should attend to everything in the
cache. So the mask is simply all zeros:

```rust
let mask = if s == 1 {
    // Single-token step: attend to all cached entries
    Tensor::zeros((1, 1, 1, kv_len), DType::F32, device)?
} else {
    // Prefill: standard causal mask
    // ...
};
```

This optimization avoids constructing and applying a mask tensor for the
common case (step calls), and is correct because the KV cache only ever
contains entries from the past -- there is no "future" to mask.

#### Memory Threading Through Cached Generation

The KV cache path threads external memory through every layer of the
decoder. Both `forward_with_prefix_cached` (prefill) and
`forward_step_cached` (per-token steps) accept an optional `memory`
parameter:

```rust
pub fn forward_with_prefix_cached(
    &self,
    prefix: &Tensor,
    input_ids: &Tensor,
    kv_caches: &mut Vec<Option<(Tensor, Tensor)>>,
    memory: Option<&Tensor>,    // cross-attention K/V source
) -> Result<Tensor> { ... }

pub fn forward_step_cached(
    &self,
    token_id: &Tensor,
    seq_offset: usize,
    kv_caches: &mut Vec<Option<(Tensor, Tensor)>>,
    memory: Option<&Tensor>,    // same memory for every step
) -> Result<Tensor> { ... }
```

The same projected memory tensor is passed to every generation step. This
is efficient because memory does not change during generation -- it was
retrieved once from the memory bank before generation started. The cross-
attention K/V projections for memory could theoretically be cached too
(since memory is constant), but this optimization is not yet implemented.

Note that the KV cache only stores self-attention K/V tensors. Cross-
attention has no cache because it re-computes Q (from the evolving hidden
state) against the same memory K/V at every layer and step.

### Cross-Entropy Loss

Training a language model means making the model's predicted probability
distribution match the actual next token. This is measured by cross-entropy
loss.

For a single position where the model predicts logits `z` (raw scores before
softmax) and the correct token is class `c`:

```
loss = -log(softmax(z)[c])
     = -z[c] + log(sum(exp(z[j]) for all j))
```

This has a nice interpretation: the loss is low when the model assigns high
probability to the correct token, and high when it assigns low probability.
The minimum possible loss is 0 (the model is 100% confident in the correct
answer).

For a full sequence, the loss is averaged over all positions:

```rust
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (b, s, v) = logits.dims3()?;
    let logits_flat = logits.reshape((b * s, v))?;
    let targets_flat = targets.reshape(b * s)?;
    candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)
}
```

### Gradient Verification

Given the candle-nn bugs described above, GESTALT includes a built-in
numerical gradient checker. This computes gradients two ways and compares:

1. **Autograd:** The standard backward pass through the computation graph
2. **Numerical:** Perturb each parameter by +epsilon and -epsilon, measure
   the loss change, compute (loss+ - loss-) / (2*epsilon)

If the relative error between these two exceeds a threshold, something is
wrong with the backward pass. After installing `GradRmsNorm` and
`grad_softmax_last_dim`, the maximum relative error dropped below 0.04 --
confirming that gradient flow is now correct through the entire model.

---

## 5. The Brain -- Where Everything Meets

The `Brain` struct is the top-level neural network that owns all components
and orchestrates their interaction. It is the central data structure of
GESTALT.

### Components

```rust
pub struct Brain {
    // --- Language Pipeline ---
    concept_encoder: WiredTransformer,    // goal -> concept_vec (no cross-attn)
    concept_projector: Linear,            // concept_vec -> 16 prefix embeddings
    memory_projector: Linear,             // memory vec -> cross-attn K/V input
    language_decoder: WiredTransformer,   // prefix + tokens -> next token logits
                                          // (with cross-attn for memory)
    memory_bank: MemoryBank,              // in-memory episodic store

    // --- Policy Pipeline ---
    policy_backbone: WiredTransformer,    // goal bytes -> hidden states (no cross-attn)
    head_intent: Linear,                  // hidden -> 16 intent classes
    head_actions: Linear,                 // hidden -> 6*16 action logits
    head_patterns: Linear,               // hidden -> 6*6 pattern logits
    head_files: Linear,                  // hidden -> 6*10 file logits
    head_picks: Linear,                  // hidden -> 6*129 argument logits
}
```

The Brain has two largely independent pipelines that share the concept
vector: the **language pipeline** (for generating natural language
responses) and the **policy pipeline** (for choosing actions and tools).
Only the language decoder has cross-attention enabled (for memory access).
The concept encoder and policy backbone use standard self-attention only.

### The Concept Encoder

The concept encoder takes tokenized input and produces a single concept
vector that summarizes the meaning of the entire input.

**Architecture:** A 1-layer WiredTransformer with TALK_VOCAB_SIZE=259
(256 bytes + PAD + BOS + EOS). The shallow depth is deliberate: deeper
encoders (4 layers) showed "depth-induced collapse" at initialization,
where all inputs produced nearly identical concept vectors (cosine
similarity 0.96). A single layer provides enough representational power
for the compression task without this pathological initialization behavior.

**Mean Pooling:** After the transformer produces hidden states for each
position, the concept vector is computed by averaging over all non-PAD
positions:

```rust
pub fn encode_concept(&self, goal_ids: &Tensor) -> Result<Tensor> {
    let hidden = self.concept_encoder.encode(goal_ids)?;    // (B, S, d_model)

    // Build mask: 1 for real tokens, 0 for PAD
    let pad_val = Tensor::new(&[TOK_PAD], goal_ids.device())?;
    let is_pad = goal_ids.broadcast_eq(&pad_val)?;
    let mask = is_pad.to_dtype(DType::F32)?.neg()?.affine(1.0, 1.0)?;

    // Masked mean
    let mask_3d = mask.unsqueeze(2)?;                       // (B, S, 1)
    let masked_hidden = hidden.broadcast_mul(&mask_3d)?;
    let sum = masked_hidden.sum(1)?;                        // (B, d_model)
    let count = mask.sum(1)?.unsqueeze(1)?.clamp(1.0, f64::MAX)?;
    sum.broadcast_div(&count)
}
```

Mean pooling was chosen over the common alternative of using the last
token's hidden state. The reason: with left-padded input, the last token
is always EOS at the same position. All inputs ended with the same token
at the same position, so last-token pooling produced nearly identical
vectors for every input. Mean pooling aggregates information from all
real tokens, producing properly discriminative vectors.

### The Concept Projector

The concept vector is a single 512-dimensional vector. The language decoder
expects a sequence of embedding vectors as input (to serve as a "prompt"
for generation). The concept projector bridges this gap:

```
concept_vec: (batch, 512)
    |
    v
Linear(512, 16*512 = 8192)
    |
    v
reshape: (batch, 16, 512)    -- 16 prefix embedding vectors
```

Each of these 16 vectors becomes a "concept token" that the decoder can
attend to. Think of it as the model explaining the concept to itself in
16 different ways, giving the decoder rich access to the encoded meaning.

The number 16 was chosen as a balance: fewer tokens lose too much
information from the bottleneck; more tokens dilute the signal and
increase computation. In practice, the decoder produces coherent responses
conditioned on these 16 prefix tokens, confirming that the bottleneck
preserves enough information.

### The Memory Projector

When previous interactions are retrieved from the memory bank, their
concept vectors are projected through a separate linear layer before
being passed to the decoder's cross-attention:

```
memory_vecs: (batch, K, 512)     K retrieved memory concept vectors
    |
    v
Linear(512, 512)                  per-vector projection
    |
    v
projected_mem: (batch, K, 512)   K projected memory vectors
    |
    v
[passed to cross-attention as K/V source in every decoder layer]
```

In the original design, memory was concatenated into the prefix alongside
concept tokens. The current architecture separates these two pathways:

- **Concept prefix** enters through self-attention (prepended to text tokens)
- **Memory** enters through cross-attention (separate K/V pathway per layer)

This separation means the self-attention prefix is smaller (16 tokens for
concepts only, not 24), and memory gets dedicated attention weights that
can specialize in memory retrieval rather than sharing bandwidth with
text generation.

### The Language Decoder

The language decoder is a 4-layer WiredTransformer with cross-attention
enabled that generates text one token at a time. It is conditioned on the
concept prefix (through self-attention) and optionally on external memory
(through cross-attention). It uses the same base architecture as the
encoder but with more layers, a causal attention mask, cross-attention
layers, and YaRN-aware RoPE.

**Forward pass with prefix and memory:**

```rust
pub fn forward_with_prefix_t(
    &self, prefix: &Tensor, input_ids: &Tensor, train: bool, memory: Option<&Tensor>
) {
    let tok_embs = self.tok_emb.forward(input_ids)?;  // embed tokens
    let x = Tensor::cat(&[prefix, &tok_embs], 1)?;     // prepend concept prefix
    let n_prefix = prefix.dim(1)?;

    for layer in &self.layers {
        x = layer.forward(
            &x, cos, sin, causal_mask, train, n_prefix, self.attn_temp, memory
        )?;
    }
    x = self.final_norm.forward(&x)?;

    // Return logits only for token positions (not prefix positions)
    let token_hidden = x.narrow(1, n_prefix, seq_len)?;
    self.lm_head.forward(&token_hidden)
}
```

Key details:

- **Logits are only computed for token positions**, not prefix positions.
  The prefix serves as conditioning context; we do not need to predict
  what comes after a concept token.

- **`memory` is passed to every layer's cross-attention.** If memory is
  `None` (no memories available), the cross-attention sub-layer is skipped
  and the block behaves like a standard self-attention + MLP block.

- **`n_prefix` is threaded to every layer** for RoPE prefix decoupling.
  Prefix tokens get no positional encoding; text tokens start at position 0.

- **`attn_temp` is threaded to every layer** for YaRN attention temperature
  compensation when processing sequences longer than the training window.

### The Generation Loop

At inference time, the decoder generates text autoregressively -- one
token at a time, using the KV cache for efficiency and cross-attention
for memory access.

```
1. ENCODE: goal -> concept_vec
2. BUILD PREFIX: concept_vec -> (1, 16, 512)     [concept only, no memory]
3. PROJECT MEMORY: retrieve top-K memories -> project -> (1, K, 512)
4. PREFILL: [prefix + BOS] through decoder with memory cross-attn, cache K/V
5. SAMPLE first token from logits
6. LOOP:
   a. Process new token through decoder (single-token step, cached, with memory)
   b. Get logits for next position
   c. Apply sampling: temperature -> top-k -> top-p -> n-gram penalty
   d. Sample next token
   e. If EOS, stop
7. DECODE: token IDs -> text via ConceptTokenizer
```

The memory tensor is projected once (step 3) and passed to every decoder
call (steps 4 and 6a). It does not enter the prefix or the KV cache --
it is a constant external input to cross-attention at every layer.

#### Sampling Strategy

Raw model logits are transformed through several sampling stages:

**Temperature scaling:** Divide logits by temperature T before softmax.
T < 1 makes the distribution sharper (more deterministic), T > 1 makes it
flatter (more random). Default T=0.8.

```
scaled_logit = logit / temperature
```

**Top-K filtering:** Zero out all logits except the K highest. Default K=40.
This prevents sampling from the long tail of unlikely tokens.

**Top-P (nucleus) sampling:** After top-K, sort remaining tokens by
probability and include only enough to cover probability mass P. Default
P=0.9. This adapts the number of candidates: for confident positions
(one dominant token), very few candidates; for uncertain positions, more.

**N-gram repetition penalty:** Penalize any token that would create a
repeated N-gram (default N=4, penalty=2.0). This is specifically designed
for byte-level and sub-word vocabularies where individual tokens naturally
repeat (the letter 'e' appears many times in any English text). A naive
token-level repetition penalty (as used in GPT-2) would destroy coherent
output at this tokenization granularity, because it penalizes every
repeated byte. The N-gram approach only penalizes actual repetitive
sequences like "light-light-light."

**Critical ordering:** Temperature must be applied FIRST, because it changes
the probability distribution shape that subsequent filters (top-K, top-P)
operate on. Applying top-P at T=1.0 and then sampling at T=0.5 produces
a different (worse) nucleus than applying temperature first.

### Byte-Level Tokenization

The encoder side uses a `TalkTokenizer` that operates at the byte level:
each byte of UTF-8 text becomes one token. Special tokens: PAD=256,
BOS=257, EOS=258. Total vocab: 259.

```
"hello" -> [BOS, 104, 101, 108, 108, 111, EOS]
         = [257, 'h', 'e',  'l', 'l',  'o', 258]
```

The encoder always uses byte-level tokenization. The decoder can use either
byte-level or concept-level tokenization (with BPE merges), depending on
the ConceptTokenizer configuration. More on this in Section 6.

---

## 6. The Tokenizer -- From Bytes to Meaning

GESTALT has three tokenizers, each serving a different purpose:

### TalkTokenizer (Byte-Level)

The simplest possible tokenizer: each byte of UTF-8 text becomes one token.
Three special tokens bring the vocab to 259.

- **PAD (256):** padding for fixed-length sequences
- **BOS (257):** beginning of sequence
- **EOS (258):** end of sequence

This is used by the concept encoder (always) and the language decoder
(when no concept merges are active).

**Advantage:** Zero information loss. Every possible text is representable.
**Disadvantage:** Long sequences. "Hello, world!" is 13 tokens. A 200-word
response might be 1,200 tokens, exceeding the decoder's 256-position
sequence length.

### ConceptTokenizer (BPE with Concept-Space Discovery)

To address the sequence length problem, GESTALT learns to compress frequent
byte patterns into single tokens. This is a variant of Byte Pair Encoding
(BPE), but with a twist: instead of merging purely by statistical frequency,
the merges are designed to capture patterns that are semantically meaningful
to the concept encoder.

**Base vocabulary:** The same 259 tokens as TalkTokenizer (256 bytes + PAD
+ BOS + EOS).

**Merge tokens:** Additional tokens starting at ID 259, each representing a
common byte pattern. With 200 merges, the total vocabulary is 459 tokens.

#### How Merges Are Discovered

```
1. SCAN CORPUS
   For each text in the corpus, extract all byte n-grams (length 2-8).
   Count the frequency of each unique n-gram.

2. SCORE
   For each n-gram with frequency >= min_frequency:
     score = frequency * (pattern_length - 1)
   The (length - 1) factor measures compression: a 4-byte pattern merged
   into 1 token saves 3 tokens per occurrence.

3. RANK AND SELECT
   Sort by score descending. Take the top max_merges patterns.
   Assign token IDs starting at 259.
```

Example merges discovered from the JARVIS dialogue corpus:

| Rank | Pattern | Score | Token ID |
|------|---------|-------|----------|
| 1    | " the " | 14,280 | 259 |
| 2    | "ing "  | 11,847 | 260 |
| 3    | " I "   | 10,521 | 261 |
| 4    | "tion"  | 9,836  | 262 |
| 5    | " you " | 8,744  | 263 |

The top merges are common English function words and suffixes -- exactly
the patterns that appear most frequently and benefit most from compression.

#### Encoding with Merges

Encoding uses greedy longest-match. Merges are sorted by pattern length
descending:

```rust
pub fn encode(&self, s: &str) -> Vec<u32> {
    let mut ids = vec![CONCEPT_TOK_BOS];
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let mut matched = false;
        for rule in &self.merges {      // sorted longest first
            if bytes[i..].starts_with(&rule.pattern) {
                ids.push(rule.token_id);
                i += rule.pattern.len();
                matched = true;
                break;
            }
        }
        if !matched {
            ids.push(bytes[i] as u32);  // fallback to raw byte
            i += 1;
        }
    }
    ids.push(CONCEPT_TOK_EOS);
    ids
}
```

**Lossless:** If a byte pattern has no matching merge, it falls back to
the raw byte. This means every possible input can be encoded, even if it
contains patterns never seen in the training corpus.

#### The Sweet Spot: 200 Merges

Extensive grid search revealed a clear sweet spot:

| Merges | Vocab Size | Compression | Val Loss @ 300 steps |
|--------|-----------|-------------|---------------------|
| 50     | 309       | ~1.3x       | 2.79                |
| 100    | 359       | ~1.5x       | 3.13                |
| **200**| **459**   | **~2.0x**   | **3.56**            |
| 500    | 759       | ~2.5x       | 4.08                |
| 1000   | 1259      | ~3.0x       | 4.70                |
| 2000   | 2259      | ~4.0x       | 5.53                |

The relationship is clear: fewer merges = faster convergence. At 200
merges, the vocabulary is small enough for the model to learn each token's
meaning quickly, while achieving ~2x compression (halving sequence lengths).

Below 150 merges, compression approaches byte-level (diminishing returns).
Above 500 merges, the model struggles to learn the meaning of so many
tokens with limited training data.

The curve flattens between 180-220 merges -- there is a natural knee
where additional merges provide less incremental compression.

#### Concept-Aware Merges (Future)

The current merge discovery uses frequency times compression as the score.
The original design intention was to score merges by *concept-space
consistency*: compute the concept vector for each occurrence of a byte
pattern, then measure how similar those vectors are. Patterns that always
mean the same thing (high consistency) would be preferred over patterns
that appear in varied contexts (low consistency).

This is currently disabled because the context-free encoding makes every
occurrence produce the same vector (consistency = 1.0 for everything).
When context-aware encoding is added in a future phase, the concept_fn
parameter will be re-enabled and merge discovery will become semantically
guided rather than purely statistical.

### PlanTokenizer (Structured Plan Vocabulary)

The planner uses a completely different tokenizer with a fixed 373-token
vocabulary designed for structured tool plans:

- **Structural tokens:** PAD, BOS, UNK, newline, EOP, STEP, Goal, Plan
- **Action tokens (15):** TALK, CARGOTEST, FIXTESTS, RG, REPOREAD, etc.
- **Parameter tokens:** PAT0-PAT5, FILE0-FILE9, PICK0-PICK128, FROM0-FROM7
- **Character tokens:** ASCII 32-126 (for inline text arguments)
- **Word tokens:** common words (hello, test, search, file, etc.)
- **Extended tokens:** TEXT/ENDTEXT, THINK/ENDTHINK, BECAUSE, THEN, etc.

This vocabulary is entirely static -- no learning, no merges. It exists to
give the FSM-constrained decoder a clean, finite set of symbols to work with.

---

## 7. The Memory System -- Episodic Recall

GESTALT has two memory systems: an in-memory bank for fast access during
a session, and a SQLite-backed persistent store that survives restarts.

### In-Memory MemoryBank

The `MemoryBank` in brain.rs provides fast, in-process memory:

```rust
struct MemoryEntry {
    concept_vec: Vec<f32>,    // 512-dimensional concept vector
    response: String,          // the generated response text
}

pub struct MemoryBank {
    entries: Vec<MemoryEntry>,
    capacity: usize,           // default: 1024
}
```

**Storage:** When the brain processes a dialogue, the concept vector and
response are stored as a new entry. If the bank is at capacity, the oldest
entry is removed (FIFO eviction).

**Retrieval:** Given a query concept vector, find the K most similar stored
vectors by cosine similarity:

```
cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

Where `dot(a, b) = sum(a_i * b_i)` and `||a|| = sqrt(sum(a_i^2))`.

The top-K vectors are returned for projection and use as cross-attention
K/V inputs to the decoder. Default K=8.

**Why cosine similarity?** The concept encoder produces vectors of varying
magnitudes. Cosine similarity measures directional similarity, ignoring
magnitude. Two concepts that "point in the same direction" in the 512-
dimensional space are semantically similar, regardless of their vector
lengths.

### Persistent EpisodicMemory (SQLite)

The `EpisodicMemory` in memory.rs provides disk-backed persistence:

```sql
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    concept_vec BLOB NOT NULL,
    goal TEXT NOT NULL,
    response TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 1
);
```

**Concept vectors as BLOBs:** The 512-dimensional f32 vector is stored as
raw bytes (512 * 4 = 2,048 bytes per entry). Conversion:

```rust
fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
```

**Retrieval:** Same cosine similarity approach as the in-memory bank, but
loads all entries from SQLite first. This is O(N) in the number of stored
episodes -- acceptable for the current capacity (1024 entries), but would
need an approximate nearest neighbor index for larger scales.

**Consolidation:** Over time, the memory accumulates similar entries (e.g.,
many greetings that produce near-identical concept vectors). The
`consolidate()` method merges these:

```
For each pair of entries (i, j) where i < j:
    if cosine_sim(concept_vec_i, concept_vec_j) > threshold:
        delete the older entry (i)
        keep the newer entry (j)
```

This reduces storage without losing coverage of distinct concepts.

### How Memories Reach the Decoder

During generation, the memory retrieval and projection pipeline works as:

```
1. Encode the current goal into a concept_vec
2. Query the memory bank: top-K by cosine similarity
3. Stack the K concept vectors into a tensor: (1, K, d_model)
4. Project through memory_projector: Linear(512, 512) per vector
5. Build concept-only prefix: (1, 16, d_model) [no memory in prefix]
6. Pass prefix to decoder via self-attention prepend
7. Pass projected memory to decoder via cross-attention in every layer
```

This architecture gives memory its own dedicated pathway. The decoder's
self-attention processes the concept prefix alongside generated text
(deciding what to say based on the current prompt), while cross-attention
in every layer independently queries the memory (deciding how past
experience should influence the response).

The memory projector learns a transformation from stored concept vectors
into a representation space that the cross-attention layers can
efficiently query. This is distinct from the concept projector, which
maps the current concept vector into prefix embeddings for self-attention.
The two projectors learn different tasks: the concept projector learns
"how to explain the current prompt to the decoder," while the memory
projector learns "how to make past experiences accessible to cross-attention."

### Training With Memory From Day One

A critical lesson from V4 (mistake M-004): the decoder was initially
trained WITHOUT memory context, then memory was added at inference time.
The result was catastrophic -- the decoder had never learned to attend
to memory vectors, so they were treated as random noise that corrupted
generation.

GESTALT V5 trains the decoder with memory from the very first training
step. During SFT, a memory pool accumulates concept vectors from processed
batches. After a warmup period (500 steps), each training step has a 50%
chance of including randomly sampled memory vectors as prefix context:

```rust
let use_mem = step >= memory_warmup
    && memory_pool.len() >= memory_k
    && rng.next_f64() < memory_prob;

let mem_tensor = if use_mem {
    // Sample K random concept vectors from the pool
    // ...
    Some(tensor)
} else {
    None
};

let logits = brain.forward_from_concept_t(
    &concept_vec, &input_t, mem_tensor.as_ref(), true
)?;
```

Inside `forward_from_concept_t`, the memory is routed through cross-
attention, not the prefix. The concept-only prefix is built from
`concept_vec`, and memory (if present) is projected and passed as the
cross-attention K/V source:

```rust
pub fn forward_from_concept_t(&self, concept_vec, response_ids, memory_vecs, train) {
    let prefix = self.build_prefix(concept_vec, None)?;  // concept-only prefix
    let memory = match memory_vecs {
        Some(mem) => Some(self.memory_projector.forward(mem)?),
        None => None,
    };
    self.language_decoder.forward_with_prefix_t(&prefix, response_ids, train, memory.as_ref())
}
```

This means the decoder learns two things simultaneously:
1. How to generate responses conditioned on concept prefix alone (self-attention)
2. How to use (and when to ignore) external memory (cross-attention)

---

## 8. The Planner -- FSM-Constrained Decoding

The planner generates structured tool execution plans. Unlike the language
decoder (which produces free-form text), the planner must produce
syntactically valid plans that conform to a specific grammar. GESTALT
achieves this through a finite state machine (FSM) that masks the
transformer's output at each generation step.

### Why Constrained Decoding?

Consider a plan for "search for GPU policy and open it":

```
STEP RG PAT1 STEP REPOREAD FILE0 EOP
```

This means: Step 1 uses ripgrep with pattern PAT1, Step 2 reads FILE0
(determined by step 1's output), then end-of-plan.

An unconstrained language model might produce:

```
STEP STEP RG RG FILE0 TALK EOP STEP
```

This is syntactic nonsense -- two STEPs in a row, actions without steps,
end-of-plan followed by more tokens. Instead of hoping the model "learns"
correct syntax, GESTALT enforces it mechanically.

### The 17-State FSM

The FSM defines which tokens are legal at each point in the plan:

```
State: Start
  Legal tokens: STEP, EOP, THINK
  Transition: STEP -> AfterStep
              THINK -> InThink
              EOP -> done

State: AfterStep
  Legal tokens: [all 15 action tokens]
  Transition: RG -> AfterRg
              REPOREAD -> AfterRepoRead
              PATCHDRYRUN -> AfterPatch
              MEMADD -> AfterMemAdd
              MEMSEARCH -> AfterMemSearch
              PROVEALGEBRA -> AfterProve
              (others) -> AfterActionNoArgs

State: AfterActionNoArgs
  Legal tokens: STEP, EOP
  Transition: STEP -> AfterStep (new step)
              EOP -> done

State: AfterRg
  Legal tokens: PAT0-PAT5
  Transition: (any PAT) -> AfterActionNoArgs

State: AfterRepoRead
  Legal tokens: FILE0-FILE9, FROM0-FROM7
  Transition: FILE -> AfterActionNoArgs
              FROM -> AfterRepoReadFrom

State: AfterRepoReadFrom
  Legal tokens: PICK0-PICK128
  Transition: (any PICK) -> AfterActionNoArgs

State: InThink
  Legal tokens: ENDTHINK, NEEDS, THEN, IF, BECAUSE, [actions]
  Transition: ENDTHINK -> Start
              (others) -> InThink

... (and so on for all 17 states)
```

### How FSM Masking Works

At each generation step, the current FSM state determines which tokens
are legal. All other tokens receive -infinity logits:

```rust
fn apply_fsm_mask(logits: &Tensor, valid_ids: &[u32], vocab_size: usize, device: &Device) {
    let mut mask_data = vec![f32::NEG_INFINITY; vocab_size];
    for &id in valid_ids {
        mask_data[id as usize] = 0.0;      // keep these tokens
    }
    let mask = Tensor::from_vec(mask_data, (1, vocab_size), device)?;
    logits + mask
}
```

After adding the mask, softmax guarantees zero probability for all invalid
tokens. The model then picks the highest-scoring valid token using argmax
(greedy decoding -- sampling is not needed because plans are deterministic).

### The Decoding Loop

```rust
pub fn greedy_decode(model, tok, prompt_ids, max_tokens, device) {
    let mut context = tok.pad_or_truncate(prompt_ids, seq_len);
    let mut state = FsmState::Start;
    let mut steps_emitted = 0;

    for _ in 0..max_tokens {
        let logits = model.forward(&input)?;
        let last_logits = logits.i((0, seq_len - 1, ..))?;

        let valid = valid_tokens_for_state(state, tok, steps_emitted);
        let masked = apply_fsm_mask(&last_logits, &valid, vocab_size, device)?;

        let token_id = masked.argmax(D::Minus1)?;
        let token_str = tok.token(token_id);

        if token_str == "EOP" { break; }

        state = fsm_transition(state, &token_str, &mut steps_emitted);
        context.remove(0);         // slide window
        context.push(token_id);    // append new token
    }
}
```

### The Chain-of-Thought Extension

The FSM includes a THINK/ENDTHINK block that lets the model reason before
committing to a plan:

```
THINK NEEDS CARGOTEST BECAUSE RG THEN FIXTESTS ENDTHINK
STEP RG PAT1 STEP CARGOTEST STEP FIXTESTS EOP
```

The THINK block is optional (only emittable from the Start state, and only
on the first step). Inside THINK, the model can use reasoning keywords
(NEEDS, BECAUSE, THEN, IF) and reference action tokens. The ENDTHINK
token returns to the Start state, where actual plan steps begin.

### Why This Matters

With FSM-constrained decoding, GESTALT achieves 21/21 on its plan benchmark
-- every goal produces a syntactically valid, semantically correct plan.
The constraint eliminates an entire class of failure modes (malformed plans)
and lets the transformer focus entirely on choosing the RIGHT actions,
not on producing valid syntax.

---

## 9. The Policy Heads -- Making Decisions

While the planner generates sequential plans, the policy heads make
immediate classification decisions: what is the user's intent, and what
actions should be taken?

### Architecture

The policy pipeline uses a separate transformer backbone (not shared with
the language pipeline) that encodes the goal text into hidden states, then
applies mean pooling and five classification heads:

```
goal_bytes: (batch, seq_len)           raw bytes, 0-255 (NO special tokens)
    |
policy_backbone: WiredTransformer       d=256, 3 layers, 8 heads
    |
hidden: (batch, seq_len, 256)
    |
mean_pool: (batch, 256)                average over all positions
    |
    +-> head_intent:  Linear(256, 16)               -> intent class
    +-> head_actions: Linear(256, 6*16) -> reshape   -> 6 action slots
    +-> head_patterns: Linear(256, 6*6) -> reshape   -> 6 pattern slots
    +-> head_files:   Linear(256, 6*10) -> reshape   -> 6 file slots
    +-> head_picks:   Linear(256, 6*129) -> reshape  -> 6 argument slots
```

**Important encoding difference:** The policy backbone uses raw byte
encoding (0-255, vocab size 256), NOT the TalkTokenizer (which adds PAD,
BOS, EOS for vocab 259). This is a separate encoding path that caused a
real bug (M-026) when pipeline code accidentally used the wrong tokenizer.

### The Five Heads

**Intent (16 classes):** What is the user trying to do? Run tests, greet,
search code, fix tests, add memory, prove algebra, etc.

**Actions (6 slots x 16 tools):** For each of up to 6 steps, which tool
should be used? Tools include: TALK, CARGOTEST, FIXTESTS, RG, REPOREAD,
PATCHDRYRUN, PROVEALGEBRA, MEMADD, MEMSEARCH, etc. Slot 0 is always
ACT_END for single-action intents.

**Patterns (6 slots x 6):** For tools that take pattern arguments (like
ripgrep), which pattern index? PAT0-PAT5.

**Files (6 slots x 10):** For tools that take file arguments, which file
index? FILE0-FILE9.

**Picks (6 slots x 129):** For tools that take general arguments, which
argument index? PICK0-PICK128.

### Training the Policy

Policy training uses a curriculum of 64 goal-action pairs. Each pair maps
a natural language goal to the correct intent, action sequence, and
arguments:

```
"hello" -> intent=HELLO, actions=[TALK, END, ...], patterns=[0,0,...], ...
"run the tests" -> intent=RUN_TESTS, actions=[CARGOTEST, END, ...], ...
"search jarviscmd then open it" -> intent=COMPOSITE,
    actions=[RG, REPOREAD, END, ...], patterns=[PAT1, ...], files=[FILE0, ...]
```

The loss is a weighted sum of cross-entropy losses over all five heads:

```
total_loss = 2.0 * CE(intent) + 2.0 * CE(actions) + 2.0 * CE(patterns)
           + 2.0 * CE(files) + 1.0 * CE(picks)
```

Intent and actions are weighted higher because they are the most critical
decisions. Pick accuracy is less important for overall system behavior.

---

## 10. Training -- Teaching the Brain

Training in GESTALT is a multi-phase pipeline. Each phase trains a
different component, and the phases are largely independent.

### Phase 1: Brain SFT (Supervised Fine-Tuning)

This is the primary training phase for the language model. The brain
learns to generate JARVIS-style responses given prompts.

#### The Training Corpus

The training corpus contains 92,273 unique (prompt, response) pairs drawn
from six sources, all revoiced through a JARVIS personality pipeline:

| Source | Pairs | Description |
|--------|-------|-------------|
| SlimOrca | 29,000 | Diverse instruction-following |
| UltraChat | 25,000 | Multi-turn conversations |
| Handwritten | 21,000 | Original JARVIS dialogue |
| WizardLM | 10,000 | Complex instructions |
| No Robots | 3,500 | Human-written responses |
| Alpaca + OASST2 + Dolly | 3,200 | Mixed instruction data |

The corpus is embedded at compile time via `include_str!` and parsed
from JSON on first access. This means the training data is baked into
the binary -- no external data files needed at runtime.

#### Train/Validation Split

The corpus is split 90/10 into training and validation sets using a
seeded Fisher-Yates shuffle (seed=42 for reproducibility). The validation
set is used for early stopping and never appears during training.

#### The SFT Loop

```
For each step in 0..sft_steps:
    1. Sample a minibatch of batch_size dialogues
    2. For each dialogue:
       a. Encode prompt with TalkTokenizer -> left-padded goal_ids
       b. Encode response with ConceptTokenizer -> right-padded response_ids
       c. Apply PAD-denoising: randomly replace tokens with PAD (10% rate)
    3. Forward pass:
       a. Encode goal_ids -> concept_vec (encoder)
       b. Maybe sample memory vectors from memory pool
       c. Build prefix from concept_vec + optional memory
       d. Decode prefix + response_ids -> logits
    4. Compute weighted cross-entropy loss (weight=0 for PAD positions)
    5. Gradient accumulation: accumulate micro-batch gradients
    6. Optimizer step (every accum_steps micro-batches):
       a. AdamW with cosine LR schedule
       b. Learning rate: warmup from 0 to 3e-4 over 10% of training,
          then cosine decay to 3e-6
    7. Every 1000 steps: compute validation loss, check early stopping
```

#### PAD-Denoising: Forcing Cross-Attention

A critical training technique: during each step, 10% of non-special
tokens in the response are randomly replaced with PAD (the padding token).
This forces the decoder to attend to the concept prefix tokens, because
the local autoregressive context has been partially destroyed.

Without denoising, the decoder learns to predict the next byte purely
from the preceding bytes (bigram/trigram statistics), ignoring the prefix
entirely. The concept encoder receives no gradient signal, and the model
produces the same generic response regardless of the prompt. This failure
mode was observed in early training (M-029): loss plateaued at 2.15 --
exactly matching an unconditional bigram model.

PAD-denoising introduces information gaps that can only be filled by
attending to the prefix. The decoder learns: "when the local context is
degraded, look at the concept tokens to figure out what you should be
saying."

#### Cosine Learning Rate Schedule

The learning rate follows a cosine curve with linear warmup:

```
if step < warmup_steps:
    lr = base_lr * (step + 1) / warmup_steps        # linear warmup
else:
    progress = (step - warmup_steps) / remaining_steps
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
```

This starts at zero, ramps linearly to base_lr over the warmup period
(10% of training), then smoothly decays following a cosine curve. The
cosine schedule provides a natural annealing that spends more time at
moderate learning rates (the productive regime) and less time at very
high or very low rates.

#### Early Stopping

GESTALT uses patience-based early stopping on validation loss:

```
If no new best val_loss for patience consecutive checks:
    Stop training, restore best checkpoint weights
```

The default patience is 20 checks. Each check happens at the logging
interval (every 1000 steps). So training stops if no improvement is
seen for 20,000 steps -- generous enough to avoid premature stopping
on noisy validation curves.

When a new best val_loss is found, the model weights are saved to
`checkpoints/brain_best_sft.safetensors`. After training completes (or
early stops), these best weights are loaded back:

```rust
if std::path::Path::new("checkpoints/brain_best_sft.safetensors").exists() {
    load_checkpoint(&varmap, "checkpoints/brain_best_sft.safetensors", device)?;
}
```

This ensures the model returned by `train_brain_talk()` always has the
best validation loss weights, not the weights from the last training step
(which may have overfit past the optimum).

#### Gradient Accumulation

At d_model=512 with batch_size=48, training fits comfortably in 16GB VRAM.
But at d_model=1024 (Phase 2), even batch_size=2 approaches the memory
limit.

Gradient accumulation simulates larger batches by processing multiple
micro-batches and accumulating their gradients before taking a single
optimizer step:

```
effective_batch_size = micro_batch_size * accum_steps
```

For Phase 2: micro_batch=2, accum_steps=16, effective_batch=32.

The implementation properly scales the loss by 1/accum_steps so that
accumulated gradients are averaged, not summed:

```rust
pub fn accumulate_and_step(&mut self, loss: &Tensor) -> Result<Option<usize>> {
    let scaled = if self.config.grad_accum_steps > 1 {
        (loss / self.config.grad_accum_steps as f64)?
    } else {
        loss.clone()
    };
    let grads = scaled.backward()?;

    // Add to accumulated gradients
    match &mut self.accum_grads {
        None => { self.accum_grads = Some(grads); }
        Some(accum) => {
            for var in self.varmap.all_vars() {
                if let Some(new_g) = grads.get(var.as_tensor()) {
                    let sum = match accum.get(var.as_tensor()) {
                        Some(existing) => (existing + new_g)?,
                        None => new_g.clone(),
                    };
                    accum.insert(var.as_tensor(), sum);
                }
            }
        }
    }

    self.accum_count += 1;
    if self.accum_count >= self.config.grad_accum_steps {
        // Step optimizer with averaged gradients
        self.optimizer.step(&self.accum_grads.take().unwrap())?;
        self.accum_count = 0;
        // Update learning rate
        let new_lr = self.scheduler.step();
        self.optimizer.set_learning_rate(new_lr);
    }
}
```

**A previous bug (fixed):** The original implementation called
`backward_step` per micro-batch, which performed N optimizer steps instead
of 1. This meant the learning rate was effectively N times higher than
intended, causing training instability.

### Phase 2: Concept Tokenizer Bootstrap

After brain training, the concept tokenizer is bootstrapped from the
corpus:

```
1. Scan all corpus text for byte n-grams (2-8 bytes)
2. Filter by minimum frequency (default: 5 occurrences)
3. Score: frequency * (pattern_length - 1)
4. Take top-N by score (default: 200 merges)
5. Assign token IDs starting at 259
6. Save to checkpoints/concept_tokenizer.bin
```

This produces a ConceptTokenizer with ~459 vocab that achieves ~2x
compression. The decoder is then re-trained with this tokenizer, allowing
it to process longer responses within the same sequence length.

### Phase 3: Plan-LM Training

The planner is trained with standard supervised fine-tuning on 21 goal-plan
pairs. Each goal maps to a reference plan:

```
"hello" -> [STEP, TALK, EOP]
"run the tests" -> [STEP, CARGOTEST, EOP]
"search jarviscmd and then open it" -> [STEP, RG, PAT1, STEP, REPOREAD, FILE0, EOP]
```

The loss uses per-position weighting: STEP tokens are weighted at 0.1
(they are always at predictable positions, so the model learns them
trivially) and action tokens at 1.0 (these are the interesting decisions).

Without this weighting, STEP tokens dominate the gradient signal (M-002):
they represent 27% of all targets and are easy to predict, so the optimizer
spends all its effort perfecting STEP prediction while action tokens remain
at random chance (0.7% probability).

### Phase 4: Policy Training

Policy training uses the 64-task curriculum described in Section 9. It
runs for 16,384 steps with a separate transformer backbone (d=256, 3
layers) and AdamW optimizer.

### The Training Pipeline

The `gestalt train` command runs all phases in sequence:

```
1. Build ConceptTokenizer (or load from checkpoint)
2. Brain SFT training (encoder + decoder + projector)
3. Bootstrap concept tokenizer from trained encoder
4. Plan-LM SFT training (planner)
5. Policy training (classification heads)
6. Run gallery evaluation (comprehensive generation test)
7. Save checkpoints
```

The `--resume` flag skips brain SFT and loads from checkpoint, going
straight to tokenizer/planner/policy training. This saves ~90 minutes
when iterating on non-brain components.

---

## 11. What We Learned -- Findings From v18 Through v23

### v18: Memorization Confirmed (1,749 pairs)

**Result:** val_loss=2.02, coherent outputs are verbatim corpus entries

With only 1,749 training pairs and d_model=512, the model has roughly
3 dimensions per example -- more than enough to memorize every sample.
Coherent outputs were exact copies from training data; novel prompts
produced gibberish.

**Lesson:** The architecture works. The model CAN learn to generate
coherent text from concept vectors. But 1,749 pairs is not enough data
for generalization.

### v19: More Data, No Dropout (21K pairs)

**Result:** best val_loss=3.55 at step 6K, 0% coherent gallery

Scaling to 21,000 pairs with 2,000 BPE merges. The model began overfitting
at step 6,000. Zero coherent outputs in the evaluation gallery.

**Lesson:** Larger vocabulary (2,259 tokens) is harder to learn. Overfitting
begins early without regularization.

### v20: The Dropout Discovery (21K pairs)

**Result:** best val_loss=3.26 at step 8K, ~30% coherent

Adding dropout=0.1 extended training life from 6K (v19) to 13K+ steps
before overfitting. This single change brought coherence from 0% to ~30%.

**Lesson:** Dropout is essential for small models on limited data. It
prevents the model from memorizing training examples and forces it to
learn general patterns.

### v22: The Vocabulary Breakthrough (21K pairs, 200 merges)

**Mock hero result:** val_loss=2.04 at 5K steps, grammar breakthrough
**Hero result:** val_loss=1.9 at 30K steps, ~40-45% coherent

The key change: reducing BPE merges from 2,000 to 200 (vocab from 2,259
to 459). This was the single most impactful hyperparameter change in the
entire project.

**Why it works:** With 2,259 tokens, many merge tokens appear only a few
times in the corpus. The model cannot learn a reliable embedding for a
token it has seen 3 times. With 459 tokens, every token appears frequently
enough for the model to learn what it means.

The grid search results were dramatic: at 300 steps, merges=200 had
val_loss=3.56 vs merges=2000 at val_loss=5.53. The difference grew
larger with more training.

### v23: Scaling to 92K Pairs

**Attempt 1:** Crashed WSL at step 11K (OOM from Claude Code memory bloat,
not the training binary). val_loss=1.88 before crash. Gallery showed fluent
grammar but semantic disconnection from prompts -- "generic AI assistant"
mode.

**Key finding:** val_loss=1.88 on 92K mixed-source data is NOT equivalent
to val_loss=1.9 on 21K curated data. The mixed corpus includes diverse
instruction-following data that dilutes the JARVIS personality. Corpus
quality matters more than quantity.

**Attempt 2:** Restarted with the same configuration, ongoing.

### Cross-Cutting Findings

**Fewer BPE merges = faster convergence.** This was the single most
important finding. It contradicts the common ML wisdom that larger
vocabularies are better -- that wisdom applies to models trained on
billions of tokens, not thousands.

```
At 300 training steps, validation loss by merge count:
  50 merges:   2.79    (barely compresses, fast to learn)
  200 merges:  3.56    (2x compression, fast to learn)  <-- sweet spot
  2000 merges: 5.53    (4x compression, too many tokens to learn)
```

**Dropout extends training life by 2x.** Dropout=0.1 is not optional for
models of this scale.

**Mean pooling over last-token pooling.** Last-token pooling produced
identical concept vectors because all inputs ended with EOS at the same
position.

**Right-padding for decoders, left-padding for encoders.** The decoder
generates text starting at BOS (position 0), so BOS must always be at
position 0. Padding goes at the end. The encoder just needs the content
together; padding goes at the start. Mixing these up causes RoPE position
mismatches between training and inference (M-003).

**PAD-denoising at 10% is necessary.** Below 10%, the decoder ignores the
prefix and becomes an unconditional bigram model (M-029). Above 20%, too
much information is destroyed and training becomes inefficient.

**Gradient flow must be verified BEFORE training.** Twelve training runs
(M-031, M-032) failed because candle-nn's RmsNorm and softmax broke
gradient flow. One gradient check test would have caught this immediately.

**Repetition penalty must match tokenization granularity.** Token-level
repetition penalty (as in GPT-2) destroys byte-level output because
individual bytes like 'e' naturally repeat. N-gram penalty (4-gram) is
the correct approach for byte-level vocabularies (M-044).

**Sampling operation order matters.** Temperature must be applied before
top-K and top-P filtering, because it changes the shape of the probability
distribution that those filters operate on. Applying top-P at T=1.0 and
then sampling at T=0.5 produces a different (looser) nucleus than applying
temperature first. The correct order is: temperature -> top-K -> top-P ->
sample (M-046).

**Always save best checkpoint weights, not last-step weights.** Early
stopping saves the best-validation-loss weights incrementally, but the
training function returned the model with last-step weights in memory.
If training overshot past the optimum, the model returned was worse than
the best checkpoint on disk. The fix: reload the best checkpoint after
training exits, before any downstream evaluation (M-045).

**Training an architecture you know is wrong wastes compute.** When an
audit or design review reveals the final architecture (cross-attention for
memory, YaRN for context extension, KV cache for generation), implement
those changes before launching long training runs. Training on an
architecture that is about to be replaced is wasted GPU time. Quick
validation runs (500 steps) on non-final architecture are fine; full 30K+
step runs are not (M-048).

**DA (denoising autoencoder) fine-tuning at high learning rate destroys
coherence.** A DA phase that trains on isolated byte positions at lr=1e-4
catastrophically interfered with the autoregressive flow that SFT
established. The model became good at predicting individual bytes but
lost sequence-level coherence. SFT-only training (with early stopping)
is the proven-optimal approach for GESTALT's current scale (M-039).

---

## 12. What Comes Next

### Recently Implemented (Phase 2.5)

Several major architectural improvements have been completed and are
documented throughout this paper:

- **YaRN RoPE Scaling** (Section 4): Frequency-selective context extension
  with three bands (high/mid/low) and attention temperature compensation.
  Enables the model to process sequences longer than its training window
  at inference time.

- **Cross-Attention Memory** (Section 4): Dedicated cross-attention layers
  in the decoder for querying external memory. Replaces the previous prefix
  concatenation approach. Memory now has its own attention pathway separate
  from text generation.

- **RoPE Prefix Decoupling** (Section 4): Prefix tokens (concept projections)
  receive no positional encoding. Only text tokens get RoPE, starting at
  position 0. This removes artificial positional dependencies between the
  concept projector and the decoder.

- **KV Cache with Memory Threading** (Section 4): Two-phase generation
  (prefill + step) with cached K/V tensors. Memory is threaded through
  every cached forward call for cross-attention access.

### Planned: Perceiver IO Encoder

The current concept encoder processes the full input sequence through
transformer attention (O(N^2) in sequence length). For very long inputs,
a Perceiver IO architecture would use a small fixed-size set of latent
vectors that cross-attend to the input:

```
Input: (1, 1000, 512)       long input sequence
Latents: (1, 16, 512)       fixed set of learned queries

Cross-attention: latents attend to input
  Q from latents (16 queries), K/V from input (1000 keys)
  Cost: O(16 * 1000) instead of O(1000 * 1000)
```

This would make the encoder cost independent of input length -- important
for scaling to multi-turn conversations where the context grows with each
turn.

### Planned: Fused Softmax Kernel

The current softmax implementation uses four separate CUDA kernel launches:
max, subtract, exp, sum, divide. A fused CUDA kernel would combine these
into a single launch, eliminating kernel launch overhead and intermediate
memory allocations. For the small sequence lengths in GESTALT, kernel
launch latency dominates actual computation time.

### Planned: Online Micro-Training

The long-term goal is for GESTALT to improve from every interaction:

```
After N successful interactions:
  1. Collect (prompt, response, reward) tuples
  2. Run 100-500 SFT steps on successful interactions
  3. Optionally: GRPO on preferred vs rejected responses
  4. Save updated checkpoint
```

This turns the deployed model into a continually learning system -- the
more you use it, the better it gets at YOUR specific tasks and preferences.

### Phase Roadmap

| Phase | Components | Status |
|-------|-----------|--------|
| 0 | Foundation: transformer, tokenizer, RoPE, causal mask | Complete |
| 1 | Pipeline: brain, policy, planner, executor | Complete |
| 2 | BPE tokenizer + Language model + Memory | Complete |
| 2.5 | KV cache, YaRN RoPE, cross-attention memory, prefix decoupling | Complete |
| 3 | ReAct loops + concept-space chain-of-thought | Planned |
| 4 | Experience buffer + online micro-training | Planned |
| 5 | Daemon mode + context monitoring + proactive suggestions | Planned |
| 6 | JARVIS personality training + unified server | Planned |

---

## 13. Glossary

**Attention:** The mechanism by which each position in a sequence gathers
information from all other positions. Uses Query, Key, Value projections.

**Attention temperature:** A scaling factor applied to attention logits
to compensate for entropy changes. In YaRN, equals sqrt(L_test / L_train)
when extending beyond the training context window.

**Autoregressive:** Generating one token at a time, where each token is
conditioned on all previous tokens.

**BOS (Beginning of Sequence):** Special token marking the start of text.

**BPE (Byte Pair Encoding):** A tokenization method that merges frequent
byte pairs into single tokens, compressing text.

**Causal mask:** A triangular matrix that prevents future positions from
being attended to, enforcing left-to-right generation.

**candle-rs:** Hugging Face's Rust tensor library. Provides CUDA-accelerated
tensor operations without PyTorch's dependency overhead.

**Concept bottleneck:** The architectural constraint that forces all
information through a fixed-size concept vector.

**Concept vector:** The d_model-dimensional (e.g. 512) vector produced by
the concept encoder. Summarizes the meaning of the input.

**Cross-attention:** An attention mechanism where Q comes from one source
(e.g. decoder hidden states) and K/V come from another (e.g. external
memory). Used to inject information from a separate modality without
concatenating it into the sequence.

**Cross-entropy:** The loss function for classification and language
modeling. Measures how far the model's predicted probabilities are from
the correct answer.

**d_ff:** Feed-forward intermediate dimension. Default: 2048 (4x d_model).

**d_model:** The dimension of all hidden states in the transformer.
Default: 512 (Phase 1) or 1024 (Phase 2).

**Dropout:** Regularization technique that randomly zeros elements during
training to prevent overfitting.

**Early stopping:** Halting training when validation loss stops improving,
to prevent overfitting.

**Embedding:** A lookup table mapping discrete token IDs to dense vectors.

**EOS (End of Sequence):** Special token marking the end of text.

**Episodic memory:** Memory system that stores specific past experiences
(concept vector + text) and retrieves them by similarity.

**FIFO eviction:** First In, First Out. When memory is full, the oldest
entry is removed.

**FSM (Finite State Machine):** A formal model of computation with a
finite set of states and transitions. Used to constrain the planner's
output.

**f32:** 32-bit floating point. The only precision supported by candle-rs
v0.8.4 for training.

**GELU:** Gaussian Error Linear Unit. A smooth activation function used
in transformer MLPs.

**Gradient accumulation:** Simulating larger batch sizes by accumulating
gradients over multiple micro-batches before updating weights.

**Head dimension:** The dimension of each attention head. Equals d_model
divided by n_heads. Default: 64.

**Intent:** The classified purpose of a user's goal (greet, search,
run tests, etc.).

**KV cache:** Cached Key and Value tensors from previous generation steps,
enabling O(1) per-token generation instead of O(N).

**Logits:** Raw (pre-softmax) scores output by the model. Each logit
corresponds to one token in the vocabulary.

**Mean pooling:** Averaging hidden states across the sequence dimension,
excluding padding positions.

**MLP (Multi-Layer Perceptron):** A feed-forward neural network with one
hidden layer and a nonlinear activation. Used per-position in transformers.

**n_heads:** Number of attention heads. Default: 8.

**NTK (Neural Tangent Kernel) scaling:** A method of adjusting the RoPE
frequency base to extend context length while preserving the relative
spacing between frequency dimensions.

**PAD:** Special padding token used to fill sequences to a fixed length.

**Prefix tokens:** Concept and memory vectors that are prepended to the
decoder's input sequence, providing conditioning context.

**Right-padding:** Placing PAD tokens at the end of the sequence. Used for
decoder sequences so BOS is always at position 0.

**Left-padding:** Placing PAD tokens at the start of the sequence. Used for
encoder sequences.

**RmsNorm (Root Mean Square Layer Normalization):** Normalizes by dividing
by the root mean square of the vector.

**RoPE (Rotary Position Embeddings):** Positional encoding that rotates
Q and K vectors, encoding relative positions through dot product properties.

**RoPE prefix decoupling:** The design decision to exclude prefix tokens
(concept projections) from RoPE rotation. Only text tokens receive
positional encoding.

**Safetensors:** A file format for storing neural network weights. Used by
candle-nn for checkpoint saving and loading.

**Softmax:** Converts a vector of real numbers into a probability
distribution. `softmax(z)_i = exp(z_i) / sum(exp(z_j))`.

**SFT (Supervised Fine-Tuning):** Training on (input, target) pairs where
the correct output is known.

**Temperature:** A parameter that controls the randomness of sampling.
Lower = more deterministic, higher = more random.

**Top-K:** A sampling strategy that considers only the K highest-probability
tokens.

**Top-P (nucleus sampling):** A sampling strategy that considers the smallest
set of tokens whose cumulative probability exceeds P.

**Transformer:** A neural network architecture based on self-attention,
used as the building block for all components in GESTALT.

**VarMap:** candle-nn's parameter container. Holds all trainable tensors
(Vars) and provides access to the optimizer.

**VRAM:** Video RAM. The GPU's dedicated memory. The RTX 5070 Ti has 16GB.

**YaRN (Yet another RoPE extensioN):** A method for extending a model's
effective context window beyond its training length by applying frequency-
selective scaling to RoPE dimensions. High-frequency dimensions are left
unchanged, low-frequency dimensions are NTK-scaled, and mid-frequency
dimensions receive a linear interpolation between the two.

---

*This document describes the GESTALT WIRED-V5 system as of Phase 2.5,
February 2026. Built on an RTX 5070 Ti with 16GB VRAM, running on
WSL2 under Windows 11.*
