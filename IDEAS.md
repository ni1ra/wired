# GESTALT WIRED-V6 — Ideas & Optimization Backlog

> Generated from 3 parallel deep-exploration agents (brain.rs, transformer.rs, generation pipeline).
> Ranked by impact × feasibility. Updated: 2026-02-27.

---

## Tier 1: High Impact, Low Risk (Implement Now)

### I-001: Cache L2 Norms in Memory Bank
**File:** `src/brain.rs:350-375` | **Impact:** 8-12% GPU util gain | **Effort:** 30 min
- `retrieve_vecs()` recomputes L2 norm for ALL memory entries on every call
- At memory_pool=5K vectors × d=384: 5000 norms recomputed per retrieval
- **Fix:** Add `vec_norm: f32` to MemoryEntry, compute once on `.store()`
- Also: replace O(N log N) full sort with partial_sort for top-K

### I-002: Fix Temperature/Filtering Order in Sampling
**File:** `src/brain.rs:1733-1738` | **Impact:** 5-10% coherence | **Effort:** 3 lines
- Temperature is applied AFTER top-K/top-P filtering (backwards)
- Temperature should shape the distribution FIRST, then filters select from it
- **Fix:** Move temperature scaling before `apply_top_k` and `apply_top_p` calls

### I-003: Token-Frequency Repetition Penalty
**File:** `src/brain.rs:1625-1653` | **Impact:** 15-25% less repetition | **Effort:** 10 lines
- Current n-gram penalty is too blunt for byte-level vocab (459 tokens)
- Natural repeats like "tion", "ing", "the" get incorrectly penalized
- **Fix:** Track token frequency during generation, apply: `logits[id] /= (1 + alpha * count[id])`
- Alpha=0.2 allows natural repetition while preventing loops

### I-004: Use Precomputed Causal Mask in forward_cached()
**File:** `src/transformer.rs:356-369` | **Impact:** 2-3% speedup | **Effort:** 10 min
- `forward_cached()` rebuilds causal mask every call (256×256 = 256KB per prefill)
- `forward()` already uses cached `self.causal_mask` correctly
- **Fix:** Pass precomputed mask to `forward_cached()`, narrow to (s, kv_len)

### I-005: Async Memory Pool Exports (Reduce GPU→CPU Stalls)
**File:** `src/brain.rs:1047-1058` | **Impact:** 3-5% GPU util | **Effort:** 1-2 hours
- `concept_vec.to_vec2()` forces synchronous GPU→CPU copy every training step
- 100M steps = 100M synchronous transfers (each stalls GPU pipeline)
- **Fix:** Only export every 100 steps instead of every step. Slightly stale pool, but GPU never blocks.

---

## Tier 2: Medium Impact, Medium Risk (Oracle Review First)

### I-006: Temperature Annealing During Generation
**File:** `src/brain.rs` | **Impact:** 10-15% long-form coherence | **Effort:** 5 lines
- Fixed temperature throughout generation → incoherent long outputs
- **Fix:** High temp early (T=0.9, first 50 tokens), linear decay to T=0.3
- Needs tuning — Oracle should review the annealing schedule

### I-007: Dynamic Memory Retrieval During Generation
**File:** `src/brain.rs:1800-1817` | **Impact:** 15-25% better factual grounding | **Effort:** 12 lines
- Memory retrieved once based on concept vector, frozen for all tokens
- No way for decoder to refine retrieval as it generates
- **Fix:** At each autoregressive step, retrieve top-K based on hidden state, not concept
- Risk: computational cost per step increases

### I-008: KV Cache Clone Removal
**File:** `src/transformer.rs:350` | **Impact:** 5-8% for generation | **Effort:** Medium
- `*kv_cache = Some((k.clone(), v.clone()))` clones K/V on every cached step
- O(N^2) clone overhead for sequence of length N
- **Fix:** Use reference-counting instead of explicit clone
- Risk: Need to verify candle's Tensor refcount semantics

### I-009: Concept-Guided Attention Masking
**File:** `src/transformer.rs:512-515` | **Impact:** 5-15% off-topic reduction | **Effort:** 8 lines
- After self-attention, boost scores to concept prefix tokens by 1.2-1.5x
- Keeps early generation concept-aligned without retraining
- Risk: may cause over-reliance on prefix, reducing flexibility

### I-010: Pre-generate Noise Patterns
**File:** `src/brain.rs:987-992` | **Impact:** 2-3% CPU→GPU | **Effort:** 15 min
- Noise injection runs per-token RNG in hot loop (1024 calls × 100M steps = 102B ops)
- **Fix:** Pre-generate noise patterns for the epoch, reuse across steps
- Risk: patterns become correlated → less noise diversity (acceptable tradeoff)

---

## Tier 3: High Impact, High Risk (Architecture Changes)

### I-011: Multi-Scale Concept Projection
**File:** `src/brain.rs:1768-1789` | **Impact:** 10-20% semantic coherence | **Effort:** 15 lines
- Single (n_concept_tokens, d_model) prefix projection
- **Fix:** Create 3 scales (1 summary, 4 decomposition, 16 detailed), concatenate
- Risk: changes prefix length → may need retraining

### I-012: Attention-Based Concept Pooling
**File:** `src/brain.rs:523-531` | **Impact:** 8-15% semantic fidelity | **Effort:** 20 lines
- Mean pooling for concept extraction is lossy (word order lost)
- "check the build" ≡ "build the check" under mean pooling
- **Fix:** Use encoder attention scores to weight the pooling
- Risk: requires reprocessing all inputs through frozen encoder

### I-013: GPU-Resident Memory Pool
**File:** `src/brain.rs` | **Impact:** 10%+ GPU util | **Effort:** Substantial refactor
- Keep memory pool as tensor on GPU instead of Vec<Vec<f32>> on CPU
- Eliminates GPU→CPU→GPU roundtrip for every memory access
- Risk: major plumbing change, needs careful gradient handling

### I-014: Cross-Attention Memory K/V Caching
**File:** `src/transformer.rs:441-468` | **Impact:** 2-5% (large memory only) | **Effort:** Medium
- Cross-attention re-projects static memory K/V every forward pass
- **Fix:** Cache projected K/V when memory is constant
- Risk: invalidation logic if memory changes mid-generation

---

## Tier 4: Diagnostics & Evaluation (Knowledge, Not Speed)

### I-015: Expand Eval Gallery to 32+ Prompts
**File:** `src/brain.rs:1907-1930` | **Impact:** Detect 2-3 failure modes | **Effort:** 15 lines
- Current: 8 fixed prompts (all casual English)
- **Fix:** Add code, math, memory, edge case prompts
- Zero performance cost, high diagnostic value

### I-016: Automatic Generation Quality Metrics
**File:** `src/brain.rs` eval section | **Impact:** Quantitative feedback | **Effort:** 30 lines
- No metrics currently — just human-read strings
- **Fix:** Compute per-output: length, vocab diversity, ngram diversity, concept alignment
- Essential for tuning sampling parameters systematically

### I-017: Memory Attention Diagnostics
**File:** `src/transformer.rs:441-469` | **Impact:** Understand memory usage | **Effort:** 6 lines
- Cross-attention weights computed but never inspected
- **Fix:** Log which memory entries actually influence generation
- Reveals if memory is being used or ignored entirely

### I-018: Vocab Saturation Analysis
**File:** `src/tokenizer.rs` | **Impact:** 10-15% better merge quality | **Effort:** 40 lines
- 200 learned merges — unknown how many are "dead" (never used in output)
- **Fix:** Analyze merge frequency in actual generated outputs, prune bottom 10%
- One-time analysis, not a runtime change

### I-019: Add Common Linguistic N-grams as Forced Merges
**File:** `src/tokenizer.rs:604-669` | **Impact:** 5-10% token compression | **Effort:** 30 lines
- BPE discovers merges from corpus frequency, misses high-value linguistic patterns
- Common endings ("tion", "ing"), digraphs ("th", "sh"), and prefixes missing
- **Fix:** Manually curate 20-30 high-value forced merges, regenerate tokenizer

---

## Implementation Priority Queue

1. **I-002** (3 lines, temperature order fix — do first, zero risk)
2. **I-001** (cache L2 norms — biggest training speedup)
3. **I-003** (token-frequency penalty — biggest generation quality win)
4. **I-005** (async memory exports — remove GPU stalls)
5. **I-004** (cached causal mask — straightforward fix)
6. **I-015** (expand gallery — diagnostic value)
7. **I-006** (temperature annealing — needs Oracle review)
8. **I-016** (quality metrics — enables data-driven tuning)

---

*Last updated: 2026-02-27 03:30 UTC*
