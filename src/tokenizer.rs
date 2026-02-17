// PlanTokenizer: structured plan vocab (373 tokens) for FSM-constrained decoding.
// ConceptTokenizer: language decoder vocab, learns merges from concept space.
// Port from V4 tokenizer.rs (444 LOC) -- T-003, T-014

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Token IDs (stable offsets into the plan vocab)
// ---------------------------------------------------------------------------

pub const TOK_PAD: u32 = 0;
pub const TOK_BOS: u32 = 1;
pub const TOK_UNK: u32 = 2;
pub const TOK_NEWLINE: u32 = 3;
pub const TOK_EOP: u32 = 4;
pub const TOK_STEP: u32 = 5;
pub const TOK_GOAL: u32 = 6;
pub const TOK_PLAN: u32 = 7;

// Action tokens (offset 8..22)
pub const TOK_TALK: u32 = 8;
pub const TOK_CARGOTEST: u32 = 9;
pub const TOK_FIXTESTS: u32 = 10;
pub const TOK_CARGOCHECK: u32 = 11;
pub const TOK_DOCSLINT: u32 = 12;
pub const TOK_RG: u32 = 13;
pub const TOK_REPOLIST: u32 = 14;
pub const TOK_REPOREAD: u32 = 15;
pub const TOK_PATCHDRYRUN: u32 = 16;
pub const TOK_PROVEALGEBRA: u32 = 17;
pub const TOK_LEANSUITE: u32 = 18;
pub const TOK_WIREDEVAL: u32 = 19;
pub const TOK_WIREDTRAIN: u32 = 20;
pub const TOK_MEMADD: u32 = 21;
pub const TOK_MEMSEARCH: u32 = 22;

// Ranges
pub const PAT_START: u32 = 23;
pub const FILE_START: u32 = 29;
pub const PICK_START: u32 = 39;
pub const FROM_START: u32 = 168;
pub const CHAR_START: u32 = 176;

pub const ACTION_TOKENS: &[u32] = &[
    TOK_TALK,
    TOK_CARGOTEST,
    TOK_FIXTESTS,
    TOK_CARGOCHECK,
    TOK_DOCSLINT,
    TOK_RG,
    TOK_REPOLIST,
    TOK_REPOREAD,
    TOK_PATCHDRYRUN,
    TOK_PROVEALGEBRA,
    TOK_LEANSUITE,
    TOK_WIREDEVAL,
    TOK_WIREDTRAIN,
    TOK_MEMADD,
    TOK_MEMSEARCH,
];

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PlanTokenizer {
    pub vocab: Vec<String>,
    tok2id: HashMap<String, u32>,
}

impl PlanTokenizer {
    pub fn new() -> Self {
        let mut vocab: Vec<String> = vec![
            "<PAD>".into(),
            "<BOS>".into(),
            "<UNK>".into(),
            "\n".into(),
            "EOP".into(),
            "STEP".into(),
            "Goal".into(),
            "Plan".into(),
        ];

        for action in [
            "TALK",
            "CARGOTEST",
            "FIXTESTS",
            "CARGOCHECK",
            "DOCSLINT",
            "RG",
            "REPOLIST",
            "REPOREAD",
            "PATCHDRYRUN",
            "PROVEALGEBRA",
            "LEANSUITE",
            "WIREDEVAL",
            "WIREDTRAIN",
            "MEMADD",
            "MEMSEARCH",
        ] {
            vocab.push(action.into());
        }

        for i in 0..6u32 {
            vocab.push(format!("PAT{i}"));
        }
        for i in 0..10u32 {
            vocab.push(format!("FILE{i}"));
        }
        for i in 0..129u32 {
            vocab.push(format!("PICK{i}"));
        }
        for i in 0..8u32 {
            vocab.push(format!("FROM{i}"));
        }

        for b in 32u8..=126u8 {
            vocab.push(format!("<C#{b}>"));
        }

        for w in [
            "a",
            "an",
            "and",
            "apply",
            "build",
            "check",
            "code",
            "docs",
            "enforcement",
            "file",
            "files",
            "find",
            "fix",
            "gpu",
            "gpupolicy",
            "hello",
            "jarviscmd",
            "lint",
            "lean",
            "list",
            "open",
            "patch",
            "policy",
            "policyrunctx",
            "repair",
            "repo",
            "rg",
            "runid",
            "search",
            "suite",
            "sync",
            "test",
            "tests",
            "the",
            "then",
            "verify",
            "wired",
            "workspace",
            // V2 expansions
            "blue",
            "color",
            "fact",
            "favorite",
            "memory",
            "note",
            "preference",
            "procedure",
            "prove",
            "recall",
            "remember",
            "retrieve",
            "save",
            "store",
        ] {
            vocab.push(w.into());
        }

        for t in [
            "TEXT",
            "ENDTEXT",
            "PATH",
            "LHS",
            "RHS",
            "KINDFACT",
            "KINDPROCEDURE",
            "KINDPREFERENCE",
            "is",
        ] {
            vocab.push(t.into());
        }

        for i in 0..=32u32 {
            vocab.push(format!("TEXTLEN{i}"));
        }

        for t in ["x*(y+z)", "x*y+x*z"] {
            vocab.push(t.into());
        }

        for t in ["THINK", "ENDTHINK", "BECAUSE", "THEN", "IF", "NEEDS"] {
            vocab.push(t.into());
        }

        let mut tok2id = HashMap::with_capacity(vocab.len());
        for (i, tok) in vocab.iter().enumerate() {
            tok2id.insert(tok.clone(), i as u32);
        }

        Self { vocab, tok2id }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn id(&self, tok: &str) -> u32 {
        self.tok2id.get(tok).copied().unwrap_or(TOK_UNK)
    }

    pub fn token(&self, id: u32) -> &str {
        self.vocab
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<UNK>")
    }

    pub fn pad_id(&self) -> u32 {
        TOK_PAD
    }
    pub fn bos_id(&self) -> u32 {
        TOK_BOS
    }
    pub fn unk_id(&self) -> u32 {
        TOK_UNK
    }
    pub fn eop_id(&self) -> u32 {
        TOK_EOP
    }
    pub fn step_id(&self) -> u32 {
        TOK_STEP
    }

    fn tokenize_text(&self, s: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut cur = String::new();
        fn flush(out: &mut Vec<String>, cur: &mut String) {
            if !cur.is_empty() {
                out.push(std::mem::take(cur));
            }
        }
        for ch in s.chars() {
            if ch == '\n' {
                flush(&mut out, &mut cur);
                out.push("\n".to_string());
                continue;
            }
            if ch.is_ascii_alphanumeric() || ch == '\'' {
                cur.push(ch);
                continue;
            }
            if ch.is_ascii_whitespace() {
                flush(&mut out, &mut cur);
                continue;
            }
            flush(&mut out, &mut cur);
            if ch.is_ascii_punctuation() {
                out.push(ch.to_string());
            }
        }
        flush(&mut out, &mut cur);
        out
    }

    /// Encode text to token IDs. Unknown words fall back to character tokens.
    pub fn encode(&self, s: &str) -> Vec<u32> {
        let toks = self.tokenize_text(s);
        let mut out: Vec<u32> = Vec::new();
        for t in &toks {
            let id = self.id(t);
            if id != TOK_UNK {
                out.push(id);
                continue;
            }
            if t.is_ascii() {
                for b in t.as_bytes() {
                    let char_tok = format!("<C#{b}>");
                    let cid = self.id(&char_tok);
                    if cid != TOK_UNK {
                        out.push(cid);
                    } else {
                        out.push(TOK_UNK);
                    }
                }
            } else {
                out.push(TOK_UNK);
            }
        }
        out
    }

    /// Decode token IDs back to text with spacing heuristics.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut prev_was_space = true;
        let mut prev_was_char_tok = false;
        for &id in ids {
            let tok = self.token(id);
            if matches!(tok, "<PAD>" | "<BOS>" | "<UNK>") {
                continue;
            }
            if tok == "\n" {
                out.push('\n');
                prev_was_space = true;
                prev_was_char_tok = false;
                continue;
            }
            if let Some(n) = tok.strip_prefix("<C#").and_then(|s| s.strip_suffix('>')) {
                if let Ok(v) = n.parse::<u16>() {
                    let b = (v.min(255) as u8) as char;
                    if !out.is_empty() && !prev_was_space && !prev_was_char_tok {
                        out.push(' ');
                    }
                    out.push(b);
                    prev_was_space = false;
                    prev_was_char_tok = true;
                    continue;
                }
            }
            if tok.len() == 1 {
                let ch = tok.chars().next().unwrap();
                if ch.is_ascii_punctuation() {
                    match ch {
                        '.' | ',' | '!' | '?' | ';' | ':' => {
                            out.push(ch);
                            prev_was_space = false;
                            prev_was_char_tok = false;
                            continue;
                        }
                        _ => {}
                    }
                }
            }
            if !out.is_empty() && !prev_was_space {
                out.push(' ');
            }
            out.push_str(tok);
            prev_was_space = false;
            prev_was_char_tok = false;
        }
        out
    }

    /// Build a plan prompt string: "Goal\n{goal}\nPlan\n"
    pub fn plan_prompt(&self, goal: &str) -> String {
        let g = goal.trim().replace('\n', " ").to_lowercase();
        format!("Goal\n{g}\nPlan\n")
    }

    pub fn encode_prompt(&self, goal: &str) -> Vec<u32> {
        self.encode(&self.plan_prompt(goal))
    }

    /// Pad or truncate token IDs to a fixed length (left-pad).
    pub fn pad_or_truncate(&self, ids: &[u32], seq_len: usize) -> Vec<u32> {
        if ids.len() >= seq_len {
            ids[ids.len() - seq_len..].to_vec()
        } else {
            let mut padded = vec![TOK_PAD; seq_len - ids.len()];
            padded.extend_from_slice(ids);
            padded
        }
    }

    pub fn is_structural(&self, id: u32) -> bool {
        let tok = self.token(id);
        matches!(
            tok,
            "<PAD>"
                | "<BOS>"
                | "<UNK>"
                | "\n"
                | "EOP"
                | "STEP"
                | "Goal"
                | "Plan"
                | "TEXT"
                | "ENDTEXT"
                | "PATH"
                | "LHS"
                | "RHS"
                | "THINK"
                | "ENDTHINK"
                | "BECAUSE"
                | "THEN"
                | "IF"
                | "NEEDS"
        ) || ACTION_TOKENS.contains(&id)
            || tok.starts_with("PAT")
            || tok.starts_with("FILE")
            || tok.starts_with("PICK")
            || tok.starts_with("FROM")
            || tok.starts_with("KIND")
            || tok.starts_with("TEXTLEN")
    }

    pub fn payload_ids(&self) -> Vec<u32> {
        (0..self.vocab_size() as u32)
            .filter(|&id| !self.is_structural(id))
            .collect()
    }

    pub fn pat_ids(&self) -> Vec<u32> {
        (0..6).map(|i| PAT_START + i).collect()
    }
    pub fn file_ids(&self) -> Vec<u32> {
        (0..10).map(|i| FILE_START + i).collect()
    }
    pub fn pick_ids(&self) -> Vec<u32> {
        (0..129).map(|i| PICK_START + i).collect()
    }
    pub fn from_ids(&self) -> Vec<u32> {
        (0..8).map(|i| FROM_START + i).collect()
    }
    pub fn kind_ids(&self) -> Vec<u32> {
        vec![
            self.id("KINDFACT"),
            self.id("KINDPROCEDURE"),
            self.id("KINDPREFERENCE"),
        ]
    }
    pub fn textlen_ids(&self) -> Vec<u32> {
        (1..=32).map(|i| self.id(&format!("TEXTLEN{i}"))).collect()
    }
    pub fn action_ids(&self) -> Vec<u32> {
        ACTION_TOKENS.to_vec()
    }
}

impl Default for PlanTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ConceptTokenizer — brain-inspired, learns merges from concept space
// ---------------------------------------------------------------------------

/// Special token IDs for language encoding (shared with brain.rs TalkTokenizer).
pub const CONCEPT_TOK_PAD: u32 = 256;
pub const CONCEPT_TOK_BOS: u32 = 257;
pub const CONCEPT_TOK_EOS: u32 = 258;
const CONCEPT_BASE_VOCAB: usize = 259; // 256 bytes + PAD + BOS + EOS

/// A learned merge rule: a byte pattern that maps to a single token.
#[derive(Clone, Debug)]
pub struct MergeRule {
    /// The byte pattern this rule matches (2-8 bytes).
    pub pattern: Vec<u8>,
    /// The assigned token ID (>= 259).
    pub token_id: u32,
    /// Concept-space consistency score used during discovery.
    pub score: f32,
}

/// Tokenizer that learns merges from concept-space similarity.
///
/// With zero merges, behaves identically to TalkTokenizer (byte-level + BOS/EOS).
/// After merge discovery, compresses text by replacing frequent, semantically-stable
/// byte patterns with single tokens. Lossless: unknown sequences fall back to raw bytes.
#[derive(Clone, Debug)]
pub struct ConceptTokenizer {
    /// Learned merge rules, sorted longest pattern first for greedy matching.
    merges: Vec<MergeRule>,
    /// Reverse map: token_id → byte pattern (for decoding merged tokens).
    id_to_bytes: HashMap<u32, Vec<u8>>,
    /// Total vocab size = 259 + number of merges.
    total_vocab: usize,
}

impl ConceptTokenizer {
    /// Create a new tokenizer with no learned merges (byte-level only).
    pub fn new() -> Self {
        Self {
            merges: Vec::new(),
            id_to_bytes: HashMap::new(),
            total_vocab: CONCEPT_BASE_VOCAB,
        }
    }

    /// Create a tokenizer from a set of pre-discovered merge rules.
    pub fn from_merges(merges: Vec<MergeRule>) -> Self {
        let mut id_to_bytes = HashMap::with_capacity(merges.len());
        for rule in &merges {
            id_to_bytes.insert(rule.token_id, rule.pattern.clone());
        }
        let total_vocab = CONCEPT_BASE_VOCAB + merges.len();
        // Sort merges by pattern length descending for greedy longest-match.
        let mut sorted = merges;
        sorted.sort_by(|a, b| b.pattern.len().cmp(&a.pattern.len()));
        Self {
            merges: sorted,
            id_to_bytes,
            total_vocab,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.total_vocab
    }

    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    pub fn pad_id(&self) -> u32 { CONCEPT_TOK_PAD }
    pub fn bos_id(&self) -> u32 { CONCEPT_TOK_BOS }
    pub fn eos_id(&self) -> u32 { CONCEPT_TOK_EOS }

    /// Encode text into token IDs: [BOS, ...tokens..., EOS].
    /// Uses greedy longest-match for learned merges, falls back to raw bytes.
    pub fn encode(&self, s: &str) -> Vec<u32> {
        let mut ids = vec![CONCEPT_TOK_BOS];
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let mut matched = false;
            // Try longest merge first (merges are pre-sorted by pattern length desc).
            for rule in &self.merges {
                let plen = rule.pattern.len();
                if i + plen <= bytes.len() && bytes[i..i + plen] == rule.pattern[..] {
                    ids.push(rule.token_id);
                    i += plen;
                    matched = true;
                    break;
                }
            }
            if !matched {
                ids.push(bytes[i] as u32);
                i += 1;
            }
        }
        ids.push(CONCEPT_TOK_EOS);
        ids
    }

    /// Decode token IDs back to text. Expands merged tokens, skips specials.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id < 256 {
                bytes.push(id as u8);
            } else if let Some(pattern) = self.id_to_bytes.get(&id) {
                bytes.extend_from_slice(pattern);
            }
            // Skip PAD/BOS/EOS (256-258)
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Left-pad or truncate to fixed length (matches TalkTokenizer behavior).
    pub fn pad_or_truncate(&self, ids: &[u32], len: usize) -> Vec<u32> {
        if ids.len() >= len {
            ids[..len].to_vec()
        } else {
            let mut out = vec![CONCEPT_TOK_PAD; len];
            let offset = len - ids.len();
            out[offset..].copy_from_slice(ids);
            out
        }
    }

    /// Compute compression ratio: original byte count / token count (excluding BOS/EOS).
    pub fn compression_ratio(&self, s: &str) -> f32 {
        let byte_count = s.len() as f32;
        let ids = self.encode(s);
        let token_count = (ids.len() - 2) as f32; // exclude BOS and EOS
        if token_count <= 0.0 { return 1.0; }
        byte_count / token_count
    }

    // ----- Merge Discovery (offline, called after encoder training) -----

    /// Discover merge rules from corpus + concept vectors.
    ///
    /// `concept_fn` maps a byte slice to a concept vector (calls the trained encoder).
    /// This is the core innovation: merges are ranked by semantic consistency in
    /// concept space, not statistical frequency like BPE.
    pub fn discover_merges<F>(
        corpus: &[(&str, &str)],
        _concept_fn: F,
        max_merges: usize,
        min_frequency: usize,
    ) -> Vec<MergeRule>
    where
        F: Fn(&[u8]) -> Vec<f32>,
    {
        // Phase 1: Collect unique n-grams with frequency counts.
        // Previous version called concept_fn per occurrence (~8.6M calls for 3K pairs).
        // With context-free encoding, same bytes always produce the same concept vector,
        // so concept_consistency ≡ 1.0. Score simplifies to frequency × compression.
        // When context-aware encoding is added, restore per-context concept vectors.
        let mut ngram_freq: HashMap<Vec<u8>, usize> = HashMap::new();

        for (prompt, response) in corpus {
            for text in [*prompt, *response] {
                let bytes = text.as_bytes();
                for n in 2..=8usize {
                    if bytes.len() < n { continue; }
                    for window in bytes.windows(n) {
                        *ngram_freq.entry(window.to_vec()).or_default() += 1;
                    }
                }
            }
        }

        eprintln!("[tokenizer] Found {} unique n-grams from corpus", ngram_freq.len());

        // Phase 2: Filter by min_frequency, score = frequency × compression_gain.
        let mut candidates: Vec<(Vec<u8>, f32)> = ngram_freq
            .into_iter()
            .filter(|(_, freq)| *freq >= min_frequency)
            .map(|(pattern, freq)| {
                let frequency = freq as f32;
                let compression = (pattern.len() - 1) as f32; // bytes saved per use
                let score = frequency * compression;
                (pattern, score)
            })
            .collect();

        eprintln!("[tokenizer] {} candidates above min_freq={}", candidates.len(), min_frequency);

        // Phase 3: Sort descending by score, take top-K.
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(max_merges);

        // Phase 4: Assign token IDs starting at CONCEPT_BASE_VOCAB.
        candidates
            .into_iter()
            .enumerate()
            .map(|(i, (pattern, score))| MergeRule {
                token_id: (CONCEPT_BASE_VOCAB + i) as u32,
                pattern,
                score,
            })
            .collect()
    }

    /// Compute concept-space consistency for a set of vectors.
    /// Returns mean pairwise cosine similarity (1.0 = perfectly consistent).
    /// Currently unused by discover_merges (context-free encoding → always 1.0)
    /// but kept for tests and future context-aware encoding.
    #[allow(dead_code)]
    fn concept_consistency(vecs: &[Vec<f32>]) -> f32 {
        if vecs.len() < 2 { return 1.0; }
        let mut total_sim = 0.0f64;
        let mut count = 0u64;
        for i in 0..vecs.len().min(50) { // cap pairwise comparisons for speed
            for j in (i + 1)..vecs.len().min(50) {
                total_sim += cosine_sim(&vecs[i], &vecs[j]) as f64;
                count += 1;
            }
        }
        if count == 0 { return 1.0; }
        (total_sim / count as f64) as f32
    }

    // ----- Serialization -----

    /// Serialize merge rules to bytes: [num_merges(u32), then for each: pattern_len(u8), pattern, token_id(u32), score(f32)].
    pub fn save_merges(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());
        for rule in &self.merges {
            buf.push(rule.pattern.len() as u8);
            buf.extend_from_slice(&rule.pattern);
            buf.extend_from_slice(&rule.token_id.to_le_bytes());
            buf.extend_from_slice(&rule.score.to_le_bytes());
        }
        buf
    }

    /// Deserialize merge rules from bytes.
    pub fn load_merges(data: &[u8]) -> Result<Self, String> {
        if data.len() < 4 {
            return Err("data too short for merge count".into());
        }
        let num = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut pos = 4;
        let mut merges = Vec::with_capacity(num);
        for _ in 0..num {
            if pos >= data.len() {
                return Err("unexpected end of merge data".into());
            }
            let plen = data[pos] as usize;
            pos += 1;
            if pos + plen + 8 > data.len() {
                return Err("truncated merge entry".into());
            }
            let pattern = data[pos..pos + plen].to_vec();
            pos += plen;
            let token_id = u32::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
            ]);
            pos += 4;
            let score = f32::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
            ]);
            pos += 4;
            merges.push(MergeRule { pattern, token_id, score });
        }
        Ok(Self::from_merges(merges))
    }
}

impl Default for ConceptTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity between two f32 vectors.
#[allow(dead_code)]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_size() {
        let tok = PlanTokenizer::new();
        assert_eq!(
            tok.vocab_size(),
            373,
            "vocab size mismatch: got {}",
            tok.vocab_size()
        );
    }

    #[test]
    fn test_all_tokens_roundtrip() {
        let tok = PlanTokenizer::new();
        for (i, token_str) in tok.vocab.iter().enumerate() {
            let id = tok.id(token_str);
            assert_eq!(
                id, i as u32,
                "token '{}' has id {} but expected {}",
                token_str, id, i
            );
            let recovered = tok.token(id);
            assert_eq!(
                recovered, token_str,
                "token id {} maps to '{}' but expected '{}'",
                id, recovered, token_str
            );
        }
    }

    #[test]
    fn test_special_token_ids() {
        let tok = PlanTokenizer::new();
        assert_eq!(tok.id("<PAD>"), 0);
        assert_eq!(tok.id("<BOS>"), 1);
        assert_eq!(tok.id("<UNK>"), 2);
        assert_eq!(tok.id("EOP"), 4);
        assert_eq!(tok.id("STEP"), 5);
        assert_eq!(tok.id("TALK"), 8);
        assert_eq!(tok.id("RG"), 13);
    }

    #[test]
    fn test_encode_known_words() {
        let tok = PlanTokenizer::new();
        let ids = tok.encode("hello");
        assert_eq!(ids.len(), 1);
        assert_eq!(tok.token(ids[0]), "hello");
    }

    #[test]
    fn test_encode_unknown_falls_back_to_chars() {
        let tok = PlanTokenizer::new();
        let ids = tok.encode("xyz");
        assert_eq!(ids.len(), 3);
        assert_eq!(tok.token(ids[0]), "<C#120>");
        assert_eq!(tok.token(ids[1]), "<C#121>");
        assert_eq!(tok.token(ids[2]), "<C#122>");
    }

    #[test]
    fn test_plan_prompt_format() {
        let tok = PlanTokenizer::new();
        let prompt = tok.plan_prompt("Hello World");
        assert_eq!(prompt, "Goal\nhello world\nPlan\n");
    }

    #[test]
    fn test_encode_plan_prompt() {
        let tok = PlanTokenizer::new();
        let ids = tok.encode_prompt("hello");
        assert!(ids.len() >= 4);
        assert_eq!(tok.token(ids[0]), "Goal");
        assert_eq!(tok.token(ids[1]), "\n");
        assert_eq!(tok.token(ids[2]), "hello");
    }

    #[test]
    fn test_pad_or_truncate() {
        let tok = PlanTokenizer::new();
        let ids = vec![5, 8, 4]; // STEP TALK EOP
        let padded = tok.pad_or_truncate(&ids, 8);
        assert_eq!(padded.len(), 8);
        assert_eq!(padded[0..5], [0, 0, 0, 0, 0]);
        assert_eq!(padded[5..8], [5, 8, 4]);

        let truncated = tok.pad_or_truncate(&ids, 2);
        assert_eq!(truncated, vec![8, 4]);
    }

    #[test]
    fn test_action_tokens_valid() {
        let tok = PlanTokenizer::new();
        for &aid in ACTION_TOKENS {
            let name = tok.token(aid);
            assert_ne!(name, "<UNK>", "action token {} maps to UNK", aid);
        }
    }

    #[test]
    fn test_payload_ids_excludes_structural() {
        let tok = PlanTokenizer::new();
        let payload = tok.payload_ids();
        assert!(!payload.contains(&TOK_STEP));
        assert!(!payload.contains(&TOK_EOP));
        assert!(!payload.contains(&TOK_TALK));
        assert!(payload.contains(&tok.id("hello")));
    }

    #[test]
    fn test_no_duplicate_vocab_entries() {
        let tok = PlanTokenizer::new();
        let mut seen = HashMap::new();
        for (i, t) in tok.vocab.iter().enumerate() {
            if let Some(prev) = seen.insert(t.clone(), i) {
                panic!(
                    "duplicate vocab entry '{}' at positions {} and {}",
                    t, prev, i
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // ConceptTokenizer tests
    // -----------------------------------------------------------------------

    #[test]
    fn concept_tok_no_merges_matches_byte_level() {
        let tok = ConceptTokenizer::new();
        assert_eq!(tok.vocab_size(), 259);
        assert_eq!(tok.num_merges(), 0);

        let ids = tok.encode("hello");
        // BOS + h(104) e(101) l(108) l(108) o(111) + EOS
        assert_eq!(ids[0], CONCEPT_TOK_BOS);
        assert_eq!(ids[1], 104); // 'h'
        assert_eq!(ids[2], 101); // 'e'
        assert_eq!(ids[3], 108); // 'l'
        assert_eq!(ids[4], 108); // 'l'
        assert_eq!(ids[5], 111); // 'o'
        assert_eq!(ids[6], CONCEPT_TOK_EOS);
        assert_eq!(ids.len(), 7);
    }

    #[test]
    fn concept_tok_encode_decode_roundtrip_no_merges() {
        let tok = ConceptTokenizer::new();
        let text = "Hello, world! This is JARVIS.";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn concept_tok_encode_decode_roundtrip_with_merges() {
        let merges = vec![
            MergeRule { pattern: b"he".to_vec(), token_id: 259, score: 10.0 },
            MergeRule { pattern: b"ll".to_vec(), token_id: 260, score: 8.0 },
            MergeRule { pattern: b"lo".to_vec(), token_id: 261, score: 6.0 },
        ];
        let tok = ConceptTokenizer::from_merges(merges);
        assert_eq!(tok.vocab_size(), 262); // 259 + 3 merges

        let text = "hello";
        let ids = tok.encode(text);
        // "he" → 259, "ll" → 260, "o" → 111
        // BOS, 259, 260, 111, EOS = 5 tokens (down from 7 with no merges)
        assert_eq!(ids.len(), 5);
        assert_eq!(ids[0], CONCEPT_TOK_BOS);
        assert_eq!(ids[1], 259); // "he"
        assert_eq!(ids[2], 260); // "ll"
        assert_eq!(ids[3], 111); // 'o'
        assert_eq!(ids[4], CONCEPT_TOK_EOS);

        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn concept_tok_longest_match_wins() {
        let merges = vec![
            MergeRule { pattern: b"hel".to_vec(), token_id: 259, score: 12.0 },
            MergeRule { pattern: b"he".to_vec(), token_id: 260, score: 10.0 },
        ];
        let tok = ConceptTokenizer::from_merges(merges);

        let ids = tok.encode("hello");
        // "hel" → 259, then "l" → 108, "o" → 111
        assert_eq!(ids[1], 259); // "hel" matched, not "he"
        assert_eq!(ids[2], 108); // 'l'
        assert_eq!(ids[3], 111); // 'o'
        assert_eq!(ids.len(), 5); // BOS + 3 tokens + EOS

        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn concept_tok_compression_ratio() {
        let tok_no_merge = ConceptTokenizer::new();
        let ratio = tok_no_merge.compression_ratio("hello");
        assert!((ratio - 1.0).abs() < 0.01, "no merges should give ratio ~1.0");

        let merges = vec![
            MergeRule { pattern: b"hello".to_vec(), token_id: 259, score: 20.0 },
        ];
        let tok_merged = ConceptTokenizer::from_merges(merges);
        let ratio = tok_merged.compression_ratio("hello");
        assert_eq!(ratio, 5.0, "5 bytes / 1 token = 5.0x compression");
    }

    #[test]
    fn concept_tok_pad_or_truncate() {
        let tok = ConceptTokenizer::new();
        let ids = vec![CONCEPT_TOK_BOS, 104, 105, CONCEPT_TOK_EOS]; // "hi"

        // Left-pad
        let padded = tok.pad_or_truncate(&ids, 8);
        assert_eq!(padded.len(), 8);
        assert_eq!(padded[0..4], [CONCEPT_TOK_PAD; 4]);
        assert_eq!(padded[4..8], [CONCEPT_TOK_BOS, 104, 105, CONCEPT_TOK_EOS]);

        // Truncate
        let truncated = tok.pad_or_truncate(&ids, 2);
        assert_eq!(truncated, vec![CONCEPT_TOK_BOS, 104]);
    }

    #[test]
    fn concept_tok_empty_string() {
        let tok = ConceptTokenizer::new();
        let ids = tok.encode("");
        assert_eq!(ids, vec![CONCEPT_TOK_BOS, CONCEPT_TOK_EOS]);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "");
    }

    #[test]
    fn concept_tok_discover_merges_basic() {
        // Mock concept function: returns a vector based on byte sum (deterministic).
        let concept_fn = |bytes: &[u8]| -> Vec<f32> {
            let sum: f32 = bytes.iter().map(|&b| b as f32).sum();
            vec![sum / 256.0, 1.0 - sum / 512.0, sum / 128.0, 0.5]
        };

        let corpus: Vec<(&str, &str)> = vec![
            ("hello", "hi there"),
            ("hello world", "hey"),
            ("hello friend", "hello again"),
        ];

        let merges = ConceptTokenizer::discover_merges(
            &corpus, concept_fn, 5, 2, // max 5 merges, min freq 2
        );

        // "hello" appears 4+ times, should be a merge candidate.
        // "he" appears frequently, should be a candidate.
        assert!(!merges.is_empty(), "should discover at least one merge");
        // All token IDs >= 259
        for rule in &merges {
            assert!(rule.token_id >= 259);
            assert!(rule.pattern.len() >= 2);
        }
    }

    #[test]
    fn concept_tok_serialization_roundtrip() {
        let merges = vec![
            MergeRule { pattern: b"he".to_vec(), token_id: 259, score: 10.5 },
            MergeRule { pattern: b"llo".to_vec(), token_id: 260, score: 8.2 },
            MergeRule { pattern: b"world".to_vec(), token_id: 261, score: 15.0 },
        ];
        let tok = ConceptTokenizer::from_merges(merges);
        let data = tok.save_merges();
        let tok2 = ConceptTokenizer::load_merges(&data).expect("deserialization failed");

        assert_eq!(tok.vocab_size(), tok2.vocab_size());
        assert_eq!(tok.num_merges(), tok2.num_merges());

        // Encode/decode should match.
        let text = "hello world";
        assert_eq!(tok.encode(text), tok2.encode(text));
        assert_eq!(tok.decode(&tok.encode(text)), tok2.decode(&tok2.encode(text)));
    }

    #[test]
    fn concept_tok_non_ascii() {
        let tok = ConceptTokenizer::new();
        let text = "café";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn concept_tok_consistency_fn() {
        // Identical vectors → consistency 1.0
        let vecs = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
        let c = ConceptTokenizer::concept_consistency(&vecs);
        assert!((c - 1.0).abs() < 0.01);

        // Orthogonal vectors → consistency 0.0
        let vecs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let c = ConceptTokenizer::concept_consistency(&vecs);
        assert!(c.abs() < 0.01);
    }
}
