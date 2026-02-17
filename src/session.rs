// Multi-turn session state + ReAct loop -- T-022, T-023 (Phase 4, new)
//
// Ring buffer of turns (max 32)
// ReAct: Reason -> Act -> Observe -> Reason (max 10 iterations)

use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// Turn types
// ---------------------------------------------------------------------------

/// Role of a conversation turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    /// System observations (tool output, ReAct "Observe" steps)
    Observation,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Observation => write!(f, "observation"),
        }
    }
}

/// A single conversation turn.
#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub content: String,
}

impl Turn {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }

    pub fn observation(content: impl Into<String>) -> Self {
        Self { role: Role::Observation, content: content.into() }
    }
}

// ---------------------------------------------------------------------------
// Session ring buffer
// ---------------------------------------------------------------------------

/// Fixed-capacity ring buffer of conversation turns.
/// When full, the oldest turn is evicted (FIFO).
pub struct SessionBuffer {
    turns: VecDeque<Turn>,
    capacity: usize,
}

impl SessionBuffer {
    /// Create a new session buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SessionBuffer capacity must be > 0");
        Self {
            turns: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Default session buffer (32 turns).
    pub fn default_session() -> Self {
        Self::new(32)
    }

    /// Push a turn. Evicts oldest if at capacity.
    pub fn push(&mut self, turn: Turn) {
        if self.turns.len() >= self.capacity {
            self.turns.pop_front();
        }
        self.turns.push_back(turn);
    }

    /// Push a user message.
    pub fn push_user(&mut self, content: impl Into<String>) {
        self.push(Turn::user(content));
    }

    /// Push an assistant response.
    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.push(Turn::assistant(content));
    }

    /// Push a system observation (tool output, etc).
    pub fn push_observation(&mut self, content: impl Into<String>) {
        self.push(Turn::observation(content));
    }

    /// Number of turns currently in the buffer.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all turns.
    pub fn clear(&mut self) {
        self.turns.clear();
    }

    /// Get the last N turns (most recent).
    pub fn last_n(&self, n: usize) -> Vec<&Turn> {
        let skip = self.turns.len().saturating_sub(n);
        self.turns.iter().skip(skip).collect()
    }

    /// Get all turns in order (oldest first).
    pub fn all_turns(&self) -> impl Iterator<Item = &Turn> {
        self.turns.iter()
    }

    /// Build a context string from recent turns for the decoder.
    /// Format: "<role>: <content>\n" per turn, truncated to `max_bytes`.
    pub fn context_string(&self, max_turns: usize, max_bytes: usize) -> String {
        let recent = self.last_n(max_turns);
        let mut parts: Vec<String> = Vec::new();
        let mut total_bytes = 0;

        // Build from most recent backward, then reverse
        for turn in recent.iter().rev() {
            let line = format!("{}: {}", turn.role, turn.content);
            let line_bytes = line.len() + 1; // +1 for newline
            if total_bytes + line_bytes > max_bytes && !parts.is_empty() {
                break;
            }
            total_bytes += line_bytes;
            parts.push(line);
        }

        parts.reverse();
        parts.join("\n")
    }

    /// Get the last user message, if any.
    pub fn last_user_message(&self) -> Option<&str> {
        self.turns.iter().rev()
            .find(|t| t.role == Role::User)
            .map(|t| t.content.as_str())
    }

    /// Get the last assistant response, if any.
    pub fn last_assistant_response(&self) -> Option<&str> {
        self.turns.iter().rev()
            .find(|t| t.role == Role::Assistant)
            .map(|t| t.content.as_str())
    }
}

// ---------------------------------------------------------------------------
// ReAct loop state -- T-023
// ---------------------------------------------------------------------------

/// ReAct step types: Reason → Act → Observe
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReActPhase {
    Reason,
    Act,
    Observe,
}

/// Tracks ReAct loop iterations within a single goal.
pub struct ReActState {
    pub phase: ReActPhase,
    pub iteration: usize,
    pub max_iterations: usize,
}

impl ReActState {
    pub fn new(max_iterations: usize) -> Self {
        Self {
            phase: ReActPhase::Reason,
            iteration: 0,
            max_iterations,
        }
    }

    /// Default ReAct state (max 10 iterations).
    pub fn default_react() -> Self {
        Self::new(10)
    }

    /// Advance to the next phase. Returns false if max iterations reached.
    pub fn advance(&mut self) -> bool {
        match self.phase {
            ReActPhase::Reason => {
                self.phase = ReActPhase::Act;
                true
            }
            ReActPhase::Act => {
                self.phase = ReActPhase::Observe;
                true
            }
            ReActPhase::Observe => {
                self.iteration += 1;
                if self.iteration >= self.max_iterations {
                    return false;
                }
                self.phase = ReActPhase::Reason;
                true
            }
        }
    }

    /// Whether the loop has exhausted its iterations.
    pub fn is_done(&self) -> bool {
        self.iteration >= self.max_iterations
    }

    /// Reset for a new goal.
    pub fn reset(&mut self) {
        self.phase = ReActPhase::Reason;
        self.iteration = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_buffer_basic() {
        let mut buf = SessionBuffer::new(4);
        assert!(buf.is_empty());
        assert_eq!(buf.capacity(), 4);

        buf.push_user("hello");
        buf.push_assistant("hi there");
        assert_eq!(buf.len(), 2);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_session_buffer_eviction() {
        let mut buf = SessionBuffer::new(3);
        buf.push_user("first");
        buf.push_assistant("r1");
        buf.push_user("second");
        assert_eq!(buf.len(), 3);

        // This should evict "first"
        buf.push_assistant("r2");
        assert_eq!(buf.len(), 3);

        let turns: Vec<&Turn> = buf.all_turns().collect();
        assert_eq!(turns[0].content, "r1");
        assert_eq!(turns[1].content, "second");
        assert_eq!(turns[2].content, "r2");
    }

    #[test]
    fn test_session_buffer_last_n() {
        let mut buf = SessionBuffer::new(10);
        for i in 0..5 {
            buf.push_user(format!("msg {}", i));
        }
        let last2 = buf.last_n(2);
        assert_eq!(last2.len(), 2);
        assert_eq!(last2[0].content, "msg 3");
        assert_eq!(last2[1].content, "msg 4");
    }

    #[test]
    fn test_session_buffer_context_string() {
        let mut buf = SessionBuffer::new(10);
        buf.push_user("hello");
        buf.push_assistant("Hi! How can I help?");
        buf.push_user("run tests");

        let ctx = buf.context_string(10, 1000);
        assert!(ctx.contains("user: hello"));
        assert!(ctx.contains("assistant: Hi! How can I help?"));
        assert!(ctx.contains("user: run tests"));
    }

    #[test]
    fn test_session_buffer_context_string_truncation() {
        let mut buf = SessionBuffer::new(10);
        buf.push_user("a]".repeat(100)); // long message
        buf.push_assistant("short");

        // Very small byte limit — should get at least the most recent turn
        let ctx = buf.context_string(10, 30);
        assert!(ctx.contains("short"));
    }

    #[test]
    fn test_session_buffer_last_messages() {
        let mut buf = SessionBuffer::new(10);
        buf.push_user("hello");
        buf.push_assistant("hi");
        buf.push_observation("tool output");

        assert_eq!(buf.last_user_message(), Some("hello"));
        assert_eq!(buf.last_assistant_response(), Some("hi"));
    }

    #[test]
    fn test_session_buffer_clear() {
        let mut buf = SessionBuffer::new(10);
        buf.push_user("hello");
        buf.push_assistant("hi");
        assert_eq!(buf.len(), 2);

        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_session_buffer_observation() {
        let mut buf = SessionBuffer::new(10);
        buf.push_observation("cargo test: 106 passed");
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.all_turns().next().unwrap().role, Role::Observation);
    }

    // --- ReAct tests ---

    #[test]
    fn test_react_phases() {
        let mut state = ReActState::new(2);
        assert_eq!(state.phase, ReActPhase::Reason);
        assert!(!state.is_done());

        assert!(state.advance()); // Reason → Act
        assert_eq!(state.phase, ReActPhase::Act);

        assert!(state.advance()); // Act → Observe
        assert_eq!(state.phase, ReActPhase::Observe);

        assert!(state.advance()); // Observe → Reason (iteration 1)
        assert_eq!(state.phase, ReActPhase::Reason);
        assert_eq!(state.iteration, 1);

        assert!(state.advance()); // Reason → Act
        assert!(state.advance()); // Act → Observe
        assert!(!state.advance()); // Observe → done (iteration 2 = max)
        assert!(state.is_done());
    }

    #[test]
    fn test_react_reset() {
        let mut state = ReActState::new(1);
        state.advance(); // R → A
        state.advance(); // A → O
        state.advance(); // O → done
        assert!(state.is_done());

        state.reset();
        assert_eq!(state.phase, ReActPhase::Reason);
        assert_eq!(state.iteration, 0);
        assert!(!state.is_done());
    }

    #[test]
    fn test_react_single_iteration() {
        let mut state = ReActState::new(1);
        assert!(state.advance());  // R → A
        assert!(state.advance());  // A → O
        assert!(!state.advance()); // O → done
        assert!(state.is_done());
    }

    #[test]
    fn test_default_constructors() {
        let buf = SessionBuffer::default_session();
        assert_eq!(buf.capacity(), 32);

        let state = ReActState::default_react();
        assert_eq!(state.max_iterations, 10);
    }
}
