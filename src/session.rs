// Multi-turn session state + ReAct loop -- T-022, T-023 (Phase 4)
//
// Ring buffer of turns (max 32)
// ReAct: Reason -> Act -> Observe -> Reason (max 10 iterations)
// ReactLoop: orchestrates brain reasoning + executor actions

use std::collections::VecDeque;
use std::fmt;
use std::path::PathBuf;

use crate::brain::{Brain, brain_generate, ACT_TALK, ACT_END};
use crate::executor::{Executor, ToolArgs, ToolOutput};
use crate::pipeline::{classify_goal, action_name};
use crate::tokenizer::ConceptTokenizer;
use candle_core::Device;

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
// ReactLoop — orchestrates Reason → Act → Observe cycles
// ---------------------------------------------------------------------------

/// A single step result from the ReAct loop.
#[derive(Debug, Clone)]
pub struct ReactStep {
    pub phase: ReActPhase,
    pub iteration: usize,
    pub content: String,
}

/// Result of a complete ReAct execution.
pub struct ReactResult {
    pub steps: Vec<ReactStep>,
    pub final_response: String,
    pub tool_outputs: Vec<ToolOutput>,
}

/// Orchestrates the full ReAct loop: brain reasons, executor acts, observations fed back.
pub struct ReactLoop {
    pub max_iterations: usize,
    pub max_gen_tokens: usize,
    pub temperature: f64,
}

impl ReactLoop {
    pub fn new(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            max_gen_tokens: 192,
            temperature: 0.5,
        }
    }

    /// Run the ReAct loop for a given user goal.
    ///
    /// Flow per iteration:
    /// 1. **Reason**: Brain generates a thought about what to do next
    /// 2. **Act**: Classify the goal + context, execute the top action
    /// 3. **Observe**: Capture tool output, feed back into context
    ///
    /// Stops early if:
    /// - Brain generates a final response (conversational, no tool needed)
    /// - Max iterations reached
    /// - Brain signals completion via ACT_END
    pub fn run(
        &self,
        brain: &Brain,
        goal: &str,
        session: &mut SessionBuffer,
        executor: &Executor,
        decoder_tok: &ConceptTokenizer,
        device: &Device,
    ) -> anyhow::Result<ReactResult> {
        let mut state = ReActState::new(self.max_iterations);
        let mut steps: Vec<ReactStep> = Vec::new();
        let mut tool_outputs: Vec<ToolOutput> = Vec::new();
        let mut final_response = String::new();

        // Build context from session history + current goal
        let history = session.context_string(8, 2048);
        let mut context = if history.is_empty() {
            goal.to_string()
        } else {
            format!("{}\nuser: {}", history, goal)
        };

        loop {
            match state.phase {
                ReActPhase::Reason => {
                    // Brain generates a reasoning step or direct response
                    let thought_prompt = format!(
                        "Given this conversation, think step by step about what to do:\n{}\nThought:",
                        context
                    );
                    let thought = brain_generate(
                        brain, &thought_prompt, self.max_gen_tokens,
                        self.temperature, true, decoder_tok, device,
                    ).unwrap_or_else(|_| "I'll help with that.".to_string());

                    steps.push(ReactStep {
                        phase: ReActPhase::Reason,
                        iteration: state.iteration,
                        content: thought.clone(),
                    });

                    // Check if this is a pure conversational response (no tool needed)
                    // Classify to see if brain wants to talk or use a tool
                    let (_, actions) = classify_goal(brain, goal, device)?;
                    let first_action = actions[0];

                    if first_action == ACT_TALK || first_action == ACT_END {
                        // Direct conversational response — generate and return
                        let response = brain_generate(
                            brain, goal, self.max_gen_tokens,
                            self.temperature, true, decoder_tok, device,
                        ).unwrap_or_else(|_| thought.clone());

                        final_response = response;
                        break;
                    }

                    if !state.advance() { break; }
                }

                ReActPhase::Act => {
                    // Execute the classified action
                    let (_, actions) = classify_goal(brain, goal, device)?;
                    let action = actions[0];
                    let name = action_name(action);

                    steps.push(ReactStep {
                        phase: ReActPhase::Act,
                        iteration: state.iteration,
                        content: format!("Action: {}", name),
                    });

                    // Map action to tool args and execute
                    let tool_args = action_to_tool_args_simple(
                        action, &context, executor,
                    );

                    match tool_args {
                        Some(args) => {
                            match executor.run(&args) {
                                Ok(output) => {
                                    let stdout = output.stdout.clone();
                                    tool_outputs.push(output);
                                    context = format!("{}\nObservation: {}", context,
                                        truncate_output(&stdout, 500));
                                }
                                Err(e) => {
                                    context = format!("{}\nObservation: Error: {}", context, e);
                                }
                            }
                        }
                        None => {
                            // Unknown action or talk — generate response directly
                            let response = brain_generate(
                                brain, goal, self.max_gen_tokens,
                                self.temperature, true, decoder_tok, device,
                            ).unwrap_or_else(|_| "I'm not sure how to help with that.".to_string());
                            final_response = response;
                            break;
                        }
                    }

                    if !state.advance() { break; }
                }

                ReActPhase::Observe => {
                    // Record observation and decide whether to continue
                    steps.push(ReactStep {
                        phase: ReActPhase::Observe,
                        iteration: state.iteration,
                        content: format!("Observed result, iteration {}", state.iteration),
                    });

                    if !state.advance() { break; }
                }
            }
        }

        // If we didn't get a final response from the loop, generate one from context
        if final_response.is_empty() {
            final_response = brain_generate(
                brain, &context, self.max_gen_tokens,
                self.temperature, true, decoder_tok, device,
            ).unwrap_or_else(|_| "Task completed.".to_string());
        }

        Ok(ReactResult { steps, final_response, tool_outputs })
    }
}

/// Simple action → tool args mapping for ReAct loop (doesn't need full PipelineConfig).
fn action_to_tool_args_simple(
    action: usize, context: &str, _executor: &Executor,
) -> Option<ToolArgs> {
    use crate::brain::*;
    let dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    match action {
        ACT_END | ACT_TALK => None,
        ACT_CARGO_TEST => Some(ToolArgs::CargoTest {
            dir, filter: None, features: vec![],
        }),
        ACT_CARGO_CHECK => Some(ToolArgs::CargoCheck {
            dir, features: vec![],
        }),
        ACT_RG => Some(ToolArgs::Rg {
            pattern: extract_keyword(context).to_string(),
            dir,
            file_type: None,
        }),
        ACT_REPO_READ => Some(ToolArgs::RepoRead {
            path: PathBuf::from(extract_filepath(context)),
        }),
        ACT_REPO_LIST => Some(ToolArgs::RepoList { dir, pattern: None }),
        ACT_MEMORY_ADD => Some(ToolArgs::MemoryAdd {
            key: "auto".to_string(),
            value: context.to_string(),
        }),
        ACT_MEMORY_SEARCH => Some(ToolArgs::MemorySearch {
            query: context.to_string(), limit: 5,
        }),
        _ => None,
    }
}

fn extract_keyword(context: &str) -> &str {
    context.split_whitespace().last().unwrap_or("*")
}

fn extract_filepath(context: &str) -> &str {
    context.split_whitespace()
        .find(|w| w.contains('/') || w.contains('.'))
        .unwrap_or("src/main.rs")
}

fn truncate_output(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars { s } else { &s[..max_chars] }
}

// ---------------------------------------------------------------------------
// JarvisSession — ties together session + memory + brain for serve mode
// ---------------------------------------------------------------------------

/// Complete JARVIS session state for the serve command.
pub struct JarvisSession {
    pub buffer: SessionBuffer,
    pub react: ReactLoop,
}

impl JarvisSession {
    pub fn new() -> Self {
        Self {
            buffer: SessionBuffer::default_session(),
            react: ReactLoop::new(5), // 5 iterations max per turn
        }
    }

    /// Process a user message through the full JARVIS pipeline:
    /// 1. Add to session buffer
    /// 2. Run ReAct loop (reason → act → observe)
    /// 3. Store response in session buffer
    /// 4. Return response text
    pub fn process_message(
        &mut self,
        user_input: &str,
        brain: &Brain,
        executor: &Executor,
        decoder_tok: &ConceptTokenizer,
        device: &Device,
    ) -> anyhow::Result<String> {
        self.buffer.push_user(user_input);

        let result = self.react.run(
            brain, user_input, &mut self.buffer,
            executor, decoder_tok, device,
        )?;

        self.buffer.push_assistant(&result.final_response);

        Ok(result.final_response)
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
