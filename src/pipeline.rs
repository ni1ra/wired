// Unified inference pipeline — T-011 (Phase 1)
//
// fn run_goal(brain, goal, config, device) -> ExecutionResult
// Steps: encode → classify → plan → compile → execute → collect
//
// Pipeline flow:
//   1. Encode goal text via TalkTokenizer
//   2. Classify intent + actions via Brain.classify()
//   3. For each action step: map to tool → execute via Executor → chain output
//   4. Conversational actions use brain_generate instead of tool execution

use crate::brain::{
    Brain, brain_generate, PLAN_STEPS,
    ACT_END, ACT_TALK, ACT_CARGO_TEST, ACT_DOCS_LINT, ACT_RG,
    ACT_PROVE_ALGEBRA, ACT_PATCH_DRY_RUN, ACT_WIRED_EVAL, ACT_WIRED_TRAIN_TEST,
    ACT_MEMORY_ADD, ACT_MEMORY_SEARCH, ACT_CARGO_CHECK, ACT_REPO_LIST,
    ACT_REPO_READ, ACT_FIX_TESTS, ACT_LEAN_SUITE,
};
use crate::executor::{Executor, ToolArgs, ToolOutput};
use anyhow::Result;
use candle_core::{Device, Tensor};
use std::path::PathBuf;
use std::time::Duration;

/// Pipeline configuration.
pub struct PipelineConfig {
    pub work_dir: PathBuf,
    pub allow_writes: bool,
    pub timeout: Duration,
    pub max_tokens: usize,
    pub temperature: f64,
}

impl PipelineConfig {
    pub fn default_config(work_dir: PathBuf) -> Self {
        Self {
            work_dir,
            allow_writes: false,
            timeout: Duration::from_secs(30),
            max_tokens: 128,
            temperature: 0.8,
        }
    }
}

/// Result of a single pipeline step.
pub struct StepResult {
    pub step_index: usize,
    pub action: usize,
    pub action_name: String,
    pub tool_output: Option<ToolOutput>,
    pub response: Option<String>,
    pub success: bool,
}

/// Result of the full pipeline execution.
pub struct ExecutionResult {
    pub goal: String,
    pub intent: usize,
    pub steps: Vec<StepResult>,
    pub final_output: String,
    pub success: bool,
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Classify a goal string using the brain's policy heads.
/// Returns (intent_id, per-step action IDs).
pub fn classify_goal(
    brain: &Brain, goal: &str, device: &Device,
) -> Result<(usize, [usize; PLAN_STEPS])> {
    // Policy backbone uses raw byte encoding (BYTE_VOCAB=256), padded with 0
    let bytes: Vec<u32> = goal.bytes().map(|b| b as u32).collect();
    let seq_len = brain.config.encoder_seq_len;
    let mut padded = vec![0u32; seq_len];
    let len = bytes.len().min(seq_len);
    padded[..len].copy_from_slice(&bytes[..len]);
    let input = Tensor::new(padded, device)?.unsqueeze(0)?;

    let output = brain.classify(&input)?;

    // Intent: argmax over [NUM_INTENTS]
    let intent_vec: Vec<f32> = output.intent_logits.squeeze(0)?.to_vec1()?;
    let intent = intent_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Actions: argmax per step over [PLAN_STEPS, NUM_ACTIONS]
    let act_data: Vec<Vec<f32>> = output.act_logits.squeeze(0)?.to_vec2()?;
    let mut actions = [ACT_END; PLAN_STEPS];
    for (i, step_logits) in act_data.iter().enumerate().take(PLAN_STEPS) {
        actions[i] = step_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(ACT_END);
    }

    Ok((intent, actions))
}

// ---------------------------------------------------------------------------
// Action mapping
// ---------------------------------------------------------------------------

/// Map an action ID to a human-readable name.
pub fn action_name(action: usize) -> &'static str {
    match action {
        ACT_END => "end",
        ACT_TALK => "talk",
        ACT_CARGO_TEST => "cargo_test",
        ACT_DOCS_LINT => "docs_lint",
        ACT_RG => "rg",
        ACT_PROVE_ALGEBRA => "prove_algebra",
        ACT_PATCH_DRY_RUN => "patch_dry_run",
        ACT_WIRED_EVAL => "wired_eval",
        ACT_WIRED_TRAIN_TEST => "wired_train",
        ACT_MEMORY_ADD => "memory_add",
        ACT_MEMORY_SEARCH => "memory_search",
        ACT_CARGO_CHECK => "cargo_check",
        ACT_REPO_LIST => "repo_list",
        ACT_REPO_READ => "repo_read",
        ACT_FIX_TESTS => "fix_tests",
        ACT_LEAN_SUITE => "lean_suite",
        _ => "unknown",
    }
}

/// Map an action ID to executor ToolArgs using context for argument extraction.
fn action_to_tool_args(
    action: usize, config: &PipelineConfig, context: &str,
) -> Option<ToolArgs> {
    let dir = config.work_dir.clone();
    match action {
        ACT_END | ACT_TALK => None,
        ACT_CARGO_TEST => Some(ToolArgs::CargoTest {
            dir, filter: None, features: vec![],
        }),
        ACT_CARGO_CHECK => Some(ToolArgs::CargoCheck {
            dir, features: vec![],
        }),
        ACT_RG => Some(ToolArgs::Rg {
            pattern: extract_pattern(context).to_string(),
            dir,
            file_type: None,
        }),
        ACT_REPO_READ => Some(ToolArgs::RepoRead {
            path: PathBuf::from(extract_path(context)),
        }),
        ACT_REPO_LIST => Some(ToolArgs::RepoList { dir, pattern: None }),
        ACT_DOCS_LINT => Some(ToolArgs::DocsLint { dir }),
        ACT_PROVE_ALGEBRA => Some(ToolArgs::ProveAlgebra {
            expr: context.to_string(),
        }),
        ACT_LEAN_SUITE => Some(ToolArgs::LeanSuite {
            file: PathBuf::from(extract_path(context)),
        }),
        ACT_PATCH_DRY_RUN => Some(ToolArgs::PatchDryRun {
            patch: context.to_string(), dir,
        }),
        ACT_WIRED_EVAL => Some(ToolArgs::WiredEval),
        ACT_WIRED_TRAIN_TEST => Some(ToolArgs::WiredTrain),
        ACT_MEMORY_ADD => Some(ToolArgs::MemoryAdd {
            key: "auto".to_string(),
            value: context.to_string(),
        }),
        ACT_MEMORY_SEARCH => Some(ToolArgs::MemorySearch {
            query: context.to_string(), limit: 5,
        }),
        ACT_FIX_TESTS => Some(ToolArgs::FixTests { dir }),
        _ => None,
    }
}

/// Extract a search pattern from context (last meaningful word).
fn extract_pattern(context: &str) -> &str {
    context.split_whitespace().last().unwrap_or("*")
}

/// Extract a file path from context (first token containing '/' or '.').
fn extract_path(context: &str) -> &str {
    context
        .split_whitespace()
        .find(|w| w.contains('/') || w.contains('.'))
        .unwrap_or("src/main.rs")
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Execute a pre-classified plan (intent + actions already determined).
/// This is the testable core — tests can supply known actions directly.
pub fn run_with_plan(
    brain: &Brain, goal: &str, intent: usize, actions: &[usize; PLAN_STEPS],
    config: &PipelineConfig, device: &Device,
) -> Result<ExecutionResult> {
    let executor = Executor::new(
        config.work_dir.clone(), config.allow_writes, config.timeout,
    );

    let mut steps = Vec::new();
    let mut context = goal.to_string();
    let mut all_success = true;

    for (i, &action) in actions.iter().enumerate() {
        if action == ACT_END {
            break;
        }

        let name = action_name(action).to_string();

        // Conversational: generate via brain instead of tool
        if action == ACT_TALK {
            let response = brain_generate(
                brain, &context, config.max_tokens, config.temperature,
                false, device,
            ).unwrap_or_else(|_| "(generation failed)".to_string());

            steps.push(StepResult {
                step_index: i,
                action,
                action_name: name,
                tool_output: None,
                response: Some(response.clone()),
                success: true,
            });
            context = response;
            continue;
        }

        // Tool execution: map action → ToolArgs → Executor
        match action_to_tool_args(action, config, &context) {
            Some(tool_args) => {
                match executor.run(&tool_args) {
                    Ok(output) => {
                        let success = output.exit_code == 0;
                        let step_output = output.stdout.clone();
                        steps.push(StepResult {
                            step_index: i,
                            action,
                            action_name: name,
                            tool_output: Some(output),
                            response: None,
                            success,
                        });
                        if !success {
                            all_success = false;
                            break;
                        }
                        context = step_output;
                    }
                    Err(e) => {
                        steps.push(StepResult {
                            step_index: i,
                            action,
                            action_name: name,
                            tool_output: None,
                            response: Some(format!("Error: {}", e)),
                            success: false,
                        });
                        all_success = false;
                        break;
                    }
                }
            }
            None => {
                steps.push(StepResult {
                    step_index: i,
                    action,
                    action_name: name,
                    tool_output: None,
                    response: Some("Unknown action".to_string()),
                    success: false,
                });
                all_success = false;
                break;
            }
        }
    }

    let final_output = steps
        .last()
        .and_then(|s| {
            s.tool_output
                .as_ref()
                .map(|o| o.stdout.clone())
                .or_else(|| s.response.clone())
        })
        .unwrap_or_default();

    Ok(ExecutionResult {
        goal: goal.to_string(),
        intent,
        steps,
        final_output,
        success: all_success,
    })
}

/// Full pipeline: classify goal with brain, then execute the resulting plan.
pub fn run_goal(
    brain: &Brain, goal: &str, config: &PipelineConfig, device: &Device,
) -> Result<ExecutionResult> {
    let (intent, actions) = classify_goal(brain, goal, device)?;
    run_with_plan(brain, goal, intent, &actions, config, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::{
        BrainConfig, PolicyConfig, NUM_INTENTS,
        ACT_CARGO_CHECK, ACT_MEMORY_SEARCH, ACT_REPO_READ,
    };
    use candle_nn::VarMap;

    fn test_brain() -> (Brain, Device) {
        let device = Device::Cpu;
        let config = BrainConfig::test_brain();
        let policy_cfg = PolicyConfig::test();
        let varmap = VarMap::new();
        let brain = Brain::new(config, &policy_cfg, &varmap, &device).unwrap();
        (brain, device)
    }

    fn test_config() -> PipelineConfig {
        PipelineConfig::default_config(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        )
    }

    #[test]
    fn test_run_goal_hello() {
        let (brain, device) = test_brain();
        let config = test_config();
        // Explicit conversational plan — no tool execution
        let actions = [ACT_TALK, ACT_END, ACT_END, ACT_END, ACT_END, ACT_END];
        let result = run_with_plan(&brain, "hello", 0, &actions, &config, &device).unwrap();
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.steps[0].action, ACT_TALK);
        assert!(result.steps[0].response.is_some());
        assert!(result.steps[0].tool_output.is_none());
        assert!(result.success);
    }

    #[test]
    fn test_run_goal_cargo_check() {
        let (brain, device) = test_brain();
        let config = test_config();
        // Tool execution: cargo check (faster than cargo test, avoids recursion)
        let actions = [ACT_CARGO_CHECK, ACT_END, ACT_END, ACT_END, ACT_END, ACT_END];
        let result = run_with_plan(
            &brain, "check code", 2, &actions, &config, &device,
        ).unwrap();
        assert_eq!(result.steps.len(), 1);
        assert_eq!(result.steps[0].action_name, "cargo_check");
        assert!(result.steps[0].tool_output.is_some());
        assert!(result.steps[0].success);
    }

    #[test]
    fn test_run_goal_composite() {
        let (brain, device) = test_brain();
        let config = test_config();
        // Multi-step: read Cargo.toml → chain output to memory_search
        let actions = [ACT_REPO_READ, ACT_MEMORY_SEARCH, ACT_END, ACT_END, ACT_END, ACT_END];
        let result = run_with_plan(
            &brain, "Cargo.toml", 3, &actions, &config, &device,
        ).unwrap();
        assert_eq!(result.steps.len(), 2);
        assert!(result.steps[0].success); // repo_read succeeded
        assert!(result.steps[1].success); // memory_search stub succeeded
        assert!(result.success);
    }

    #[test]
    fn test_run_goal_failure() {
        let (brain, device) = test_brain();
        let config = test_config();
        // repo_read with nonexistent file → error → pipeline stops
        let actions = [ACT_REPO_READ, ACT_CARGO_CHECK, ACT_END, ACT_END, ACT_END, ACT_END];
        let result = run_with_plan(
            &brain, "/nonexistent/file.txt", 4, &actions, &config, &device,
        ).unwrap();
        assert!(!result.success);
        assert_eq!(result.steps.len(), 1); // Stopped after first failure
        assert!(!result.steps[0].success);
    }

    #[test]
    fn test_classify_goal_returns_valid() {
        let (brain, device) = test_brain();
        let (intent, actions) = classify_goal(&brain, "hello", &device).unwrap();
        assert!(intent < NUM_INTENTS);
        for &a in &actions {
            assert!(a < 16, "Action {} out of range", a);
        }
    }

    #[test]
    fn test_action_name_coverage() {
        assert_eq!(action_name(ACT_END), "end");
        assert_eq!(action_name(ACT_TALK), "talk");
        assert_eq!(action_name(ACT_CARGO_TEST), "cargo_test");
        assert_eq!(action_name(ACT_RG), "rg");
        assert_eq!(action_name(99), "unknown");
    }

    #[test]
    fn test_pipeline_config_defaults() {
        let config = PipelineConfig::default_config(PathBuf::from("/tmp"));
        assert!(!config.allow_writes);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_tokens, 128);
    }

    #[test]
    fn test_run_goal_full_pipeline() {
        // Full pipeline: brain classifies (untrained → random), then executes
        let (brain, device) = test_brain();
        let config = test_config();
        let result = run_goal(&brain, "hello", &config, &device).unwrap();
        // Untrained brain → random intent, but pipeline should not crash
        assert!(result.intent < NUM_INTENTS);
        assert!(!result.goal.is_empty());
    }
}
