// End-to-end integration tests — T-012 (Phase 1)
//
// Tests the full pipeline: goal → classify → plan → execute → result
// Uses test-sized models (untrained) to verify structural correctness.

use gestalt::brain::{
    Brain, BrainConfig, PolicyConfig,
    ACT_TALK, ACT_END, ACT_CARGO_CHECK, ACT_REPO_READ, ACT_MEMORY_SEARCH,
    NUM_INTENTS,
};
use gestalt::pipeline::{
    run_goal, run_with_plan, classify_goal, action_name, PipelineConfig,
};
use candle_core::Device;
use candle_nn::VarMap;
use std::path::PathBuf;
use std::time::Duration;

fn create_test_brain() -> (Brain, Device) {
    let device = Device::Cpu;
    let config = BrainConfig::test_brain();
    let policy_cfg = PolicyConfig::test();
    let varmap = VarMap::new();
    let brain = Brain::new(config, &policy_cfg, &varmap, &device).unwrap();
    (brain, device)
}

fn test_pipeline_config() -> PipelineConfig {
    PipelineConfig {
        work_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        allow_writes: false,
        timeout: Duration::from_secs(30),
        max_tokens: 64,
        temperature: 0.8,
    }
}

// ---------------------------------------------------------------------------
// Integration tests from spec
// ---------------------------------------------------------------------------

#[test]
fn test_hello_conversational() {
    // "hello" → TALK plan → greeting (no tool execution)
    let (brain, device) = create_test_brain();
    let config = test_pipeline_config();
    let actions = [ACT_TALK, ACT_END, ACT_END, ACT_END, ACT_END, ACT_END];
    let result = run_with_plan(&brain, "hello", 0, &actions, &config, &device).unwrap();

    assert_eq!(result.goal, "hello");
    assert_eq!(result.steps.len(), 1);
    assert_eq!(result.steps[0].action, ACT_TALK);
    assert!(result.steps[0].response.is_some(), "Talk should produce a response");
    assert!(result.steps[0].tool_output.is_none(), "No tool execution for talk");
    assert!(result.success);
    assert!(!result.final_output.is_empty(), "Should have non-empty output");
}

#[test]
fn test_build_check_pipeline() {
    // "cargo check" → Executor → cargo check → results
    let (brain, device) = create_test_brain();
    let config = test_pipeline_config();
    let actions = [ACT_CARGO_CHECK, ACT_END, ACT_END, ACT_END, ACT_END, ACT_END];
    let result = run_with_plan(
        &brain, "cargo check", 2, &actions, &config, &device,
    ).unwrap();

    assert_eq!(result.steps.len(), 1);
    assert_eq!(result.steps[0].action_name, "cargo_check");
    assert!(result.steps[0].tool_output.is_some());
    let output = result.steps[0].tool_output.as_ref().unwrap();
    assert_eq!(output.exit_code, 0, "cargo check should pass");
    assert!(result.success);
}

#[test]
fn test_composite_search_then_read() {
    // "search and open" → repo_read Cargo.toml → chain to memory_search
    let (brain, device) = create_test_brain();
    let config = test_pipeline_config();
    let actions = [ACT_REPO_READ, ACT_MEMORY_SEARCH, ACT_END, ACT_END, ACT_END, ACT_END];
    let result = run_with_plan(
        &brain, "Cargo.toml", 3, &actions, &config, &device,
    ).unwrap();

    // Both steps should complete
    assert_eq!(result.steps.len(), 2);
    assert!(result.success);

    // Step 0: repo_read gets Cargo.toml content
    assert!(result.steps[0].success);
    assert!(result.steps[0].tool_output.is_some());
    let cargo_content = &result.steps[0].tool_output.as_ref().unwrap().stdout;
    assert!(cargo_content.contains("[package]"), "Should read actual Cargo.toml");

    // Step 1: memory_search with chained context from step 0
    assert!(result.steps[1].success);
    assert_eq!(result.steps[1].action_name, "memory_search");
}

// ---------------------------------------------------------------------------
// Full pipeline (classify → execute)
// ---------------------------------------------------------------------------

#[test]
fn test_full_pipeline_hello() {
    // Full end-to-end: brain classifies, then executes
    let (brain, device) = create_test_brain();
    let config = test_pipeline_config();
    let result = run_goal(&brain, "hello", &config, &device).unwrap();

    // With untrained brain, classification is random but pipeline must not crash
    assert_eq!(result.goal, "hello");
    assert!(result.intent < NUM_INTENTS, "Intent {} out of range", result.intent);
}

#[test]
fn test_full_pipeline_classify() {
    // Verify classify_goal returns valid intent and actions
    let (brain, device) = create_test_brain();
    let (intent, actions) = classify_goal(&brain, "run cargo test", &device).unwrap();
    assert!(intent < NUM_INTENTS);
    for (i, &a) in actions.iter().enumerate() {
        assert!(a < 16, "Action {} at step {} out of range", a, i);
    }
    // Verify action_name works for all returned actions
    for &a in &actions {
        let name = action_name(a);
        assert!(!name.is_empty());
    }
}

#[test]
fn test_pipeline_failure_stops_execution() {
    // Tool failure at step 0 prevents step 1 from running
    let (brain, device) = create_test_brain();
    let config = test_pipeline_config();
    let actions = [ACT_REPO_READ, ACT_CARGO_CHECK, ACT_END, ACT_END, ACT_END, ACT_END];
    let result = run_with_plan(
        &brain, "/nonexistent/path.rs", 4, &actions, &config, &device,
    ).unwrap();

    assert!(!result.success);
    assert_eq!(result.steps.len(), 1, "Should stop after first failure");
    assert!(!result.steps[0].success);
}
