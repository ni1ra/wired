// End-to-end integration tests — T-012 (Phase 1), T-023 (Phase 3 ReAct)
//
// Tests the full pipeline: goal → classify → plan → execute → result
// Uses test-sized models (untrained) to verify structural correctness.

use gestalt::brain::{
    Brain, BrainConfig, PolicyConfig,
    ACT_TALK, ACT_END, ACT_CARGO_CHECK, ACT_REPO_READ, ACT_MEMORY_SEARCH,
    NUM_INTENTS,
};
use gestalt::executor::Executor;
use gestalt::memory::EpisodicMemory;
use gestalt::pipeline::{
    run_goal, run_with_plan, classify_goal, action_name, PipelineConfig,
};
use gestalt::session::{JarvisSession, SessionBuffer, ReactLoop};
use gestalt::tokenizer::ConceptTokenizer;
use candle_core::Device;
use candle_nn::VarMap;
use std::path::PathBuf;
use std::time::Duration;

fn create_test_brain() -> (Brain, ConceptTokenizer, Device) {
    let device = Device::Cpu;
    let config = BrainConfig::test_brain();
    let policy_cfg = PolicyConfig::test();
    let varmap = VarMap::new();
    let brain = Brain::new(config, &policy_cfg, &varmap, &device).unwrap();
    (brain, ConceptTokenizer::new(), device)
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
    let (brain, dtok, device) = create_test_brain();
    let config = test_pipeline_config();
    let actions = [ACT_TALK, ACT_END, ACT_END, ACT_END, ACT_END, ACT_END];
    let result = run_with_plan(&brain, "hello", 0, &actions, &config, &dtok, &device).unwrap();

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
    let (brain, dtok, device) = create_test_brain();
    let config = test_pipeline_config();
    let actions = [ACT_CARGO_CHECK, ACT_END, ACT_END, ACT_END, ACT_END, ACT_END];
    let result = run_with_plan(
        &brain, "cargo check", 2, &actions, &config, &dtok, &device,
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
    let (brain, dtok, device) = create_test_brain();
    let config = test_pipeline_config();
    let actions = [ACT_REPO_READ, ACT_MEMORY_SEARCH, ACT_END, ACT_END, ACT_END, ACT_END];
    let result = run_with_plan(
        &brain, "Cargo.toml", 3, &actions, &config, &dtok, &device,
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
    let (brain, dtok, device) = create_test_brain();
    let config = test_pipeline_config();
    let result = run_goal(&brain, "hello", &config, &dtok, &device).unwrap();

    // With untrained brain, classification is random but pipeline must not crash
    assert_eq!(result.goal, "hello");
    assert!(result.intent < NUM_INTENTS, "Intent {} out of range", result.intent);
}

#[test]
fn test_full_pipeline_classify() {
    // Verify classify_goal returns valid intent and actions
    let (brain, _dtok, device) = create_test_brain();
    let (intent, actions) = classify_goal(&brain, "run cargo test", &_dtok, &device).unwrap();
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
    let (brain, dtok, device) = create_test_brain();
    let config = test_pipeline_config();
    let actions = [ACT_REPO_READ, ACT_CARGO_CHECK, ACT_END, ACT_END, ACT_END, ACT_END];
    let result = run_with_plan(
        &brain, "/nonexistent/path.rs", 4, &actions, &config, &dtok, &device,
    ).unwrap();

    assert!(!result.success);
    assert_eq!(result.steps.len(), 1, "Should stop after first failure");
    assert!(!result.steps[0].success);
}

// ---------------------------------------------------------------------------
// Cross-session memory recall (T-021)
// ---------------------------------------------------------------------------

#[test]
fn test_cross_session_memory_recall() {
    // Simulate: session 1 stores memories → process restart → session 2 loads them
    let db_path = "/tmp/gestalt_test_cross_session.db";
    let _ = std::fs::remove_file(db_path);

    let d_model = BrainConfig::test_brain().d_model;

    // --- Session 1: store memories ---
    {
        let device = Device::Cpu;
        let config = BrainConfig::test_brain();
        let policy_cfg = PolicyConfig::test();
        let varmap = VarMap::new();
        let mut brain = Brain::new(config, &policy_cfg, &varmap, &device).unwrap();

        // Simulate two interactions
        let cv1 = vec![1.0f32; d_model];
        let cv2 = vec![0.0f32; d_model / 2]
            .into_iter()
            .chain(vec![1.0f32; d_model - d_model / 2])
            .collect::<Vec<_>>();

        brain.store_memory(cv1.clone(), "greeting response".into());
        brain.store_memory(cv2.clone(), "code review response".into());
        assert_eq!(brain.memory_count(), 2);

        // Persist to SQLite (like cmd_serve does after each interaction)
        let episodic = EpisodicMemory::open(db_path, d_model, 1000).unwrap();
        episodic.store(&cv1, "hello", "greeting response", true).unwrap();
        episodic.store(&cv2, "review my code", "code review response", true).unwrap();
        assert_eq!(episodic.len().unwrap(), 2);

        // Brain and episodic memory drop here (simulating process exit)
    }

    // --- Session 2: reload from SQLite ---
    {
        let device = Device::Cpu;
        let config = BrainConfig::test_brain();
        let policy_cfg = PolicyConfig::test();
        let varmap = VarMap::new();
        let mut brain = Brain::new(config, &policy_cfg, &varmap, &device).unwrap();

        // Fresh brain has no memories
        assert_eq!(brain.memory_count(), 0);

        // Load from SQLite (like cmd_serve does on startup)
        let episodic = EpisodicMemory::open(db_path, d_model, 1000).unwrap();
        let records = episodic.retrieve_recent(100).unwrap();
        assert_eq!(records.len(), 2);

        let memories: Vec<(Vec<f32>, String)> = records
            .into_iter()
            .map(|r| (r.concept_vec, r.response))
            .collect();
        brain.load_memories(&memories);

        // Verify memories restored
        assert_eq!(brain.memory_count(), 2);

        // Verify concept vectors survived the roundtrip
        let vecs = brain.export_concept_vecs();
        assert_eq!(vecs.len(), 2);

        // The first loaded memory should have d_model dimensions
        assert_eq!(vecs[0].len(), d_model);
        assert_eq!(vecs[1].len(), d_model);
    }

    let _ = std::fs::remove_file(db_path);
}

#[test]
fn test_episodic_memory_consolidation_across_sessions() {
    // Test that consolidation works on restored data
    let db_path = "/tmp/gestalt_test_consolidation.db";
    let _ = std::fs::remove_file(db_path);

    let d_model = 3; // small for readability

    // Session 1: store similar + different memories
    {
        let episodic = EpisodicMemory::open(db_path, d_model, 1000).unwrap();
        episodic.store(&[1.0, 0.0, 0.0], "hello", "Hi!", true).unwrap();
        episodic.store(&[0.99, 0.01, 0.0], "hey", "Hey!", true).unwrap();
        episodic.store(&[0.0, 0.0, 1.0], "run tests", "Running...", true).unwrap();
        assert_eq!(episodic.len().unwrap(), 3);
    }

    // Session 2: reopen and consolidate
    {
        let episodic = EpisodicMemory::open(db_path, d_model, 1000).unwrap();
        assert_eq!(episodic.len().unwrap(), 3);

        let removed = episodic.consolidate(0.95).unwrap();
        assert_eq!(removed, 1); // "hello" merged into "hey"
        assert_eq!(episodic.len().unwrap(), 2);

        // "hey" (newer) survived, "hello" (older) removed
        let all = episodic.retrieve_top_k(&[1.0, 0.0, 0.0], 10).unwrap();
        assert!(all.iter().any(|r| r.goal == "hey"));
        assert!(!all.iter().any(|r| r.goal == "hello"));
    }

    let _ = std::fs::remove_file(db_path);
}

// ---------------------------------------------------------------------------
// Phase 3: ReAct loop + JarvisSession integration tests
// ---------------------------------------------------------------------------

fn test_executor() -> Executor {
    Executor::new(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        false,
        Duration::from_secs(30),
    )
}

#[test]
fn test_react_loop_conversational() {
    // ReAct loop should handle simple greetings without crashing
    let (brain, dtok, device) = create_test_brain();
    let executor = test_executor();
    let mut session = SessionBuffer::default_session();
    let react = ReactLoop::new(3);

    let result = react.run(
        &brain, "hello", &mut session, &executor, &dtok, &device,
    ).unwrap();

    assert!(!result.final_response.is_empty(), "Should produce a response");
    assert!(!result.steps.is_empty(), "Should have at least one step");
}

#[test]
fn test_jarvis_session_multi_turn() {
    // JarvisSession should maintain conversation context across turns
    let (brain, dtok, device) = create_test_brain();
    let executor = test_executor();
    let mut session = JarvisSession::new();

    // Turn 1
    let r1 = session.process_message(
        "hello", &brain, &executor, &dtok, &device,
    ).unwrap();
    assert!(!r1.is_empty());
    assert_eq!(session.buffer.len(), 2); // user + assistant

    // Turn 2
    let r2 = session.process_message(
        "what can you do", &brain, &executor, &dtok, &device,
    ).unwrap();
    assert!(!r2.is_empty());
    assert_eq!(session.buffer.len(), 4); // 2 user + 2 assistant

    // Verify session context includes both turns
    let ctx = session.buffer.context_string(10, 4096);
    assert!(ctx.contains("hello"), "Context should include first turn");
    assert!(ctx.contains("what can you do"), "Context should include second turn");
}

#[test]
fn test_jarvis_session_buffer_eviction() {
    // SessionBuffer should evict oldest turns when full
    let (brain, dtok, device) = create_test_brain();
    let executor = test_executor();
    let mut session = JarvisSession::new();

    // Fill beyond the 32-turn capacity (16 exchanges = 32 turns)
    for i in 0..18 {
        let _ = session.process_message(
            &format!("message {}", i), &brain, &executor, &dtok, &device,
        );
    }

    // Should be capped at capacity
    assert!(session.buffer.len() <= session.buffer.capacity());
}
