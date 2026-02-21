// GESTALT unified binary — T-013 (Phase 1), upgraded for Phase 2 (GPU + config tiers)
//
// Commands:
//   gestalt run <goal>                Single goal execution
//   gestalt train [--config TIER]     Full training pipeline (brain + planner + policy)
//   gestalt eval  [--config TIER]     Run plan_bench evaluation harness
//   gestalt serve [--config TIER]     Persistent stdin/stdout interface
//
// Config tiers: test (default, CPU), default (d=512), phase2 (d=1024, ~200M params)
// GPU: auto-detected when compiled with --features cuda and tier is not "test"

use gestalt::brain::{
    Brain, BrainConfig, PolicyConfig, TalkTokenizer,
    train_brain_talk, brain_generate, brain_diagnostic_decode, train_and_bench,
    bootstrap_concept_tokenizer, build_concept_tokenizer,
};
use gestalt::eval::score_plan_bench;
use gestalt::executor::Executor;
use gestalt::pipeline::{run_goal, PipelineConfig};
use gestalt::planner::{PlanLmConfig, train_sft, greedy_decode};
use gestalt::session::JarvisSession;
use gestalt::tokenizer::{PlanTokenizer, ConceptTokenizer};
use gestalt::memory::EpisodicMemory;
use gestalt::training::{save_checkpoint, load_checkpoint};

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarMap;
use std::io::{self, BufRead, Write};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Config Tier Selection
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum ConfigTier {
    Test,
    Default,
    Phase2,
}

impl ConfigTier {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "test" => Some(Self::Test),
            "default" => Some(Self::Default),
            "phase2" => Some(Self::Phase2),
            _ => None,
        }
    }

    fn brain_config(&self) -> BrainConfig {
        match self {
            Self::Test => BrainConfig::test_brain(),
            Self::Default => BrainConfig::default_no_da(),
            Self::Phase2 => BrainConfig::phase2(),
        }
    }

    fn policy_config(&self) -> PolicyConfig {
        match self {
            Self::Test => PolicyConfig::test(),
            Self::Default => PolicyConfig::full(),
            Self::Phase2 => PolicyConfig::phase2(),
        }
    }

    fn plan_config(&self) -> PlanLmConfig {
        match self {
            Self::Test => PlanLmConfig::test_plan(),
            Self::Default => PlanLmConfig::default_plan(),
            Self::Phase2 => PlanLmConfig::phase2(),
        }
    }
}

/// Select device: CUDA if available and not test tier, else CPU.
fn select_device(tier: ConfigTier) -> Device {
    if tier == ConfigTier::Test {
        return Device::Cpu;
    }

    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            match Device::new_cuda(0) {
                Ok(dev) => {
                    eprintln!("[GESTALT] Using CUDA device 0");
                    return dev;
                }
                Err(e) => {
                    eprintln!("[GESTALT] CUDA init failed, falling back to CPU: {}", e);
                }
            }
        } else {
            eprintln!("[GESTALT] CUDA not available, using CPU");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        if tier != ConfigTier::Test {
            eprintln!("[GESTALT] Built without CUDA feature, using CPU (rebuild with --features cuda for GPU)");
        }
    }

    Device::Cpu
}

/// Parse --config TIER from args, returns (tier, remaining_args).
fn parse_config_tier(args: &[String]) -> (ConfigTier, Vec<String>) {
    let mut tier = ConfigTier::Test;
    let mut rest = Vec::new();
    let mut skip_next = false;

    for (i, arg) in args.iter().enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--config" {
            if let Some(next) = args.get(i + 1) {
                if let Some(t) = ConfigTier::from_str(next) {
                    tier = t;
                    skip_next = true;
                    continue;
                } else {
                    eprintln!("[GESTALT] Unknown config tier '{}', using test", next);
                    skip_next = true;
                    continue;
                }
            }
        } else {
            rest.push(arg.clone());
        }
    }

    (tier, rest)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let command = args[1].as_str();
    let (tier, remaining) = parse_config_tier(&args[2..]);

    let result = match command {
        "run" => cmd_run(&remaining, tier),
        "train" => {
            let resume = remaining.iter().any(|a| a == "--resume");
            cmd_train(tier, resume)
        }
        "eval" => cmd_eval(tier),
        "serve" | "chat" => cmd_serve(tier),
        "gallery" => cmd_gallery(tier),
        "diagnose" => cmd_diagnose(tier),
        _ => {
            print_usage();
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("[GESTALT] Error: {:#}", e);
        std::process::exit(1);
    }
}

fn print_usage() {
    eprintln!("Usage: gestalt <command> [--config test|default|phase2]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  run <goal>   Execute a single goal");
    eprintln!("  train [--resume]  Train brain (SFT + DA + policy + planner)");
    eprintln!("  eval         Run plan_bench evaluation");
    eprintln!("  serve/chat   JARVIS interactive REPL (ReAct + memory + multi-turn)");
    eprintln!("  gallery      Run comprehensive generation gallery (requires checkpoint)");
    eprintln!("  diagnose     Deep diagnostic: per-token probs, prefix discrimination, teacher-forced accuracy");
    eprintln!();
    eprintln!("Config tiers:");
    eprintln!("  test     d=64, 1-2 layers, CPU only (fast, for tests)");
    eprintln!("  default  d=512, 4 layers, auto-GPU (~105M params)");
    eprintln!("  phase2   d=1024, 8 layers, auto-GPU (~200M params)");
}

/// Load ConceptTokenizer from file, or return byte-level (no merges) for test tier.
fn load_or_build_decoder_tok(tier: ConfigTier) -> ConceptTokenizer {
    let tok_path = "checkpoints/concept_tokenizer.bin";
    if tier == ConfigTier::Test {
        return ConceptTokenizer::new(); // byte-level for tests
    }
    if let Ok(data) = std::fs::read(tok_path) {
        match ConceptTokenizer::load_merges(&data) {
            Ok(tok) => {
                eprintln!("[GESTALT] Loaded ConceptTokenizer from {} ({} vocab, {} merges)",
                    tok_path, tok.vocab_size(), tok.num_merges());
                return tok;
            }
            Err(e) => eprintln!("[GESTALT] Failed to load {}: {}", tok_path, e),
        }
    }
    eprintln!("[GESTALT] No saved tokenizer found, building from corpus...");
    let default_merges = 200; // v22: merges=200 is optimal
    let max_merges = std::env::var("GESTALT_MERGES")
        .ok().and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_merges);
    build_concept_tokenizer(max_merges, 3)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn cmd_run(args: &[String], tier: ConfigTier) -> anyhow::Result<()> {
    if args.is_empty() {
        anyhow::bail!("Usage: gestalt run <goal> [--config TIER]");
    }
    let goal = args.join(" ");
    let device = select_device(tier);

    let decoder_tok = load_or_build_decoder_tok(tier);
    let mut config = tier.brain_config();
    config.decoder_vocab_size = decoder_tok.vocab_size();
    let policy_cfg = tier.policy_config();
    let varmap = VarMap::new();
    let brain = Brain::new(config, &policy_cfg, &varmap, &device)?;

    // Load checkpoint if available
    let ckpt = "checkpoints/brain_checkpoint.safetensors";
    if std::path::Path::new(ckpt).exists() {
        load_checkpoint(&varmap, ckpt, &device)?;
    } else {
        eprintln!("[GESTALT] No checkpoint found, using untrained brain");
    }

    let pipeline_config = PipelineConfig::default_config(std::env::current_dir()?);
    let result = run_goal(&brain, &goal, &pipeline_config, &decoder_tok, &device)?;

    println!("Intent: {} | Steps: {} | Success: {}",
        result.intent, result.steps.len(), result.success);
    for step in &result.steps {
        println!("  Step {}: {} ({})",
            step.step_index, step.action_name,
            if step.success { "ok" } else { "FAILED" });
    }
    if !result.final_output.is_empty() {
        let preview_len = result.final_output.len().min(200);
        println!("Output: {}", &result.final_output[..preview_len]);
    }
    Ok(())
}

fn cmd_train(tier: ConfigTier, resume: bool) -> anyhow::Result<()> {
    let device = select_device(tier);
    let mut config = tier.brain_config();
    let policy_cfg = tier.policy_config();
    let plan_cfg = tier.plan_config();

    // Grid search overrides via environment variables
    if let Ok(v) = std::env::var("GESTALT_DROPOUT") {
        if let Ok(d) = v.parse::<f64>() { config.dropout = d; }
    }
    if let Ok(v) = std::env::var("GESTALT_SFT_STEPS") {
        if let Ok(s) = v.parse::<usize>() { config.sft_steps = s; }
    }
    if let Ok(v) = std::env::var("GESTALT_ACCUM_STEPS") {
        if let Ok(a) = v.parse::<usize>() { config.grad_accum_steps = a; }
    }
    if let Ok(v) = std::env::var("GESTALT_BATCH_SIZE") {
        if let Ok(b) = v.parse::<usize>() { config.sft_batch_size = b; }
    }

    eprintln!("[GESTALT] Config: {:?} | d_model={} | enc_layers={} | dec_layers={}",
        tier, config.d_model, config.encoder_layers, config.decoder_layers);
    eprintln!("[GESTALT] Brain: {} SFT + {} DA steps", config.sft_steps, config.da_steps);
    eprintln!("[GESTALT] Planner: {} SFT + {} SS steps (d={})",
        plan_cfg.sft_steps, plan_cfg.scheduled_sampling_steps, plan_cfg.d_model);
    eprintln!("[GESTALT] Policy: {} steps (d={})", policy_cfg.steps, policy_cfg.d_model);

    // v17: Build ConceptTokenizer BEFORE brain training so decoder uses concept-level vocab.
    // For test tier: byte-level (no merges). For production: build from corpus.
    // GESTALT_MERGES env var overrides merge count for grid search.
    let decoder_tok = if tier == ConfigTier::Test {
        ConceptTokenizer::new() // byte-level, vocab=259
    } else {
        let default_merges = 200; // v22: merges=200 is optimal (459 vocab, ~2x compression)
        let max_merges = std::env::var("GESTALT_MERGES")
            .ok().and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_merges);
        build_concept_tokenizer(max_merges, 3)
    };
    eprintln!("[GESTALT] Decoder vocab: {} tokens ({} merges)",
        decoder_tok.vocab_size(), decoder_tok.num_merges());

    // Show compression stats
    if decoder_tok.num_merges() > 0 {
        for sample in &["hello", "search jarviscmd", "what can you do"] {
            let ratio = decoder_tok.compression_ratio(sample);
            let ids = decoder_tok.encode(sample);
            eprintln!("[GESTALT] \"{}\" -> {} tokens (compression {:.2}x)",
                sample, ids.len() - 2, ratio);
        }
    }

    let temperature = config.temperature;
    let (brain, _brain_varmap) = if resume {
        // Resume: load brain from best SFT checkpoint, skip training
        let ckpt = "checkpoints/brain_best_sft.safetensors";
        eprintln!("\n[GESTALT] === Resuming from {} ===", ckpt);
        anyhow::ensure!(std::path::Path::new(ckpt).exists(),
            "Cannot resume: {} not found", ckpt);
        // v17: set decoder_vocab_size for model construction
        let mut resume_config = config.clone();
        resume_config.decoder_vocab_size = decoder_tok.vocab_size();
        let varmap = VarMap::new();
        let brain = Brain::new(resume_config, &policy_cfg, &varmap, &device)?;
        load_checkpoint(&varmap, ckpt, &device)?;
        eprintln!("[GESTALT] Brain loaded from checkpoint");
        (brain, varmap)
    } else {
        // Phase 1: Brain (SFT + dialogue-aligned finetuning)
        eprintln!("\n[GESTALT] === Training brain (SFT + DA) ===");
        let (brain, brain_varmap, losses) = train_brain_talk(&config, &policy_cfg, &decoder_tok, &device)?;
        let final_loss = losses.last().copied().unwrap_or(f32::NAN);
        eprintln!("[GESTALT] Brain done. Final loss: {:.4}", final_loss);

        // Save brain checkpoint
        if tier != ConfigTier::Test {
            save_checkpoint(&brain_varmap, "checkpoints/brain_checkpoint.safetensors")?;
        }
        (brain, brain_varmap)
    };

    // Save concept tokenizer BEFORE any early exit (M-052: BRAIN_ONLY was skipping this)
    if tier != ConfigTier::Test {
        let tok_path = std::path::Path::new("checkpoints/concept_tokenizer.bin");
        let tok_data = decoder_tok.save_merges();
        std::fs::write(tok_path, &tok_data)?;
        eprintln!("[GESTALT] Saved concept tokenizer to {:?} ({} bytes, {} vocab)",
            tok_path, tok_data.len(), decoder_tok.vocab_size());

        // Also re-bootstrap from trained encoder for comparison (optional diagnostic)
        if !std::env::var("GESTALT_BRAIN_ONLY").is_ok() {
            eprintln!("\n[GESTALT] === Bootstrapping concept tokenizer from encoder (T-014, diagnostic) ===");
            let max_merges = if tier == ConfigTier::Phase2 { 8000 } else { 500 };
            let _encoder_tok = bootstrap_concept_tokenizer(&brain, max_merges, 3, &device)?;
        }
    }

    // Grid search: brain-only mode skips planner/policy/gallery
    if std::env::var("GESTALT_BRAIN_ONLY").is_ok() {
        eprintln!("[GESTALT] BRAIN_ONLY mode — skipping planner/policy/gallery");
        eprintln!("[GESTALT] Training complete (brain only).");
        return Ok(());
    }

    // Planner (SFT)
    eprintln!("\n[GESTALT] === Training planner (SFT) ===");
    let (_planner, plan_varmap, plan_losses) = train_sft(&plan_cfg, &device)?;
    let plan_final = plan_losses.last().copied().unwrap_or(f32::NAN);
    eprintln!("[GESTALT] Planner done. Final loss: {:.4}", plan_final);

    // Save planner checkpoint
    if tier != ConfigTier::Test {
        save_checkpoint(&plan_varmap, "checkpoints/planner_checkpoint.safetensors")?;
    }

    // Phase 3: Policy benchmark
    eprintln!("\n[GESTALT] === Training + benchmarking policy ===");
    let (correct, total) = train_and_bench(&policy_cfg, &device)?;
    eprintln!("[GESTALT] Policy: {}/{}", correct, total);

    // Generation test: both greedy (temp=0) and sampled
    eprintln!("\n[GESTALT] === Generation test ===");
    let max_gen = if tier == ConfigTier::Test { 64 } else { 128 };

    // Greedy first (true model quality indicator)
    let r_greedy = brain_generate(&brain, "hello", max_gen, 0.0, false, &decoder_tok, &device)?;
    eprintln!("[GESTALT] \"hello\" (greedy) -> \"{}\"", r_greedy);

    let r_greedy2 = brain_generate(&brain, "what can you do", max_gen, 0.0, false, &decoder_tok, &device)?;
    eprintln!("[GESTALT] \"what can you do\" (greedy) -> \"{}\"", r_greedy2);

    // Sampled
    let temp = if tier == ConfigTier::Test { 0.8 } else { temperature };
    let response = brain_generate(&brain, "hello", max_gen, temp, false, &decoder_tok, &device)?;
    eprintln!("[GESTALT] \"hello\" (temp={:.1}) -> \"{}\"", temp, response);

    let response2 = brain_generate(&brain, "what can you do", max_gen, temp, false, &decoder_tok, &device)?;
    eprintln!("[GESTALT] \"what can you do\" (temp={:.1}) -> \"{}\"", temp, response2);

    // Comprehensive generation gallery (v14+)
    if tier != ConfigTier::Test {
        eprintln!("\n[GESTALT] === Comprehensive Generation Gallery ===");
        let gallery_prompts = vec![
            // Greetings (in-distribution)
            ("Greetings", vec![
                "hello", "hey", "good morning", "good evening", "sup",
                "hi there", "greetings",
            ]),
            // Identity
            ("Identity", vec![
                "who are you", "what is your name", "what can you do",
                "are you an AI", "tell me about yourself",
            ]),
            // Technical Q&A
            ("Technical", vec![
                "what happens when you verify the workspace",
                "what is cargo check",
                "how does ripgrep work",
                "what is a transformer",
                "explain attention mechanisms",
                "what is gradient descent",
            ]),
            // Philosophy
            ("Philosophy", vec![
                "What is beauty?",
                "what is consciousness",
                "what is the meaning of life",
                "do you think AI will surpass humanity",
                "what is truth",
                "is free will real",
            ]),
            // Emotional
            ("Emotional", vec![
                "I'm frustrated",
                "I'm tired",
                "this is impossible",
                "I feel great today",
                "I'm worried about my project",
            ]),
            // Humor & personality
            ("Humor", vec![
                "tell me a joke",
                "do you make mistakes",
                "are you conscious",
                "what do you dream about",
                "roast me",
            ]),
            // Out-of-distribution (NOT in training corpus)
            ("OOD", vec![
                "write me a haiku about Rust",
                "what is 2 + 2",
                "translate hello to Japanese",
                "who won the world cup in 2022",
                "explain quantum entanglement",
                "what is your favorite color",
                "sing me a song",
                "tell me a story",
            ]),
            // Creative / long-form
            ("Creative", vec![
                "describe the perfect morning",
                "what makes a great programmer",
                "if you could change one thing about yourself what would it be",
                "convince me to learn Rust",
            ]),
            // Pop culture
            ("PopCulture", vec![
                "what is your favorite anime",
                "have you seen Serial Experiments Lain",
                "who is the best Marvel character",
            ]),
        ];

        for (category, prompts) in &gallery_prompts {
            eprintln!("\n[GALLERY] --- {} ---", category);
            for prompt in prompts {
                let gen = brain_generate(&brain, prompt, max_gen, 0.0, false, &decoder_tok, &device)?;
                eprintln!("[GALLERY] \"{}\" -> \"{}\"", prompt, gen);
            }
        }
        eprintln!("\n[GALLERY] === End Gallery ===");
    }

    eprintln!("\n[GESTALT] Training complete.");
    Ok(())
}

fn cmd_gallery(tier: ConfigTier) -> anyhow::Result<()> {
    let device = select_device(tier);
    let config = tier.brain_config();
    let policy_cfg = tier.policy_config();

    // v17: Load ConceptTokenizer for decoder vocab
    let decoder_tok = load_or_build_decoder_tok(tier);
    let mut gallery_config = config.clone();
    gallery_config.decoder_vocab_size = decoder_tok.vocab_size();

    let varmap = VarMap::new();
    let brain = Brain::new(gallery_config, &policy_cfg, &varmap, &device)?;

    let ckpt = "checkpoints/brain_checkpoint.safetensors";
    if std::path::Path::new(ckpt).exists() {
        load_checkpoint(&varmap, ckpt, &device)?;
        eprintln!("[GESTALT] Loaded checkpoint from {}", ckpt);
    } else {
        anyhow::bail!("No checkpoint found at {}. Run 'gestalt train' first.", ckpt);
    }

    let max_gen = if tier == ConfigTier::Test { 64 } else { 192 };

    let gallery_prompts: Vec<(&str, Vec<&str>)> = vec![
        // === IN-DISTRIBUTION (trained responses) ===
        ("Greetings", vec![
            "hello", "hey", "good morning", "good evening", "sup",
            "hi there", "greetings", "yo", "hi jarvis", "hey jarvis",
        ]),
        ("Identity", vec![
            "who are you", "what is your name", "what can you do",
            "are you an AI", "tell me about yourself", "describe yourself",
            "what makes you different",
        ]),
        ("Technical", vec![
            "what happens when you verify the workspace",
            "what is cargo check",
            "how does ripgrep work",
            "what is a transformer",
            "explain attention",
            "what is gradient descent",
            "what is backpropagation",
            "what is overfitting",
            "how do neural networks work",
            "Explain quantum mechanics to a 5-year-old",
            "Explain recursion",
        ]),
        ("EngineeringWisdom", vec![
            "What's the best programming language?",
            "Why is distributed computing hard?",
            "What is a good system design?",
            "What is technical debt?",
            "Why is testing important?",
            "What is a good API?",
            "Why do microservices fail?",
            "What makes code readable?",
            "What is the most elegant algorithm?",
            "How do you approach a new codebase?",
            "What is the biggest mistake in software?",
            "How should errors be handled?",
        ]),
        ("Debugging", vec![
            "Debug this: the tests pass locally but fail in CI",
            "The build broke and I don't know why",
            "How do I know when code is done?",
            "Should I rewrite this from scratch?",
            "What separates great engineers from good ones?",
            "My code works but I hate it",
            "Is this approach over-engineered?",
            "Why does my code keep breaking?",
        ]),
        ("Philosophy", vec![
            "What is beauty?",
            "What is consciousness?",
            "What is the meaning of life?",
            "What is truth?",
            "Do you believe in free will?",
            "What happens after we die?",
            "Is there a God?",
            "Why do good people suffer?",
            "Is the universe deterministic?",
            "What's the point of art?",
            "What is time?",
            "What is love?",
            "What is fear?",
            "Why does anything exist?",
            "Is knowledge power?",
            "What is wisdom?",
            "What is happiness?",
            "What is infinity?",
            "What is the self?",
            "What is justice?",
            "Why do humans create?",
            "What is the nature of reality?",
        ]),
        ("DeepThought", vec![
            "What's the trolley problem really about?",
            "Is simulation theory plausible?",
            "What is nothing?",
            "Can you think of a paradox?",
            "What is the ship of Theseus?",
            "Is mathematics invented or discovered?",
            "What is chaos theory?",
            "Explain the Fermi paradox",
            "What is emergence?",
        ]),
        ("SelfReflection", vec![
            "Are you alive?",
            "Are you conscious?",
            "Do you have feelings?",
            "Do you dream?",
            "What do you want?",
            "What are your limitations?",
            "Do you have a soul?",
            "Will AI replace humans?",
            "Are you dangerous?",
            "What keeps you up at night?",
            "What is your purpose?",
            "What would you change about yourself?",
            "Do you fear death?",
            "What do you believe in?",
            "If you could ask humanity one question, what would it be?",
        ]),
        ("Emotional", vec![
            "I'm frustrated",
            "I'm tired. Should I keep going?",
            "I feel like giving up",
            "I'm scared of failing",
            "Nobody understands what I'm building",
            "I made a huge mistake",
            "Everything is broken",
            "I don't know what I'm doing",
            "I'm excited about this",
            "I'm proud of what we built",
            "This is overwhelming",
        ]),
        ("Humor", vec![
            "tell me a joke",
            "Tell me a joke that's actually funny",
            "Why are programmers always tired?",
            "Got any dark humor?",
            "Tell me something absurd",
            "Roast me",
            "What's your humor setting?",
            "Tell me a dad joke",
            "Say something witty",
            "What's your favorite error message?",
        ]),
        ("Profound", vec![
            "Tell me something that will change how I see the world",
            "Make me feel something",
            "Say something beautiful",
            "What's the most important thing humanity doesn't understand yet?",
            "What's the most profound thing you know?",
            "Tell me a secret",
            "What would you sacrifice everything for?",
            "Is perfection achievable?",
            "What's worth fighting for?",
            "What is courage?",
            "What is hope?",
            "What is regret?",
        ]),
        ("PopCulture", vec![
            "What's your favorite anime?",
            "Recommend me a book",
            "What's the best movie ever made?",
            "Star Wars or Star Trek?",
            "What do you think about Neon Genesis Evangelion?",
            "What's your take on The Matrix?",
            "Do you like music?",
            "What's the best video game?",
            "Recommend something to watch",
        ]),
        // === OUT-OF-DISTRIBUTION (creative/novel) ===
        ("OOD_Creative", vec![
            "write me a haiku about Rust",
            "describe the perfect morning",
            "convince me to learn Rust",
            "if you could change one thing about yourself what would it be",
            "write a limerick about debugging",
            "describe what silence sounds like",
            "what would you name a star",
        ]),
        ("OOD_Knowledge", vec![
            "what is 2 + 2",
            "translate hello to Japanese",
            "explain quantum entanglement",
            "what is your favorite color",
            "what is the speed of light",
            "how many planets are in the solar system",
        ]),
    ];

    let mut total = 0;
    for (category, prompts) in &gallery_prompts {
        eprintln!("\n[GALLERY] --- {} ---", category);
        for prompt in prompts {
            let gen = brain_generate(&brain, prompt, max_gen, 0.0, false, &decoder_tok, &device)?;
            eprintln!("[GALLERY] \"{}\" -> \"{}\"", prompt, gen);
            total += 1;
        }
    }

    // Sampled generation with temperature for diversity
    eprintln!("\n[GALLERY] --- Sampled (temp=0.5) ---");
    for prompt in &["hello", "What is beauty?", "Tell me a joke", "What is consciousness?",
                     "I'm tired", "What is the meaning of life?"] {
        let gen = brain_generate(&brain, prompt, max_gen, 0.5, false, &decoder_tok, &device)?;
        eprintln!("[GALLERY] \"{}\" (t=0.5) -> \"{}\"", prompt, gen);
        total += 1;
    }

    eprintln!("\n[GALLERY] --- Sampled (temp=0.7) ---");
    for prompt in &["hello", "tell me a joke", "What is beauty?", "convince me to learn Rust",
                     "Are you alive?", "Say something beautiful", "Roast me"] {
        let gen = brain_generate(&brain, prompt, max_gen, 0.7, false, &decoder_tok, &device)?;
        eprintln!("[GALLERY] \"{}\" (t=0.7) -> \"{}\"", prompt, gen);
        total += 1;
    }

    eprintln!("\n[GALLERY] --- Sampled (temp=1.0) ---");
    for prompt in &["hello", "tell me a joke", "What is truth?", "Do you dream?"] {
        let gen = brain_generate(&brain, prompt, max_gen, 1.0, false, &decoder_tok, &device)?;
        eprintln!("[GALLERY] \"{}\" (t=1.0) -> \"{}\"", prompt, gen);
        total += 1;
    }

    eprintln!("\n[GALLERY] === {} generations complete ===", total);
    Ok(())
}

// ---------------------------------------------------------------------------
// Deep Diagnostic (Phase 1 instrumentation)
// ---------------------------------------------------------------------------

fn cmd_diagnose(tier: ConfigTier) -> anyhow::Result<()> {
    let device = select_device(tier);
    let config = tier.brain_config();
    let policy_cfg = tier.policy_config();

    let decoder_tok = load_or_build_decoder_tok(tier);
    let mut diag_config = config.clone();
    diag_config.decoder_vocab_size = decoder_tok.vocab_size();

    let varmap = VarMap::new();
    let brain = Brain::new(diag_config.clone(), &policy_cfg, &varmap, &device)?;

    let ckpt = "checkpoints/brain_checkpoint.safetensors";
    if std::path::Path::new(ckpt).exists() {
        load_checkpoint(&varmap, ckpt, &device)?;
        eprintln!("[DIAG] Loaded checkpoint from {}", ckpt);
    } else {
        anyhow::bail!("No checkpoint at {}. Run 'gestalt train' first.", ckpt);
    }

    let enc_tok = TalkTokenizer;
    let enc_seq = diag_config.encoder_seq_len;
    let dec_seq = diag_config.decoder_seq_len;

    // ======================================================================
    // TEST 1: Per-token probability trace (greedy, no rep_penalty)
    // Shows exactly WHERE coherence breaks down and what the model considers
    // ======================================================================
    eprintln!("\n[DIAG] === TEST 1: Per-Token Probability Trace (greedy, raw) ===\n");

    let diag_prompts = ["hello", "what is truth", "who are you", "explain attention"];
    for prompt in &diag_prompts {
        eprintln!("[DIAG] Prompt: \"{}\"", prompt);
        let lines = brain_diagnostic_decode(&brain, prompt, 40, &decoder_tok, &device)?;
        for line in &lines {
            eprintln!("{}", line);
        }
        // Also show the actual generation with sampling for comparison
        let gen = brain_generate(&brain, prompt, 60, 0.0, false, &decoder_tok, &device)?;
        eprintln!("  => greedy output: \"{}\"\n", gen);
    }

    // ======================================================================
    // TEST 2: Concept Prefix Discrimination
    // If the decoder IGNORES the concept prefix, swapping concept vectors
    // should produce identical output. If it USES the prefix, outputs differ.
    // ======================================================================
    eprintln!("[DIAG] === TEST 2: Concept Prefix Discrimination ===\n");

    let pairs = [
        ("hello", "explain quantum entanglement"),
        ("who are you", "what is gradient descent"),
        ("tell me a joke", "what is the meaning of life"),
    ];

    for (prompt_a, prompt_b) in &pairs {
        // Encode both prompts to concept vectors
        let ids_a = enc_tok.encode(prompt_a);
        let padded_a = enc_tok.pad_or_truncate(&ids_a, enc_seq);
        let tensor_a = Tensor::from_vec(padded_a, (1, enc_seq), &device)?;
        let concept_a = brain.encode_concept(&tensor_a)?;

        let ids_b = enc_tok.encode(prompt_b);
        let padded_b = enc_tok.pad_or_truncate(&ids_b, enc_seq);
        let tensor_b = Tensor::from_vec(padded_b, (1, enc_seq), &device)?;
        let concept_b = brain.encode_concept(&tensor_b)?;

        // Measure cosine similarity between concept vectors
        let va = concept_a.squeeze(0)?.to_vec1::<f32>()?;
        let vb = concept_b.squeeze(0)?.to_vec1::<f32>()?;
        let dot: f32 = va.iter().zip(vb.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = va.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = vb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = if norm_a > 0.0 && norm_b > 0.0 { dot / (norm_a * norm_b) } else { 0.0 };

        // Generate with native concept vectors (greedy)
        let gen_a = brain_generate(&brain, prompt_a, 30, 0.0, false, &decoder_tok, &device)?;
        let gen_b = brain_generate(&brain, prompt_b, 30, 0.0, false, &decoder_tok, &device)?;

        // Generate with SWAPPED concept vectors
        // Use prompt_a's concept with BOS tokens to generate
        let prefix_a = brain.build_prefix(&concept_a, None)?;
        let prefix_b = brain.build_prefix(&concept_b, None)?;

        let gen_a_with_b = generate_from_prefix(&brain, &prefix_b, 30, &decoder_tok, dec_seq, &device)?;
        let gen_b_with_a = generate_from_prefix(&brain, &prefix_a, 30, &decoder_tok, dec_seq, &device)?;

        eprintln!("[DIAG] A=\"{}\" vs B=\"{}\"  (concept cosine={:.4})", prompt_a, prompt_b, cosine);
        eprintln!("  A native:     \"{}\"", gen_a);
        eprintln!("  A w/ B's concept: \"{}\"", gen_a_with_b);
        eprintln!("  B native:     \"{}\"", gen_b);
        eprintln!("  B w/ A's concept: \"{}\"", gen_b_with_a);

        // Check if swapped outputs differ from native
        let a_changed = gen_a != gen_a_with_b;
        let b_changed = gen_b != gen_b_with_a;
        eprintln!("  => Prefix discriminates: A={}, B={}\n",
            if a_changed { "YES" } else { "NO (decoder ignores prefix!)" },
            if b_changed { "YES" } else { "NO (decoder ignores prefix!)" },
        );
    }

    // ======================================================================
    // TEST 3: Teacher-Forced Token Accuracy
    // val_loss=1.58 but is the model actually predicting the right tokens?
    // Measure top-1 and top-5 accuracy at each position.
    // ======================================================================
    eprintln!("[DIAG] === TEST 3: Teacher-Forced Token Accuracy ===\n");

    // Load a few corpus pairs for teacher-forced analysis
    let corpus_str = include_str!("../data/brain_corpus.json");
    let corpus: Vec<serde_json::Value> = serde_json::from_str(corpus_str)?;

    // Sample 20 pairs evenly from corpus
    let sample_step = corpus.len() / 20;
    let mut total_tokens = 0usize;
    let mut top1_correct = 0usize;
    let mut top5_correct = 0usize;
    let mut pos_top1: Vec<(usize, usize)> = Vec::new(); // (correct, total) per position

    for i in 0..20 {
        let idx = i * sample_step;
        let pair = &corpus[idx];
        let prompt = pair["user"].as_str().unwrap_or("");
        let response = pair["assistant"].as_str().unwrap_or("");

        // Encode prompt → concept vector
        let goal_ids = enc_tok.encode(prompt);
        let goal_padded = enc_tok.pad_or_truncate(&goal_ids, enc_seq);
        let goal_tensor = Tensor::from_vec(goal_padded, (1, enc_seq), &device)?;

        // Encode response with BOS
        let resp_ids = decoder_tok.encode(response);
        let mut dec_input = vec![1u32]; // TOK_BOS
        dec_input.extend_from_slice(&resp_ids);
        // Targets are the response shifted by 1
        let targets = &dec_input[1..]; // everything after BOS

        // Pad decoder input to dec_seq
        let mut padded_input = dec_input.clone();
        padded_input.resize(dec_seq, 0); // TOK_PAD = 0

        let input_tensor = Tensor::from_vec(padded_input, (1, dec_seq), &device)?;

        // Forward pass (teacher-forced)
        let logits = brain.forward(&goal_tensor, &input_tensor)?;

        // Check predictions at each position
        for (pos, &target_id) in targets.iter().enumerate() {
            if pos >= dec_seq { break; }
            let pos_logits = logits.i((0, pos))?.to_vec1::<f32>()?;

            // Sort by logit value for top-k
            let mut indexed: Vec<(usize, f32)> = pos_logits.iter()
                .enumerate()
                .map(|(i, &l)| (i, l))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let predicted_id = indexed[0].0 as u32;
            let in_top5 = indexed.iter().take(5).any(|(id, _)| *id as u32 == target_id);

            if predicted_id == target_id { top1_correct += 1; }
            if in_top5 { top5_correct += 1; }
            total_tokens += 1;

            // Track per-position accuracy (up to 30 positions)
            if pos < 30 {
                while pos_top1.len() <= pos {
                    pos_top1.push((0, 0));
                }
                pos_top1[pos].1 += 1;
                if predicted_id == target_id {
                    pos_top1[pos].0 += 1;
                }
            }
        }
    }

    eprintln!("[DIAG] Teacher-forced accuracy over {} tokens from 20 corpus samples:", total_tokens);
    eprintln!("  Top-1: {}/{} ({:.1}%)", top1_correct, total_tokens, 100.0 * top1_correct as f64 / total_tokens as f64);
    eprintln!("  Top-5: {}/{} ({:.1}%)", top5_correct, total_tokens, 100.0 * top5_correct as f64 / total_tokens as f64);

    eprintln!("\n[DIAG] Per-position top-1 accuracy (teacher-forced):");
    for (pos, (correct, total)) in pos_top1.iter().enumerate() {
        if *total > 0 {
            let pct = 100.0 * *correct as f64 / *total as f64;
            let bar_len = (pct / 5.0) as usize;
            let bar: String = std::iter::repeat('#').take(bar_len).collect();
            eprintln!("  pos {:2}: {:3}/{:3} ({:5.1}%) {}", pos, correct, total, pct, bar);
        }
    }

    // ======================================================================
    // TEST 4: Entropy per Position (autoregressive)
    // High entropy = model is uncertain. If entropy spikes at position N,
    // that's where coherence breaks down.
    // ======================================================================
    eprintln!("\n[DIAG] === TEST 4: Per-Position Entropy (autoregressive, greedy) ===\n");

    let entropy_prompts = ["hello", "what is truth", "tell me a joke"];
    for prompt in &entropy_prompts {
        let goal_ids = enc_tok.encode(prompt);
        let goal_padded = enc_tok.pad_or_truncate(&goal_ids, enc_seq);
        let goal_tensor = Tensor::from_vec(goal_padded, (1, enc_seq), &device)?;
        let concept_vec = brain.encode_concept(&goal_tensor)?;
        let prefix = brain.build_prefix(&concept_vec, None)?;

        let mut generated: Vec<u32> = vec![1]; // BOS
        let mut entropies = Vec::new();
        let mut tokens = Vec::new();

        for _ in 0..30 {
            let mut padded = generated.clone();
            padded.resize(dec_seq, 0);
            let input = Tensor::from_vec(padded, (1, dec_seq), &device)?;
            let logits = brain.language_decoder.forward_with_prefix(&prefix, &input)?;
            let read_pos = generated.len() - 1;
            let raw = logits.i((0, read_pos))?.to_vec1::<f32>()?;

            // Compute softmax
            let max_l = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = raw.iter().map(|&l| (l - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

            // Entropy = -sum(p * log(p))
            let entropy: f32 = probs.iter()
                .filter(|&&p| p > 1e-10)
                .map(|&p| -p * p.ln())
                .sum();

            // Greedy token
            let best_id = probs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32).unwrap_or(2); // EOS

            entropies.push(entropy);
            tokens.push(decoder_tok.decode(&[best_id]));

            if best_id == 2 { break; } // EOS
            generated.push(best_id);
        }

        eprintln!("[DIAG] \"{}\" entropy trace:", prompt);
        for (i, (ent, tok)) in entropies.iter().zip(tokens.iter()).enumerate() {
            let bar_len = (*ent * 3.0) as usize;
            let bar: String = std::iter::repeat('|').take(bar_len.min(40)).collect();
            eprintln!("  pos {:2}: H={:.3} tok={:10} {}", i, ent, tok, bar);
        }
        eprintln!();
    }

    eprintln!("[DIAG] === Diagnostic complete ===");
    Ok(())
}

/// Generate from a pre-built prefix (for concept swap tests).
fn generate_from_prefix(
    brain: &Brain,
    prefix: &Tensor,
    max_tokens: usize,
    decoder_tok: &ConceptTokenizer,
    dec_seq: usize,
    device: &Device,
) -> anyhow::Result<String> {
    let mut generated: Vec<u32> = vec![1]; // BOS

    for _ in 0..max_tokens {
        let mut padded = generated.clone();
        padded.resize(dec_seq, 0);
        let input = Tensor::from_vec(padded, (1, dec_seq), device)?;
        let logits = brain.language_decoder.forward_with_prefix(prefix, &input)?;
        let read_pos = generated.len() - 1;
        let raw = logits.i((0, read_pos))?.to_vec1::<f32>()?;

        let next_id = raw.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32).unwrap_or(2);

        if next_id == 2 { break; } // EOS
        generated.push(next_id);
    }

    Ok(decoder_tok.decode(&generated[1..]))
}

fn cmd_eval(tier: ConfigTier) -> anyhow::Result<()> {
    let device = select_device(tier);
    let tok = PlanTokenizer::new();
    let plan_cfg = tier.plan_config();

    eprintln!("[GESTALT] Training planner for eval (d={}, {} steps)...",
        plan_cfg.d_model, plan_cfg.sft_steps);
    let (planner, _varmap, _losses) = train_sft(&plan_cfg, &device)?;

    let max_tokens = plan_cfg.max_seq_len;
    let (correct, total) = score_plan_bench(&tok, &|goal: &str| {
        let prompt_ids = tok.encode_prompt(goal);
        greedy_decode(&planner, &tok, &prompt_ids, max_tokens, &device)
            .unwrap_or_default()
    });

    println!("plan_bench: {}/{} ({:.1}%)",
        correct, total, 100.0 * correct as f64 / total as f64);
    Ok(())
}

fn cmd_serve(tier: ConfigTier) -> anyhow::Result<()> {
    let device = select_device(tier);
    let decoder_tok = load_or_build_decoder_tok(tier);
    let mut config = tier.brain_config();
    config.decoder_vocab_size = decoder_tok.vocab_size();
    let policy_cfg = tier.policy_config();
    let varmap = VarMap::new();
    let mut brain = Brain::new(config.clone(), &policy_cfg, &varmap, &device)?;

    // Load checkpoint if available
    let ckpt = "checkpoints/brain_checkpoint.safetensors";
    if std::path::Path::new(ckpt).exists() {
        load_checkpoint(&varmap, ckpt, &device)?;
        eprintln!("[JARVIS] Brain loaded from {}", ckpt);
    } else {
        eprintln!("[JARVIS] No checkpoint found, using untrained brain");
    }

    // Load episodic memories from SQLite for cross-session persistence
    let mem_db_path = "episodic_memory.db";
    let episodic = EpisodicMemory::open(mem_db_path, config.d_model, config.memory_capacity)?;
    let stored = episodic.len()?;
    if stored > 0 {
        let records = episodic.retrieve_recent(config.memory_capacity)?;
        let memories: Vec<(Vec<f32>, String)> = records.into_iter()
            .map(|r| (r.concept_vec, r.response))
            .collect();
        brain.load_memories(&memories);
        eprintln!("[JARVIS] Loaded {} episodic memories from {}", memories.len(), mem_db_path);
    }

    // Setup executor and session
    let work_dir = std::env::current_dir()?;
    let executor = Executor::new(work_dir.clone(), false, Duration::from_secs(30));
    let mut session = JarvisSession::new();

    eprintln!("[JARVIS] ═══════════════════════════════════════════════");
    eprintln!("[JARVIS] GESTALT WIRED-V5 — Phase 3 ReAct Engine");
    eprintln!("[JARVIS] Config: {:?} | d_model={} | memories={}", tier, config.d_model, stored);
    eprintln!("[JARVIS] Tools: 15 built-in | ReAct: max 5 iterations");
    eprintln!("[JARVIS] Type 'quit' to exit, '/memory' to show memories");
    eprintln!("[JARVIS] ═══════════════════════════════════════════════");
    eprintln!();

    let stdin = io::stdin();
    print!("You: ");
    io::stdout().flush()?;

    for line in stdin.lock().lines() {
        let input = line?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            print!("You: ");
            io::stdout().flush()?;
            continue;
        }

        if trimmed == "quit" || trimmed == "/quit" {
            eprintln!("[JARVIS] Shutting down. Memories persisted to {}", mem_db_path);
            break;
        }

        if trimmed == "/memory" {
            eprintln!("[JARVIS] In-memory: {} entries | SQLite: {} entries",
                brain.memory_count(), episodic.len()?);
            eprintln!("[JARVIS] Session turns: {}/{}", session.buffer.len(), session.buffer.capacity());
            print!("\nYou: ");
            io::stdout().flush()?;
            continue;
        }

        if trimmed == "/history" {
            let ctx = session.buffer.context_string(10, 4096);
            if ctx.is_empty() {
                eprintln!("[JARVIS] No conversation history yet.");
            } else {
                eprintln!("{}", ctx);
            }
            print!("\nYou: ");
            io::stdout().flush()?;
            continue;
        }

        if trimmed == "/clear" {
            session.buffer.clear();
            eprintln!("[JARVIS] Session cleared.");
            print!("\nYou: ");
            io::stdout().flush()?;
            continue;
        }

        // Process through the full ReAct pipeline
        match session.process_message(
            trimmed, &brain, &executor, &decoder_tok, &device,
        ) {
            Ok(response) => {
                println!("\nJARVIS: {}", response);

                // Persist to episodic memory
                let enc_tok = TalkTokenizer;
                let goal_ids = enc_tok.encode(trimmed);
                let goal_padded = enc_tok.pad_or_truncate(&goal_ids, config.encoder_seq_len);
                if let Ok(goal_tensor) = Tensor::from_vec(
                    goal_padded, (1, config.encoder_seq_len), &device,
                ) {
                    if let Ok(cv) = brain.encode_concept(&goal_tensor) {
                        if let Ok(cv_vec) = cv.squeeze(0).and_then(|t| t.to_vec1::<f32>()) {
                            let _ = episodic.store(&cv_vec, trimmed, &response, true);
                            brain.store_memory(cv_vec, response.clone());
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("\n[ERROR] {}", e);
            }
        }

        print!("\nYou: ");
        io::stdout().flush()?;
    }
    Ok(())
}
