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
    Brain, BrainConfig, PolicyConfig,
    train_brain_talk, brain_generate, train_and_bench,
    bootstrap_concept_tokenizer,
};
use gestalt::eval::score_plan_bench;
use gestalt::pipeline::{run_goal, PipelineConfig};
use gestalt::planner::{PlanLmConfig, train_sft, greedy_decode};
use gestalt::tokenizer::PlanTokenizer;
use gestalt::training::{save_checkpoint, load_checkpoint};

use candle_core::Device;
use candle_nn::VarMap;
use std::io::{self, BufRead, Write};

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
        "serve" => cmd_serve(tier),
        "gallery" => cmd_gallery(tier),
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
    eprintln!("  serve        Persistent stdin/stdout interface");
    eprintln!("  gallery      Run comprehensive generation gallery (requires checkpoint)");
    eprintln!();
    eprintln!("Config tiers:");
    eprintln!("  test     d=64, 1-2 layers, CPU only (fast, for tests)");
    eprintln!("  default  d=512, 4 layers, auto-GPU (~105M params)");
    eprintln!("  phase2   d=1024, 8 layers, auto-GPU (~200M params)");
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

    let config = tier.brain_config();
    let policy_cfg = tier.policy_config();
    let varmap = VarMap::new();
    let brain = Brain::new(config, &policy_cfg, &varmap, &device)?;

    // Load checkpoint if available
    let ckpt = "brain_checkpoint.safetensors";
    if std::path::Path::new(ckpt).exists() {
        load_checkpoint(&varmap, ckpt, &device)?;
    } else {
        eprintln!("[GESTALT] No checkpoint found, using untrained brain");
    }

    let pipeline_config = PipelineConfig::default_config(std::env::current_dir()?);
    let result = run_goal(&brain, &goal, &pipeline_config, &device)?;

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
    let config = tier.brain_config();
    let policy_cfg = tier.policy_config();
    let plan_cfg = tier.plan_config();

    eprintln!("[GESTALT] Config: {:?} | d_model={} | enc_layers={} | dec_layers={}",
        tier, config.d_model, config.encoder_layers, config.decoder_layers);
    eprintln!("[GESTALT] Brain: {} SFT + {} DA steps", config.sft_steps, config.da_steps);
    eprintln!("[GESTALT] Planner: {} SFT + {} SS steps (d={})",
        plan_cfg.sft_steps, plan_cfg.scheduled_sampling_steps, plan_cfg.d_model);
    eprintln!("[GESTALT] Policy: {} steps (d={})", policy_cfg.steps, policy_cfg.d_model);

    let temperature = config.temperature;
    let (brain, _brain_varmap) = if resume {
        // Resume: load brain from best SFT checkpoint, skip training
        let ckpt = "brain_best_sft.safetensors";
        eprintln!("\n[GESTALT] === Resuming from {} ===", ckpt);
        anyhow::ensure!(std::path::Path::new(ckpt).exists(),
            "Cannot resume: {} not found", ckpt);
        let varmap = VarMap::new();
        let brain = Brain::new(config, &policy_cfg, &varmap, &device)?;
        load_checkpoint(&varmap, ckpt, &device)?;
        eprintln!("[GESTALT] Brain loaded from checkpoint");
        (brain, varmap)
    } else {
        // Phase 1: Brain (SFT + dialogue-aligned finetuning)
        eprintln!("\n[GESTALT] === Training brain (SFT + DA) ===");
        let (brain, brain_varmap, losses) = train_brain_talk(&config, &policy_cfg, &device)?;
        let final_loss = losses.last().copied().unwrap_or(f32::NAN);
        eprintln!("[GESTALT] Brain done. Final loss: {:.4}", final_loss);

        // Save brain checkpoint
        if tier != ConfigTier::Test {
            save_checkpoint(&brain_varmap, "brain_checkpoint.safetensors")?;
        }
        (brain, brain_varmap)
    };

    // Concept tokenizer bootstrap (T-014) — uses trained encoder
    if tier != ConfigTier::Test {
        eprintln!("\n[GESTALT] === Bootstrapping concept tokenizer (T-014) ===");
        let max_merges = if tier == ConfigTier::Phase2 { 8000 } else { 2000 };
        let concept_tok = bootstrap_concept_tokenizer(&brain, max_merges, 3, &device)?;

        // Save tokenizer
        let tok_path = std::path::Path::new("concept_tokenizer.bin");
        let tok_data = concept_tok.save_merges();
        std::fs::write(tok_path, &tok_data)?;
        eprintln!("[GESTALT] Saved concept tokenizer to {:?} ({} bytes)",
            tok_path, tok_data.len());

        // Show compression stats on a few sample inputs
        for sample in &["hello", "search jarviscmd", "what can you do"] {
            let ratio = concept_tok.compression_ratio(sample);
            let ids = concept_tok.encode(sample);
            eprintln!("[GESTALT] \"{}\" -> {} tokens (compression {:.2}x)",
                sample, ids.len() - 2, ratio); // -2 for BOS/EOS
        }
    }

    // Planner (SFT)
    eprintln!("\n[GESTALT] === Training planner (SFT) ===");
    let (_planner, plan_varmap, plan_losses) = train_sft(&plan_cfg, &device)?;
    let plan_final = plan_losses.last().copied().unwrap_or(f32::NAN);
    eprintln!("[GESTALT] Planner done. Final loss: {:.4}", plan_final);

    // Save planner checkpoint
    if tier != ConfigTier::Test {
        save_checkpoint(&plan_varmap, "planner_checkpoint.safetensors")?;
    }

    // Phase 3: Policy benchmark
    eprintln!("\n[GESTALT] === Training + benchmarking policy ===");
    let (correct, total) = train_and_bench(&policy_cfg, &device)?;
    eprintln!("[GESTALT] Policy: {}/{}", correct, total);

    // Generation test: both greedy (temp=0) and sampled
    eprintln!("\n[GESTALT] === Generation test ===");
    let max_gen = if tier == ConfigTier::Test { 64 } else { 128 };

    // Greedy first (true model quality indicator)
    let r_greedy = brain_generate(&brain, "hello", max_gen, 0.0, false, &device)?;
    eprintln!("[GESTALT] \"hello\" (greedy) -> \"{}\"", r_greedy);

    let r_greedy2 = brain_generate(&brain, "what can you do", max_gen, 0.0, false, &device)?;
    eprintln!("[GESTALT] \"what can you do\" (greedy) -> \"{}\"", r_greedy2);

    // Sampled
    let temp = if tier == ConfigTier::Test { 0.8 } else { temperature };
    let response = brain_generate(&brain, "hello", max_gen, temp, false, &device)?;
    eprintln!("[GESTALT] \"hello\" (temp={:.1}) -> \"{}\"", temp, response);

    let response2 = brain_generate(&brain, "what can you do", max_gen, temp, false, &device)?;
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
                let gen = brain_generate(&brain, prompt, max_gen, 0.0, false, &device)?;
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
    let varmap = VarMap::new();
    let brain = Brain::new(config, &policy_cfg, &varmap, &device)?;

    let ckpt = "brain_checkpoint.safetensors";
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
            let gen = brain_generate(&brain, prompt, max_gen, 0.0, false, &device)?;
            eprintln!("[GALLERY] \"{}\" -> \"{}\"", prompt, gen);
            total += 1;
        }
    }

    // Sampled generation with temperature for diversity
    eprintln!("\n[GALLERY] --- Sampled (temp=0.5) ---");
    for prompt in &["hello", "What is beauty?", "Tell me a joke", "What is consciousness?",
                     "I'm tired", "What is the meaning of life?"] {
        let gen = brain_generate(&brain, prompt, max_gen, 0.5, false, &device)?;
        eprintln!("[GALLERY] \"{}\" (t=0.5) -> \"{}\"", prompt, gen);
        total += 1;
    }

    eprintln!("\n[GALLERY] --- Sampled (temp=0.7) ---");
    for prompt in &["hello", "tell me a joke", "What is beauty?", "convince me to learn Rust",
                     "Are you alive?", "Say something beautiful", "Roast me"] {
        let gen = brain_generate(&brain, prompt, max_gen, 0.7, false, &device)?;
        eprintln!("[GALLERY] \"{}\" (t=0.7) -> \"{}\"", prompt, gen);
        total += 1;
    }

    eprintln!("\n[GALLERY] --- Sampled (temp=1.0) ---");
    for prompt in &["hello", "tell me a joke", "What is truth?", "Do you dream?"] {
        let gen = brain_generate(&brain, prompt, max_gen, 1.0, false, &device)?;
        eprintln!("[GALLERY] \"{}\" (t=1.0) -> \"{}\"", prompt, gen);
        total += 1;
    }

    eprintln!("\n[GALLERY] === {} generations complete ===", total);
    Ok(())
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
    let config = tier.brain_config();
    let policy_cfg = tier.policy_config();
    let varmap = VarMap::new();
    let brain = Brain::new(config, &policy_cfg, &varmap, &device)?;

    // Load checkpoint if available
    let ckpt = "brain_checkpoint.safetensors";
    if std::path::Path::new(ckpt).exists() {
        load_checkpoint(&varmap, ckpt, &device)?;
    } else {
        eprintln!("[GESTALT] No checkpoint found, using untrained brain");
    }

    let pipeline_config = PipelineConfig::default_config(std::env::current_dir()?);

    eprintln!("[GESTALT] Ready ({:?} config). Enter goals (one per line, 'quit' to exit):", tier);
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let goal = line?;
        let trimmed = goal.trim();
        if trimmed.is_empty() || trimmed == "quit" {
            break;
        }

        match run_goal(&brain, trimmed, &pipeline_config, &device) {
            Ok(result) => {
                println!("{}", result.final_output);
                io::stdout().flush()?;
            }
            Err(e) => eprintln!("[ERROR] {}", e),
        }
    }
    Ok(())
}
