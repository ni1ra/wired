// Evaluation harness: plan_bench (21 goals) + brain_bench (64 goals)
// Port from V4 eval_adapter.rs (507 LOC) -- T-006

use crate::tokenizer::PlanTokenizer;

// ---------------------------------------------------------------------------
// Plan Bench — 21 goals (V3 compatible, the hard gate)
// ---------------------------------------------------------------------------

/// The 21 plan_bench goals from V3. Each is (goal_text, intent).
/// Plan-LM must generate correct plans for these to pass eval.
pub fn plan_bench_goals() -> Vec<(&'static str, PlanIntent)> {
    vec![
        ("hello", PlanIntent::Hello),
        ("verify the workspace", PlanIntent::RunTests),
        ("check the build", PlanIntent::CargoCheck),
        ("repair the test suite", PlanIntent::FixTests),
        ("sync docs", PlanIntent::DocsLint),
        ("lean suite", PlanIntent::LeanSuite),
        ("list files", PlanIntent::RepoList),
        ("search jarviscmd", PlanIntent::RepoSearch),
        ("find PolicyRunCtx", PlanIntent::RepoSearch),
        ("open the policy code", PlanIntent::RepoRead),
        ("search gpupolicy and then open it", PlanIntent::Composite),
        ("search runid and then open it", PlanIntent::Composite),
        ("search the gpu policy enforcement and then open the policy code and then docs lint", PlanIntent::Composite),
        ("find PolicyRunCtx and then open the jarvis implementation and then docs lint", PlanIntent::Composite),
        ("remember preference: favorite color is blue", PlanIntent::MemoryAdd),
        ("recall favorite color", PlanIntent::MemorySearch),
        ("remember preference: favorite color is blue and then recall favorite color", PlanIntent::Composite),
        ("prove x*(y+z) == x*y + x*z", PlanIntent::ProveAlgebra),
        ("patch docs/ALL_TASKS.md --dry-run", PlanIntent::PatchDryRun),
        ("remember procedure: sync docs", PlanIntent::MemoryAdd),
        ("remember fact: gpu policy", PlanIntent::MemoryAdd),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanIntent {
    Hello,
    RunTests,
    CargoCheck,
    FixTests,
    DocsLint,
    LeanSuite,
    RepoList,
    RepoSearch,
    RepoRead,
    Composite,
    MemoryAdd,
    MemorySearch,
    ProveAlgebra,
    PatchDryRun,
}

// ---------------------------------------------------------------------------
// Reference plan token sequences for each goal
// ---------------------------------------------------------------------------

/// Generate the reference (teacher) plan token sequence for a goal.
/// This is the "correct answer" that Plan-LM must learn to produce.
pub fn reference_plan_tokens(_tok: &PlanTokenizer, goal: &str, intent: PlanIntent) -> Vec<String> {
    let mut toks: Vec<String> = Vec::new();

    // Build step tokens based on intent
    let steps = steps_for_intent(goal, intent);
    let mut step_actions: Vec<String> = Vec::new();

    for step in &steps {
        toks.push("STEP".into());
        match step {
            PlanStep::Talk => {
                toks.push("TALK".into());
                step_actions.push("TALK".into());
            }
            PlanStep::CargoTest => {
                toks.push("CARGOTEST".into());
                step_actions.push("CARGOTEST".into());
            }
            PlanStep::FixTests => {
                toks.push("FIXTESTS".into());
                step_actions.push("FIXTESTS".into());
            }
            PlanStep::CargoCheck => {
                toks.push("CARGOCHECK".into());
                step_actions.push("CARGOCHECK".into());
            }
            PlanStep::DocsLint => {
                toks.push("DOCSLINT".into());
                step_actions.push("DOCSLINT".into());
            }
            PlanStep::LeanSuite => {
                toks.push("LEANSUITE".into());
                step_actions.push("LEANSUITE".into());
            }
            PlanStep::RepoList => {
                toks.push("REPOLIST".into());
                step_actions.push("REPOLIST".into());
            }
            PlanStep::Rg(pat) => {
                toks.push("RG".into());
                toks.push(format!("PAT{pat}"));
                step_actions.push("RG".into());
            }
            PlanStep::RepoRead(file) => {
                toks.push("REPOREAD".into());
                toks.push(format!("FILE{file}"));
                step_actions.push("REPOREAD".into());
            }
            PlanStep::RepoReadFromPick(from, pick) => {
                toks.push("REPOREAD".into());
                toks.push(format!("FROM{from}"));
                toks.push(format!("PICK{pick}"));
                step_actions.push("REPOREAD".into());
            }
            PlanStep::PatchDryRun(path_toks) => {
                toks.push("PATCHDRYRUN".into());
                toks.push("PATH".into());
                toks.extend(path_toks.clone());
                step_actions.push("PATCHDRYRUN".into());
            }
            PlanStep::ProveAlgebra(lhs_toks, rhs_toks) => {
                toks.push("PROVEALGEBRA".into());
                toks.push("LHS".into());
                toks.extend(lhs_toks.clone());
                toks.push("RHS".into());
                toks.extend(rhs_toks.clone());
                step_actions.push("PROVEALGEBRA".into());
            }
            PlanStep::MemAdd(kind, content_toks) => {
                toks.push("MEMADD".into());
                toks.push(kind.clone());
                toks.extend(content_toks.clone());
                step_actions.push("MEMADD".into());
            }
            PlanStep::MemSearch(query_toks) => {
                toks.push("MEMSEARCH".into());
                toks.extend(query_toks.clone());
                step_actions.push("MEMSEARCH".into());
            }
        }
    }

    // CoT prefix for multi-step plans
    if step_actions.len() >= 2 {
        let mut prefix = vec!["THINK".to_string(), "NEEDS".to_string()];
        for (i, a) in step_actions.iter().enumerate() {
            if i > 0 { prefix.push("THEN".into()); }
            prefix.push(a.clone());
        }
        prefix.push("ENDTHINK".into());
        prefix.append(&mut toks);
        toks = prefix;
    }

    toks.push("EOP".into());
    toks
}

/// Encode text as plan payload tokens (chars or lexicon words).
fn encode_payload(tok: &PlanTokenizer, s: &str) -> Vec<String> {
    let ids = tok.encode(s);
    ids.iter()
        .map(|&id| tok.token(id).to_string())
        .filter(|t| !matches!(t.as_str(), "\n" | "<PAD>" | "<BOS>" | "<UNK>"))
        .collect()
}

/// Wrap payload tokens with TEXTLEN prefix.
fn textlen_payload(tok: &PlanTokenizer, s: &str) -> Vec<String> {
    let payload = encode_payload(tok, s);
    let n = payload.len().min(32);
    let mut out = vec![format!("TEXTLEN{n}")];
    out.extend(payload.into_iter().take(n));
    out
}

#[derive(Clone, Debug)]
enum PlanStep {
    Talk,
    CargoTest,
    FixTests,
    CargoCheck,
    DocsLint,
    LeanSuite,
    RepoList,
    Rg(usize),
    RepoRead(usize),
    RepoReadFromPick(usize, usize),
    PatchDryRun(Vec<String>),
    ProveAlgebra(Vec<String>, Vec<String>),
    MemAdd(String, Vec<String>),
    MemSearch(Vec<String>),
}

fn steps_for_intent(goal: &str, intent: PlanIntent) -> Vec<PlanStep> {
    let tok = PlanTokenizer::new();
    let goal_lc = goal.to_lowercase();

    match intent {
        PlanIntent::Hello => vec![PlanStep::Talk],
        PlanIntent::RunTests => vec![PlanStep::CargoTest],
        PlanIntent::CargoCheck => vec![PlanStep::CargoCheck],
        PlanIntent::FixTests => vec![PlanStep::FixTests],
        PlanIntent::DocsLint => vec![PlanStep::DocsLint],
        PlanIntent::LeanSuite => vec![PlanStep::LeanSuite],
        PlanIntent::RepoList => vec![PlanStep::RepoList],
        PlanIntent::RepoSearch => {
            vec![PlanStep::Rg(0)]
        }
        PlanIntent::RepoRead => {
            vec![PlanStep::RepoRead(0)]
        }
        PlanIntent::MemoryAdd => {
            let kind = if goal_lc.contains("preference") {
                "KINDPREFERENCE"
            } else if goal_lc.contains("procedure") {
                "KINDPROCEDURE"
            } else {
                "KINDFACT"
            };
            let content = extract_mem_content(goal);
            let payload = textlen_payload(&tok, &content);
            vec![PlanStep::MemAdd(kind.into(), payload)]
        }
        PlanIntent::MemorySearch => {
            let query = extract_mem_query(goal);
            let payload = textlen_payload(&tok, &query);
            vec![PlanStep::MemSearch(payload)]
        }
        PlanIntent::ProveAlgebra => {
            if let Some((lhs, rhs)) = extract_algebra(goal) {
                let lhs_payload = textlen_payload(&tok, &lhs);
                let rhs_payload = textlen_payload(&tok, &rhs);
                vec![PlanStep::ProveAlgebra(lhs_payload, rhs_payload)]
            } else {
                vec![PlanStep::Talk]
            }
        }
        PlanIntent::PatchDryRun => {
            let path = extract_patch_path(goal);
            let path_toks = if path.contains("ALL_TASKS") {
                vec!["FILE1".into()]
            } else {
                textlen_payload(&tok, &path)
            };
            vec![PlanStep::PatchDryRun(path_toks)]
        }
        PlanIntent::Composite => {
            parse_composite_steps(goal)
        }
    }
}

fn parse_composite_steps(goal: &str) -> Vec<PlanStep> {
    let tok = PlanTokenizer::new();
    let parts: Vec<&str> = goal.split(" and then ")
        .flat_map(|p| p.split("; then "))
        .collect();

    let mut steps = Vec::new();
    let mut last_rg_step = None;

    for (i, part) in parts.iter().enumerate() {
        let part_lc = part.trim().to_lowercase();
        if part_lc.starts_with("search") || part_lc.starts_with("find") {
            steps.push(PlanStep::Rg(0));
            last_rg_step = Some(i);
        } else if part_lc.starts_with("open") {
            if let Some(rg_idx) = last_rg_step {
                steps.push(PlanStep::RepoReadFromPick(rg_idx, 0));
            } else {
                steps.push(PlanStep::RepoRead(0));
            }
        } else if part_lc.starts_with("docs lint") || part_lc.contains("docs lint") {
            steps.push(PlanStep::DocsLint);
        } else if part_lc.starts_with("remember") {
            let kind = if part_lc.contains("preference") {
                "KINDPREFERENCE"
            } else if part_lc.contains("procedure") {
                "KINDPROCEDURE"
            } else {
                "KINDFACT"
            };
            let content = extract_mem_content(part);
            let payload = textlen_payload(&tok, &content);
            steps.push(PlanStep::MemAdd(kind.into(), payload));
        } else if part_lc.starts_with("recall") || part_lc.starts_with("retrieve") {
            let query = extract_mem_query(part);
            let payload = textlen_payload(&tok, &query);
            steps.push(PlanStep::MemSearch(payload));
        } else {
            steps.push(PlanStep::Talk);
        }
    }
    steps
}

fn extract_mem_content(goal: &str) -> String {
    if let Some(pos) = goal.find(':') {
        let after = goal[pos + 1..].trim();
        let after_lc = after.to_lowercase();
        for prefix in ["preference", "procedure", "fact"] {
            if after_lc.starts_with(prefix) {
                let rest = &after[prefix.len()..];
                return rest.trim_start_matches([' ', ':', '-']).trim().to_string();
            }
        }
        return after.to_string();
    }
    let lc = goal.to_lowercase();
    if let Some(pos) = lc.find("remember") {
        return goal[pos + "remember".len()..].trim().to_string();
    }
    goal.to_string()
}

fn extract_mem_query(goal: &str) -> String {
    let lc = goal.to_lowercase();
    for prefix in ["recall", "retrieve", "search memory", "find memory", "get"] {
        if let Some(pos) = lc.find(prefix) {
            return goal[pos + prefix.len()..].trim().to_string();
        }
    }
    goal.to_string()
}

fn extract_algebra(goal: &str) -> Option<(String, String)> {
    let lc = goal.to_lowercase();
    let frag = if let Some(pos) = lc.find("prove") {
        &goal[pos + "prove".len()..]
    } else {
        goal
    };
    let (lhs, rhs) = frag.split_once("==")?;
    let canon = |s: &str| -> String {
        s.chars().filter(|c| !c.is_ascii_whitespace()).collect()
    };
    let lhs = canon(lhs.trim());
    let rhs = canon(rhs.trim());
    if lhs.is_empty() || rhs.is_empty() { return None; }
    Some((lhs, rhs))
}

fn extract_patch_path(goal: &str) -> String {
    let lc = goal.to_lowercase();
    let frag = if let Some(pos) = lc.find("patch") {
        &goal[pos + "patch".len()..]
    } else {
        goal
    };
    frag.split_whitespace()
        .next()
        .unwrap_or("")
        .to_string()
}

// ---------------------------------------------------------------------------
// Plan Bench Scoring
// ---------------------------------------------------------------------------

/// Score a generated plan against the reference.
/// Returns (correct, total) count.
pub fn score_plan_bench(
    tok: &PlanTokenizer,
    generate_fn: &dyn Fn(&str) -> Vec<String>,
) -> (usize, usize) {
    let goals = plan_bench_goals();
    let total = goals.len();
    let mut correct = 0;

    for (goal, intent) in &goals {
        let reference = reference_plan_tokens(tok, goal, *intent);
        let generated = generate_fn(goal);

        let ref_stripped = strip_cot(&reference);
        let gen_stripped = strip_cot(&generated);

        if ref_stripped == gen_stripped {
            correct += 1;
        }
    }

    (correct, total)
}

fn strip_cot(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_cot = false;
    for t in tokens {
        if t == "THINK" { in_cot = true; continue; }
        if t == "ENDTHINK" { in_cot = false; continue; }
        if !in_cot {
            out.push(t.clone());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Brain Bench — 64 goals (stub, full port in Phase 1)
// ---------------------------------------------------------------------------

pub fn brain_bench_goal_count() -> usize {
    64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_bench_has_21_goals() {
        assert_eq!(plan_bench_goals().len(), 21);
    }

    #[test]
    fn test_reference_plan_hello() {
        let tok = PlanTokenizer::new();
        let plan = reference_plan_tokens(&tok, "hello", PlanIntent::Hello);
        assert_eq!(plan, vec!["STEP", "TALK", "EOP"]);
    }

    #[test]
    fn test_reference_plan_cargo_test() {
        let tok = PlanTokenizer::new();
        let plan = reference_plan_tokens(&tok, "verify the workspace", PlanIntent::RunTests);
        assert_eq!(plan, vec!["STEP", "CARGOTEST", "EOP"]);
    }

    #[test]
    fn test_reference_plan_composite_has_cot() {
        let tok = PlanTokenizer::new();
        let plan = reference_plan_tokens(&tok, "search gpupolicy and then open it", PlanIntent::Composite);
        assert!(plan.contains(&"THINK".to_string()), "composite should have CoT: {:?}", plan);
        assert!(plan.contains(&"ENDTHINK".to_string()));
        assert!(plan.contains(&"EOP".to_string()));
    }

    #[test]
    fn test_reference_plan_memory_add() {
        let tok = PlanTokenizer::new();
        let plan = reference_plan_tokens(&tok, "remember preference: favorite color is blue", PlanIntent::MemoryAdd);
        assert!(plan.contains(&"STEP".to_string()));
        assert!(plan.contains(&"MEMADD".to_string()));
        assert!(plan.contains(&"KINDPREFERENCE".to_string()));
        assert!(plan.contains(&"EOP".to_string()));
    }

    #[test]
    fn test_reference_plan_algebra() {
        let tok = PlanTokenizer::new();
        let plan = reference_plan_tokens(&tok, "prove x*(y+z) == x*y + x*z", PlanIntent::ProveAlgebra);
        assert!(plan.contains(&"PROVEALGEBRA".to_string()));
        assert!(plan.contains(&"LHS".to_string()));
        assert!(plan.contains(&"RHS".to_string()));
    }

    #[test]
    fn test_strip_cot() {
        let tokens: Vec<String> = vec!["THINK", "NEEDS", "RG", "THEN", "REPOREAD", "ENDTHINK",
            "STEP", "RG", "PAT0", "STEP", "REPOREAD", "FROM0", "PICK0", "EOP"]
            .into_iter().map(String::from).collect();
        let stripped = strip_cot(&tokens);
        assert!(!stripped.contains(&"THINK".to_string()));
        assert!(!stripped.contains(&"ENDTHINK".to_string()));
        assert!(stripped.contains(&"STEP".to_string()));
    }

    #[test]
    fn test_extract_algebra() {
        let (lhs, rhs) = extract_algebra("prove x*(y+z) == x*y + x*z").unwrap();
        assert_eq!(lhs, "x*(y+z)");
        assert_eq!(rhs, "x*y+x*z");
    }

    #[test]
    fn test_extract_mem_content() {
        let content = extract_mem_content("remember preference: favorite color is blue");
        assert_eq!(content, "favorite color is blue");
    }

    #[test]
    fn test_score_plan_bench_with_perfect_oracle() {
        let tok = PlanTokenizer::new();
        let (correct, total) = score_plan_bench(&tok, &|goal| {
            let goals = plan_bench_goals();
            let intent = goals.iter().find(|(g, _)| *g == goal).map(|(_, i)| *i).unwrap();
            reference_plan_tokens(&tok, goal, intent)
        });
        assert_eq!(correct, total, "perfect oracle should get 21/21, got {correct}/{total}");
    }
}
