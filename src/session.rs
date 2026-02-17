// Multi-turn session state + ReAct loop -- T-022, T-023 (Phase 4, new)
//
// Ring buffer of turns (max 32)
// ReAct: Reason -> Act -> Observe -> Reason (max 10 iterations)
