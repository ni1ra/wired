#!/bin/bash
# safe-run.sh — Resource-monitored cargo wrapper for WSL2
# Prevents OOM crashes by monitoring RAM/CPU/GPU every 5s
# Usage: ./safe-run.sh <cargo command args>
# Example: ./safe-run.sh test --lib tokenizer
#          ./safe-run.sh build --release
#          ./safe-run.sh test --lib brain -- test_name

set -euo pipefail

# === THRESHOLDS ===
RAM_KILL_PCT=85        # Kill cargo if RAM usage exceeds this %
CPU_WARN_PCT=90        # Warn if CPU load average exceeds this % of cores
SWAP_KILL_MB=4096      # Kill if swap usage exceeds this (MB)
MONITOR_INTERVAL=5     # Check every N seconds
MAX_CARGO_JOBS=4       # Limit parallel rustc processes (8 cores, use half)

# === COLORS ===
RED='\033[0;31m'
YEL='\033[1;33m'
GRN='\033[0;32m'
NC='\033[0m'

if [ $# -eq 0 ]; then
    echo -e "${RED}Usage: ./safe-run.sh <cargo subcommand> [args...]${NC}"
    echo "Example: ./safe-run.sh test --lib tokenizer"
    exit 1
fi

# === PRE-FLIGHT CHECK ===
echo -e "${GRN}=== SAFE-RUN PRE-FLIGHT ===${NC}"

TOTAL_RAM=$(free -m | awk '/Mem:/{print $2}')
USED_RAM=$(free -m | awk '/Mem:/{print $3}')
AVAIL_RAM=$(free -m | awk '/Mem:/{print $7}')
SWAP_USED=$(free -m | awk '/Swap:/{print $3}')
RAM_PCT=$((USED_RAM * 100 / TOTAL_RAM))

echo "RAM: ${USED_RAM}MB / ${TOTAL_RAM}MB (${RAM_PCT}% used, ${AVAIL_RAM}MB available)"
echo "Swap: ${SWAP_USED}MB used"

# Check GPU if available
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "N/A")
    echo "GPU: ${GPU_INFO}"
fi

LOAD=$(cat /proc/loadavg | awk '{print $1}')
CORES=$(nproc)
echo "CPU: load ${LOAD} / ${CORES} cores"

# Abort if already stressed
if [ "$RAM_PCT" -ge "$RAM_KILL_PCT" ]; then
    echo -e "${RED}ABORT: RAM already at ${RAM_PCT}% — too risky to start cargo${NC}"
    exit 1
fi

if [ "$SWAP_USED" -ge "$SWAP_KILL_MB" ]; then
    echo -e "${RED}ABORT: Swap already at ${SWAP_USED}MB — system under memory pressure${NC}"
    exit 1
fi

echo -e "${GRN}Pre-flight OK. Starting cargo with ${MAX_CARGO_JOBS} parallel jobs...${NC}"
echo "Monitoring every ${MONITOR_INTERVAL}s. Kill thresholds: RAM>${RAM_KILL_PCT}%, Swap>${SWAP_KILL_MB}MB"
echo "---"

# === RUN CARGO WITH MONITORING ===
export CARGO_BUILD_JOBS=$MAX_CARGO_JOBS

# Start cargo in background
cargo "$@" 2>&1 &
CARGO_PID=$!

# Monitor loop
KILLED=0
while kill -0 $CARGO_PID 2>/dev/null; do
    sleep $MONITOR_INTERVAL

    # Re-check if still running after sleep
    if ! kill -0 $CARGO_PID 2>/dev/null; then
        break
    fi

    USED_RAM=$(free -m | awk '/Mem:/{print $3}')
    TOTAL_RAM=$(free -m | awk '/Mem:/{print $2}')
    AVAIL_RAM=$(free -m | awk '/Mem:/{print $7}')
    SWAP_USED=$(free -m | awk '/Swap:/{print $3}')
    RAM_PCT=$((USED_RAM * 100 / TOTAL_RAM))
    LOAD=$(cat /proc/loadavg | awk '{print $1}')

    # GPU check
    GPU_STR=""
    if command -v nvidia-smi &>/dev/null; then
        GPU_STR=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "GPU:%sMB/%s%%", $1, $2}' || echo "")
    fi

    TIMESTAMP=$(date +%H:%M:%S)
    echo -e "[${TIMESTAMP}] RAM:${RAM_PCT}% (${AVAIL_RAM}MB free) Swap:${SWAP_USED}MB CPU:${LOAD} ${GPU_STR}"

    # Kill conditions
    if [ "$RAM_PCT" -ge "$RAM_KILL_PCT" ]; then
        echo -e "${RED}[${TIMESTAMP}] KILLING CARGO — RAM at ${RAM_PCT}% exceeds ${RAM_KILL_PCT}% threshold${NC}"
        kill -TERM $CARGO_PID 2>/dev/null
        sleep 2
        kill -9 $CARGO_PID 2>/dev/null
        KILLED=1
        break
    fi

    if [ "$SWAP_USED" -ge "$SWAP_KILL_MB" ]; then
        echo -e "${RED}[${TIMESTAMP}] KILLING CARGO — Swap at ${SWAP_USED}MB exceeds ${SWAP_KILL_MB}MB threshold${NC}"
        kill -TERM $CARGO_PID 2>/dev/null
        sleep 2
        kill -9 $CARGO_PID 2>/dev/null
        KILLED=1
        break
    fi

    # CPU warning (non-fatal)
    LOAD_INT=$(echo "$LOAD" | awk '{printf "%d", $1 * 100 / '"$CORES"'}')
    if [ "$LOAD_INT" -ge "$CPU_WARN_PCT" ]; then
        echo -e "${YEL}[${TIMESTAMP}] WARNING: CPU load at ${LOAD} (${LOAD_INT}% of ${CORES} cores)${NC}"
    fi
done

# Wait for cargo to finish and get exit code
if [ "$KILLED" -eq 1 ]; then
    echo -e "${RED}=== SAFE-RUN: Cargo killed to protect WSL ===${NC}"
    echo "Try reducing scope: test specific modules, not the whole suite"
    exit 137
else
    wait $CARGO_PID
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo -e "${GRN}=== SAFE-RUN: Cargo completed successfully ===${NC}"
    else
        echo -e "${RED}=== SAFE-RUN: Cargo exited with code ${EXIT_CODE} ===${NC}"
    fi
    exit $EXIT_CODE
fi
