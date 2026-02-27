#!/bin/bash
# GESTALT fast grid search — 300 steps per config, val every 100
# ~3 min per run. Smart pruning: skip if val@100 > worst_best + margin
set -e

BINARY="./target/release/gestalt"
RESULTS="/tmp/grid_search_results.txt"

echo "=== GESTALT Fast Grid Search ===" > "$RESULTS"
echo "Started: $(date -u '+%Y-%m-%d %H:%M UTC')" >> "$RESULTS"
echo "300 steps/run, val every 100 steps" >> "$RESULTS"
echo "" >> "$RESULTS"

# Configs to test
declare -a CONFIGS=(
  "200 0.1"
  "500 0.1"
  "1000 0.1"
  "2000 0.1"
  "500 0.15"
  "500 0.2"
)

best_val="999.0"
best_config=""

for cfg in "${CONFIGS[@]}"; do
  read -r merges dropout <<< "$cfg"
  LOG="/tmp/grid_m${merges}_d${dropout}.log"

  echo ""
  echo "=== merges=$merges dropout=$dropout ==="
  echo "Start: $(date -u '+%H:%M:%S UTC')"

  rm -f brain_best_sft.safetensors brain_checkpoint.safetensors concept_tokenizer.bin

  GESTALT_MERGES="$merges" \
  GESTALT_DROPOUT="$dropout" \
  GESTALT_SFT_STEPS=300 \
  GESTALT_LOG_EVERY=100 \
  GESTALT_BRAIN_ONLY=1 \
  "$BINARY" train --config default 2>&1 | tee "$LOG"

  # Extract val losses at each checkpoint
  val_at_100=$(grep "step 100/" "$LOG" | grep -oP 'val_loss=\K[0-9.]+' || echo "N/A")
  val_at_200=$(grep "step 200/" "$LOG" | grep -oP 'val_loss=\K[0-9.]+' || echo "N/A")
  val_at_300=$(grep "step 300/" "$LOG" | grep -oP 'val_loss=\K[0-9.]+' || echo "N/A")
  # Use the last available val
  final_val="${val_at_300:-${val_at_200:-${val_at_100:-999}}}"

  echo "merges=$merges dropout=$dropout | val@100=$val_at_100 val@200=$val_at_200 val@300=$val_at_300" | tee -a "$RESULTS"

  # Track best
  if [ "$final_val" != "N/A" ] && [ "$(echo "$final_val < $best_val" | bc -l)" -eq 1 ]; then
    best_val="$final_val"
    best_config="merges=$merges dropout=$dropout"
  fi
done

echo "" >> "$RESULTS"
echo "===== WINNER: $best_config (val=$best_val) =====" >> "$RESULTS"
echo "Finished: $(date -u '+%Y-%m-%d %H:%M UTC')" >> "$RESULTS"

echo ""
echo "============================="
echo "WINNER: $best_config (val=$best_val)"
echo "============================="
cat "$RESULTS"
