#!/usr/bin/env bash
# Scaffold-PGSR 7k 训练入口：按 7k/30k 的相对训练进度压缩 PGSR loss、densification 与 LR schedule。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ===== 7k 总预算 =====
export ITERATIONS="${ITERATIONS:-7000}"
export RUN_NAME="${RUN_NAME:-pipeline-7k}"
export SAVE_ITERATIONS="${SAVE_ITERATIONS:-7000}"
export TEST_ITERATIONS="${TEST_ITERATIONS:-7000}"

# ===== PGSR regularization schedule =====
# 30k: single-view/multi-view 都从 3000 (10%) 开始。
# 7k : 对应 700 (10%)。
export START_SINGLE_VIEW="${START_SINGLE_VIEW:-700}"
export START_MULTI_VIEW="${START_MULTI_VIEW:-700}"

# ===== Scaffold structural adaptation schedule =====
# 30k: start_stat=500, densify_from=1500, interval=100, densify_until=15000
# 7k : approximately preserve the same normalized training progress.
export START_STAT="${START_STAT:-120}"
export DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-350}"
export DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-3500}"
export DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-25}"

# Complete the original 30k exponential LR decay within the 7k budget.
export LR_MAX_STEPS="${LR_MAX_STEPS:-7000}"

exec bash "$SCRIPT_DIR/train_scaffold-pgsr.sh" "$@"
