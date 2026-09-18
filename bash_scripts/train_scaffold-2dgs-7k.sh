#!/usr/bin/env bash
# Scaffold-2DGS 7k 训练入口：面向 UAV/户外表面重建，压缩 schedule 并启用 2DGS distortion regularization。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ===== 7k 总预算 =====
export ITERATIONS="${ITERATIONS:-7000}"
export RUN_NAME="${RUN_NAME:-pipeline-7k}"
export SAVE_ITERATIONS="${SAVE_ITERATIONS:-7000}"
export TEST_ITERATIONS="${TEST_ITERATIONS:-7000}"

# ===== 2DGS geometry regularization =====
# 通用 30k 脚本保留仓库默认 LAMBDA_DIST=0；7k surface profile 默认启用 distortion。
# 仓库 TNT outdoor/360 评测使用 100，大场景使用 10；UAV 默认先取 100，可通过环境变量覆盖。
export LAMBDA_DIST="${LAMBDA_DIST:-100}"
# 30k: dist 3000 (10%), normal 7000 (23.3%)
# 7k : dist  700 (10%), normal 1600 (~22.9%)
export START_DIST_LOSS="${START_DIST_LOSS:-700}"
export START_NORMAL_LOSS="${START_NORMAL_LOSS:-1600}"

# ===== Scaffold structural adaptation schedule =====
# 30k: start_stat=500, densify_from=1500, interval=100, densify_until=15000
# 7k : approximately preserve the same normalized training progress.
export START_STAT="${START_STAT:-120}"
export DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-350}"
export DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-3500}"
export DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-25}"

# Complete the original 30k exponential LR decay within the 7k budget.
export LR_MAX_STEPS="${LR_MAX_STEPS:-7000}"

exec bash "$SCRIPT_DIR/train_scaffold-2dgs.sh" "$@"
