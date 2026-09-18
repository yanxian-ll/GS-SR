#!/usr/bin/env bash
# Scaffold-PGSR 7k 训练入口。其余参数完全继承 train_scaffold-pgsr.sh。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ITERATIONS="${ITERATIONS:-7000}"
export RUN_NAME="${RUN_NAME:-pipeline-7k}"

exec "$SCRIPT_DIR/train_scaffold-pgsr.sh" "$@"
