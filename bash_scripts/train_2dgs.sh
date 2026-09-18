#!/usr/bin/env bash
# 在任意目录运行；先 conda activate 3dgs。支持环境变量覆盖下列参数。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# ===== 常用参数：直接修改默认值，或通过环境变量覆盖 =====
SCENE="${SCENE:-xiaoxiang_03}"
VIEWS="${VIEWS:-003}"
SOURCE_PATH="${SOURCE_PATH:-$WORKSPACE/dataset/scenes/$SCENE/$VIEWS}"
OUTPUT_PATH="${OUTPUT_PATH:-$WORKSPACE/output}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SCENE}_${VIEWS}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
ITERATIONS="${ITERATIONS:-30000}"
RESOLUTION="${RESOLUTION:-1}"          # 1 原图，2/4/8 降采样
SEED="${SEED:-42}"
EVAL="${EVAL:-False}"                  # 少视角默认全部用于训练
SAVE_ITERATIONS="${SAVE_ITERATIONS:-7000 15000}" # 自动追加最后一步
TEST_ITERATIONS="${TEST_ITERATIONS:-7000 15000}" # 自动追加最后一步
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-}" # 可恢复训练的 checkpoint；空=不额外保存

# ===== 2DGS 参数（初值沿用仓库默认值） =====
LAMBDA_NORMAL="${LAMBDA_NORMAL:-0.05}"
LAMBDA_DIST="${LAMBDA_DIST:-0.0}"
DEPTH_RATIO="${DEPTH_RATIO:-0.0}"
START_NORMAL_LOSS="${START_NORMAL_LOSS:-7000}"
START_DIST_LOSS="${START_DIST_LOSS:-3000}"

METHOD="2dgs"
# ===== 深度缓存与 mesh：同一 RUN_NAME 重跑会复用 depth =====
RUN_NAME="${RUN_NAME:-pipeline}"
MESH_RES="${MESH_RES:-1024}"
MESH_VOXEL_SIZE="${MESH_VOXEL_SIZE:--1}" # -1 自动；单位同模型
DEPTH_TRUNC="${DEPTH_TRUNC:--1}"
SDF_TRUNC="${SDF_TRUNC:--1}"
NUM_CLUSTER="${NUM_CLUSTER:-50}"
METHOD_ARGS=()
METHOD_ARGS+=(--scene.lambda-normal "$LAMBDA_NORMAL" --scene.lambda-dist "$LAMBDA_DIST"
              --scene.depth-ratio "$DEPTH_RATIO" --scene.start-normal-loss-iter "$START_NORMAL_LOSS"
              --scene.satrt-dist-loss-iter "$START_DIST_LOSS") # satrt 是仓库原参数拼写
source "$SCRIPT_DIR/_train_common.sh"
run_training "$@"
