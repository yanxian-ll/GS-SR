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

# ===== PGSR 参数（初值沿用仓库默认损失权重） =====
USE_MVS_VIEW_SELECTION="${USE_MVS_VIEW_SELECTION:-True}"
NUM_MULTI_VIEW="${NUM_MULTI_VIEW:-auto}" # auto=min(5,注册图像数-1)
LAMBDA_NORMAL="${LAMBDA_NORMAL:-0.015}"
LAMBDA_NCC="${LAMBDA_NCC:-0.15}"
LAMBDA_GEO="${LAMBDA_GEO:-0.03}"
START_SINGLE_VIEW="${START_SINGLE_VIEW:-3000}"
START_MULTI_VIEW="${START_MULTI_VIEW:-3000}"

METHOD="scaffold-pgsr"
# ===== 深度缓存与 mesh：同一 RUN_NAME 重跑会复用 depth =====
RUN_NAME="${RUN_NAME:-pipeline}"
MESH_RES="${MESH_RES:-1024}"
MESH_VOXEL_SIZE="${MESH_VOXEL_SIZE:--1}" # -1 自动；单位同模型
DEPTH_TRUNC="${DEPTH_TRUNC:--1}"
SDF_TRUNC="${SDF_TRUNC:--1}"
NUM_CLUSTER="${NUM_CLUSTER:-50}"
METHOD_ARGS=()
# ===== Scaffold 参数 =====
VOXEL_SIZE="${VOXEL_SIZE:-0.0}"          # <=0 根据点间距自动确定
N_OFFSETS="${N_OFFSETS:-10}"
APPEARANCE_DIM="${APPEARANCE_DIM:-32}"
START_STAT="${START_STAT:-500}"
DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-1500}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-15000}"
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-100}"
LR_MAX_STEPS="${LR_MAX_STEPS:-30000}"    # Scaffold offset/MLP/appearance LR 的衰减终点
METHOD_ARGS+=(--scene.gaussians.voxel-size "$VOXEL_SIZE"
              --scene.gaussians.n-offsets "$N_OFFSETS"
              --scene.gaussians.appearance-dim "$APPEARANCE_DIM"
              --scene.gaussians.start-stat "$START_STAT"
              --scene.gaussians.densify-from-iter "$DENSIFY_FROM_ITER"
              --scene.gaussians.densify-until-iter "$DENSIFY_UNTIL_ITER"
              --scene.gaussians.densification-interval "$DENSIFICATION_INTERVAL"
              --scene.gaussians.position-lr-max-steps "$LR_MAX_STEPS"
              --scene.gaussians.offset-lr-max-steps "$LR_MAX_STEPS"
              --scene.gaussians.mlp-opacity-lr-max-steps "$LR_MAX_STEPS"
              --scene.gaussians.mlp-cov-lr-max-steps "$LR_MAX_STEPS"
              --scene.gaussians.mlp-color-lr-max-steps "$LR_MAX_STEPS"
              --scene.gaussians.mlp-featurebank-lr-max-steps "$LR_MAX_STEPS"
              --scene.gaussians.appearance-lr-max-steps "$LR_MAX_STEPS")
METHOD_ARGS+=(--scene.dataloader.use-mvs-view-selection "$USE_MVS_VIEW_SELECTION"
              --scene.lambda-normal "$LAMBDA_NORMAL" --scene.lambda-ncc "$LAMBDA_NCC"
              --scene.lambda-geo "$LAMBDA_GEO"
              --scene.start-single-view-loss-iter "$START_SINGLE_VIEW"
              --scene.start-multi-view-loss-iter "$START_MULTI_VIEW")
source "$SCRIPT_DIR/_train_common.sh"
run_training "$@"
