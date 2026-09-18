#!/usr/bin/env bash
# 四个方法共用的路径检查与启动逻辑，请从 train_*.sh 启动。
make_schedule() {
    local values="$1" step
    for step in $values "$ITERATIONS"; do
        [[ "$step" =~ ^[1-9][0-9]*$ ]] || { echo "无效迭代步: $step" >&2; return 1; }
        if ((step <= ITERATIONS)); then printf '%s\n' "$step"; fi
    done
}
run_training() {
    local dry_run=0
    if [[ "${1:-}" == '--dry-run' ]]; then dry_run=1; shift; fi
    command -v "$PYTHON_BIN" >/dev/null || { echo "找不到 $PYTHON_BIN，请激活 3dgs 环境。" >&2; return 1; }
    [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]] || { echo 'ITERATIONS 必须为正整数' >&2; return 1; }
    # 在切换到 GS-SR 前解析相对路径；空格路径通过数组安全传参。
    SOURCE_PATH="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$SOURCE_PATH")"
    OUTPUT_PATH="$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_PATH")"
    [[ "$RUN_NAME" != */* && "$RUN_NAME" != . && "$RUN_NAME" != .. && -n "$RUN_NAME" ]] || { echo 'RUN_NAME 必须为单个目录名' >&2; return 1; }
    local run_dir="$OUTPUT_PATH/$EXPERIMENT_NAME/$METHOD/$RUN_NAME"
    local -a mesh_command=("$PYTHON_BIN" "$WORKSPACE/GS-SR/script/depth_mesh.py"
        --run-dir "$run_dir" --iteration "$ITERATIONS" --source-path "$SOURCE_PATH" --method "$METHOD"
        --mesh-res "$MESH_RES" --voxel-size "$MESH_VOXEL_SIZE" --depth-trunc "$DEPTH_TRUNC"
        --sdf-trunc "$SDF_TRUNC" --num-cluster "$NUM_CLUSTER")
    if [[ -d "$run_dir/depth" ]]; then
        echo "已有 depth：跳过训练和渲染，仅融合 mesh。"
        printf '%q ' "${mesh_command[@]}"; printf '\n'
        if ((dry_run)); then return 0; fi
        "${mesh_command[@]}"
        return
    fi
    if [[ -f "$run_dir/config.yml" && -f "$run_dir/point_cloud/iteration_${ITERATIONS}.ply" ]]; then
        echo "已有最终模型：跳过训练，导出 depth 后融合 mesh。"
        if ((dry_run)); then printf '%q ' "${mesh_command[@]}"; printf '\n'; return 0; fi
        export CUDA_VISIBLE_DEVICES="$GPU"
        "${mesh_command[@]}"
        return
    fi
    if [[ -f "$run_dir/config.yml" ]]; then
        echo "已有未完成训练，请选择新的 RUN_NAME 或显式恢复训练，拒绝覆盖。" >&2
        return 1
    fi
    [[ -d "$SOURCE_PATH/images" && -d "$SOURCE_PATH/sparse/0" ]] || {
        echo "数据需要 images/ 和 sparse/0/: $SOURCE_PATH" >&2; return 1;
    }
    local schedule tests step
    schedule="$(make_schedule "$SAVE_ITERATIONS")" || return 1
    tests="$(make_schedule "$TEST_ITERATIONS")" || return 1
    local -a saves test_steps checkpoint_steps=()
    mapfile -t saves < <(printf '%s\n' "$schedule" | sort -nu)
    mapfile -t test_steps < <(printf '%s\n' "$tests" | sort -nu)
    for step in $CHECKPOINT_ITERATIONS; do
        [[ "$step" =~ ^[1-9][0-9]*$ ]] && ((step <= ITERATIONS)) || {
            echo "checkpoint 步数无效或超过总迭代: $step" >&2; return 1;
        }
        checkpoint_steps+=("$step")
    done
    if [[ "$METHOD" == *pgsr ]]; then
        local count
        count="$("$PYTHON_BIN" - "$SOURCE_PATH/sparse/0" <<'PY'
from pathlib import Path
import struct, sys
p = Path(sys.argv[1])
if (p/'images.bin').exists():
    with (p/'images.bin').open('rb') as f:
        print(struct.unpack('<Q', f.read(8))[0])
else:
    count = 0
    with (p/'images.txt').open() as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                count += 1
                next(f)
    print(count)
PY
)" || return 1
        if [[ "$NUM_MULTI_VIEW" == auto ]]; then
            NUM_MULTI_VIEW=$((count > 6 ? 5 : count - 1))
        fi
        [[ "$NUM_MULTI_VIEW" =~ ^[1-9][0-9]*$ ]] && ((NUM_MULTI_VIEW < count)) || {
            echo "NUM_MULTI_VIEW 必须在 1 到 $((count-1)) 之间" >&2; return 1;
        }
        METHOD_ARGS+=(--scene.dataloader.num-multi-view "$NUM_MULTI_VIEW")
        if [[ -f "$SOURCE_PATH/pair.txt" ]]; then
            echo "注意：已有 pair.txt 将优先使用；若选图/邻居参数改变，请先备份并移走该文件。" >&2
        fi
    fi
    local -a command=("$PYTHON_BIN" train.py "$METHOD"
        --source-path "$SOURCE_PATH" --output-path "$OUTPUT_PATH"
        --experiment-name "$EXPERIMENT_NAME" --timestamp "$RUN_NAME" --eval "$EVAL" --machine.seed "$SEED"
        --trainer.iterations "$ITERATIONS"
        --trainer.save-iterations "${saves[@]}" --trainer.test-iterations "${test_steps[@]}"
        --scene.dataloader.resolution "$RESOLUTION"
        --scene.dataloader.shuffle False
        "${METHOD_ARGS[@]}")
    if ((${#checkpoint_steps[@]})); then
        command+=(--trainer.checkpoint-iterations "${checkpoint_steps[@]}")
    fi
    command+=("$@")
    printf '工作目录: %s\nCUDA_VISIBLE_DEVICES=%s ' "$WORKSPACE/GS-SR" "$GPU"
    printf '%q ' "${command[@]}"
    printf '\n'
    printf "后续保存 depth 并提取 mesh: "
    printf '%q ' "${mesh_command[@]}"; printf '\n'
    if ((dry_run)); then return 0; fi
    cd -- "$WORKSPACE/GS-SR"
    export CUDA_VISIBLE_DEVICES="$GPU"
    "${command[@]}"
    "${mesh_command[@]}"
}
