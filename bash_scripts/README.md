# GS-SR 训练脚本

每个方法一个入口，参数集中在各脚本顶部：

- `train_scaffold-2dgs.sh`
- `train_scaffold-pgsr.sh`
- `train_scaffold-2dgs-7k.sh`
- `train_scaffold-pgsr-7k.sh`
- `train_2dgs.sh`
- `train_pgsr.sh`

`_train_common.sh` 提供公共启动逻辑，无需直接运行。脚本自动定位工作区和 GS-SR，不依赖启动时所在目录。默认 `xiaoxiang_03/003`、GPU 0、30000 步、原图分辨率、全部图像参与训练；直接使用 COLMAP 稀疏点初始化。

其中两个 `*-7k.sh` 是对应 Scaffold 方法的 7000 步入口，默认 `RUN_NAME=pipeline-7k`，其余方法参数与对应 30k 脚本保持一致。因此它们等价于把原 30k 配置截断到第 7000 步，不会自动压缩 normal loss、multi-view loss、densification 或 learning-rate schedule 的起止步数。

## 运行

在工作区根目录：

```bash
conda activate 3dgs
bash bash_scripts/train_scaffold-2dgs.sh
bash bash_scripts/train_scaffold-pgsr.sh
bash bash_scripts/train_scaffold-2dgs-7k.sh
bash bash_scripts/train_scaffold-pgsr-7k.sh
bash bash_scripts/train_2dgs.sh
bash bash_scripts/train_pgsr.sh
```

上述为六条独立训练命令。单 GPU 建议依次运行。

先检查命令，不训练、不修改数据：

```bash
bash bash_scripts/train_scaffold-pgsr.sh --dry-run
bash bash_scripts/train_scaffold-pgsr-7k.sh --dry-run
```

切换场景、视角数与常用参数：

```bash
SCENE=xiaoxiang_03 VIEWS=005 GPU=0 bash bash_scripts/train_2dgs.sh
SCENE=xiaoxiang_03 VIEWS=010 RESOLUTION=2 bash bash_scripts/train_scaffold-pgsr.sh
SCENE=xiaoxiang_03 VIEWS=003 GPU=0 bash bash_scripts/train_scaffold-2dgs-7k.sh
SCENE=xiaoxiang_03 VIEWS=005 GPU=1 bash bash_scripts/train_scaffold-pgsr-7k.sh
ITERATIONS=1000 SAVE_ITERATIONS=500 bash bash_scripts/train_scaffold-2dgs.sh
```

较短训练仅用于检查启动；损失与增密阶段仍按脚本配置的起始步数启用，不会随总迭代自动缩短。

自定义数据路径和实验名：

```bash
SOURCE_PATH=/path/to/subset EXPERIMENT_NAME=my_scene_003 \
  OUTPUT_PATH=/path/to/output bash bash_scripts/train_pgsr.sh
```

`PYTHON_BIN` 可指定 Python 绝对路径；`CHECKPOINT_ITERATIONS="15000 30000"` 可额外保存恢复训练用 checkpoint。保存模型/测试步数过滤掉超过总迭代的项目，并自动加入最后一步。其余 GS-SR 参数可追加在命令尾部（不要重复指定脚本已设置的选项，用对应顶部变量修改）。

## PGSR 视角选择

默认基于稀疏点选择邻居，`NUM_MULTI_VIEW=auto` 表示 `min(5,注册图像数-1)`。3 张对应 2，5 张对应 4，10 张对应 5。可手动设置 `NUM_MULTI_VIEW=2`。

固定 `shuffle=False` 保持缓存视角顺序稳定。已有 `pair.txt` 会被 GS-SR 优先读取；更改选图、邻居数或使用先前随机顺序生成的文件时，应先备份并移走它，让 GS-SR 重建。脚本不会自动删除已有文件。

默认 `EVAL=False`，避免从少量图像中继续划分测试视角。若开启 eval，应重新核对 PGSR 邻居数和 pair.txt 与实际训练图像集合一致。

## 输出与表面提取

默认保存到 `GS-SR/output/<场景>_<视角数>/<方法>/<时间戳>/`，包含 `config.yml`、训练日志和保存的模型。

```bash
cd GS-SR
python script/extract_mesh.py \
  --load-config output/xiaoxiang_03_003/scaffold-2dgs/<时间戳>/config.yml \
  --skip-video
```

四个原始入口已通过 Bash 语法、dry-run 和实际 GS-SR 参数解析检查；新增 7k 入口复用对应原始 Scaffold 脚本与 `_train_common.sh` 的启动逻辑。
