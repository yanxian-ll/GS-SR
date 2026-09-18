# GS-SR 训练脚本

每个方法一个入口，参数集中在各脚本顶部：

- `train_scaffold-2dgs.sh`
- `train_scaffold-pgsr.sh`
- `train_scaffold-2dgs-7k.sh`
- `train_scaffold-pgsr-7k.sh`
- `train_2dgs.sh`
- `train_pgsr.sh`

`_train_common.sh` 提供公共启动逻辑，无需直接运行。脚本自动定位工作区和 GS-SR，不依赖启动时所在目录。默认 `xiaoxiang_03/003`、GPU 0、30000 步、原图分辨率、全部图像参与训练；直接使用 COLMAP 稀疏点初始化。

两个 `*-7k.sh` 不是简单截断 30k，而是把 Scaffold 的结构调整、主要 learning-rate decay 以及对应几何 regularization 的启动时刻一起压缩到 7000 步预算。损失权重、voxel size、offset 数量、appearance dimension 等模型超参数保持与 30k 一致，只改变与训练时间尺度直接相关的参数。

## 7k schedule

7k schedule 以保持 30k 配置中的相对训练进度为原则：

| 参数 | 30k | 7k |
|---|---:|---:|
| `ITERATIONS` | 30000 | 7000 |
| `START_STAT` | 500 | 120 |
| `DENSIFY_FROM_ITER` | 1500 | 350 |
| `DENSIFICATION_INTERVAL` | 100 | 25 |
| `DENSIFY_UNTIL_ITER` | 15000 | 3500 |
| `LR_MAX_STEPS` | 30000 | 7000 |
| 2DGS `START_DIST_LOSS` | 3000 | 700 |
| 2DGS `START_NORMAL_LOSS` | 7000 | 1600 |
| PGSR `START_SINGLE_VIEW` | 3000 | 700 |
| PGSR `START_MULTI_VIEW` | 3000 | 700 |

这样 densification 仍大致占前半段训练，后半段用于固定结构后的外观与几何收敛；Scaffold 的 offset/MLP/appearance exponential LR decay 也会在 7k 结束时走完，而不是停留在 30k schedule 的早期学习率。

所有这些参数仍可通过环境变量覆盖，例如：

```bash
DENSIFICATION_INTERVAL=50 DENSIFY_UNTIL_ITER=4000 \
  bash bash_scripts/train_scaffold-pgsr-7k.sh
```

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

普通 30k Scaffold 入口仍使用原仓库时间尺度；两个专用 7k 入口会自动覆盖为上表 schedule。若直接给普通入口设置较短 `ITERATIONS`，schedule 不会自动缩放，应同时显式覆盖相应时间参数，或直接使用 `*-7k.sh`。

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

30k 入口保留原默认行为；7k 入口复用同一训练与 mesh 管线，仅覆盖训练预算相关 schedule。
