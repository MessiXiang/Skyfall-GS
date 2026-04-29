# VGGT 引导二阶段 IDU 采样说明

本仓库已把二阶段 IDU 从“平均采样所有伪相机并全部 diffusion 优化”改成可选的 **VGGT 低置信度引导采样**：

1. 先按课程学习的高到低视角（elevation list）生成更多候选伪相机。
2. 仅用 3DGS 快速渲染候选图。
3. 用 VGGT 预测每张候选图的 confidence map。
4. 按低置信度分数筛出最值得修复的少量视角。
5. 只对这些低置信度视角执行 FlowEdit/Difix3D/DreamScene 等 diffusion refine 和深度估计。

这样可以减少 diffusion 优化图片数量，加速二阶段。

## 需要手动安装的第三方 GitHub/submodule 依赖

按你的要求，这些来自 GitHub/submodule 的包不自动安装，请手动处理。

### 1. VGGT

仓库代码默认从 `submodules/vggt` 导入 VGGT。若目录不存在，请在项目根目录执行：

```bash
git submodule add https://github.com/facebookresearch/vggt.git submodules/vggt
# 或者如果 .gitmodules 已经配置过：
# git submodule update --init --recursive submodules/vggt
```

然后进入你的 conda 环境安装 VGGT 依赖：

```bash
conda activate skyfall-gs
pip install -r submodules/vggt/requirements.txt
```

### 2. FlowEdit / 3DGS 相关 submodules

二阶段默认 refinement 仍使用 FlowEdit；3DGS rasterizer/simple-knn/fused-ssim 也仍然是必要的。若这些目录为空，请按原 README 初始化：

```bash
git submodule update --init --recursive
conda activate skyfall-gs
pip install --no-build-isolation submodules/diff-gaussian-rasterization-depth
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/fused-ssim
```

FlowEdit 若有独立 requirements，请根据 `submodules/FlowEdit` 自带文档安装。

## Hugging Face 模型缓存与镜像

代码已设置：

- `HF_ENDPOINT=https://hf-mirror.com`
- `HF_HOME=/root/autodl-tmp/huggingface`
- `HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/huggingface/hub`
- `TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/hub`

VGGT 会先尝试 `local_files_only=True` 从本地缓存加载；如果没有缓存，才会从 hf-mirror 下载到 `/root/autodl-tmp/huggingface/hub/`。

## 新增关键参数

- `--idu_vggt_guided_sampling`：启用 VGGT 低置信度引导采样。
- `--idu_vggt_candidate_multiplier 3`：候选视角数量倍数。例如原来每 target 6 个方向，现在先生成 18 个候选。
- `--idu_vggt_keep_ratio 0.35`：在候选中保留的比例基准。实际保留比例会除以 `candidate_multiplier`，用于减少最终 refinement 数量。
- `--idu_vggt_min_keep 4`：每轮至少保留多少候选视角。
- `--idu_vggt_confidence_percentile 20`：用 confidence map 的低分位数作为图像置信度评分；越低越会被选中。
- `--idu_vggt_confidence_batch_size 4`：VGGT 打分批大小，显存不够可调小。
- `--idu_depth_estimator vggt`：二阶段伪深度也使用 VGGT。
- `--idu_vggt_model_name facebook/VGGT-1B`：VGGT 预训练模型名。

## 推荐二阶段命令片段

在原二阶段命令基础上加入：

```bash
--idu_vggt_guided_sampling \
--idu_depth_estimator vggt \
--idu_vggt_model_name facebook/VGGT-1B \
--idu_vggt_candidate_multiplier 3 \
--idu_vggt_keep_ratio 0.35 \
--idu_vggt_min_keep 4 \
--idu_vggt_confidence_percentile 20 \
--idu_vggt_confidence_batch_size 4
```

`scripts/run_jax_idu.py` 和 `scripts/run_nyc_idu.py` 已默认加入这些参数。

## 输出检查

每个 IDU episode 会在下面目录保存 VGGT 筛选日志：

```text
<model_path>/idu/e<elevation>_r<radius>/vggt_confidence/selected_candidates.csv
```

其中：

- `score` 越大表示 VGGT confidence 越低，越需要 refinement。
- `selected=1` 表示该候选图会进入 diffusion refine 和 IDU 训练集。

## 调参建议

- 想更快：降低 `--idu_vggt_keep_ratio`，如 `0.2`。
- 想质量更稳：提高 `--idu_vggt_keep_ratio`，如 `0.5`。
- VGGT 打分显存不足：降低 `--idu_vggt_confidence_batch_size` 到 `1` 或 `2`。
- 候选覆盖不够：提高 `--idu_vggt_candidate_multiplier` 到 `4`，但 VGGT 打分会变慢。
