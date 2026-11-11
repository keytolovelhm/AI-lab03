# AI Lab3 · CIFAR-10 图像分类

当代人工智能课程 Lab3 的冻结实验版本。第一阶段比较基础 CNN 和残差 CNN。第二阶段完成六种配置、三个种子的18次正式训练。修正版第三阶段完成七种配置、三个种子的21次正式训练，并分析温度校准、合成扰动和 Grad-CAM。

## 实验结果

第二、三阶段使用种子 20261105、20261106、20261107。以下为统一 CPU FP32 复评的测试准确率，采用三个种子的均值 ± 样本标准差，单位为百分比。

| 阶段 | 配置 | 测试准确率 |
|---|---|---:|
| 二 | 基础 CNN，无增强 | 75.600 ± 0.276 |
| 二 | 基础 CNN，仅裁剪 | 83.067 ± 0.194 |
| 二 | 基础 CNN，仅翻转 | 79.097 ± 0.544 |
| 二 | 基础 CNN，裁剪＋翻转 | **85.533 ± 0.207** |
| 二 | 残差 CNN，无增强 | 75.570 ± 0.454 |
| 二 | 残差 CNN，裁剪＋翻转 | 85.413 ± 0.274 |
| 三 | SE | 85.130 ± 0.191 |
| 三 | CBAM | 85.110 ± 0.249 |
| 三 | Label Smoothing | 85.520 ± 0.193 |
| 三 | Conv Stem，无增强 | 76.537 ± 0.684 |
| 三 | Conv Stem，裁剪＋翻转 | 85.367 ± 0.461 |
| 三 | MixUp | 85.500 ± 0.173 |
| 三 | CutMix | 82.567 ± 0.076 |

组合增强基线比无增强基础 CNN 高9.933个百分点。Label Smoothing 和 MixUp 的平均准确率与组合增强基线接近。三个训练种子使用同一测试集，配对差异用于描述性比较。

基线温度缩放仅在验证集拟合，测试 ECE 从4.092%降至0.953%，预测类别保持一致。自定义 pixel_v1 中心遮挡的五档平均准确率为基线54.401%、CutMix 66.256%。CutMix 的干净准确率为82.567%，高斯噪声表现也低于基线。CutMix 实验每批采用 alpha=1，采样框整体位于图内，按实际覆盖面积修正标签。

- [第一阶段 BF16 汇总](submission_results/stage1_comparison.csv)
- [39次 FP32 复评](submission_results/per_seed.csv)和[分组统计](submission_results/grouped.csv)
- [配对比较](submission_results/contrasts.csv)、[校准统计](submission_results/calibration_grouped.csv)和[扰动统计](submission_results/robustness/robustness_summary.csv)
- [学习曲线](submission_results/baseline_both_20261105/learning_curves.png)、[混淆矩阵](submission_results/baseline_both_20261105/confusion_matrix.png)和[逐类准确率](submission_results/baseline_both_20261105/per_class_accuracy.csv)
- [错误样本](submission_results/baseline_both_20261105/test_errors.png)、[Grad-CAM](submission_results/baseline_both_20261105/gradcam_4x4_vs_8x8.png)、[原始可靠性图](submission_results/baseline_both_20261105/reliability_raw.png)和[校准后可靠性图](submission_results/baseline_both_20261105/reliability_calibrated.png)

## 安装与基线复评

验收环境使用 Python 3.13 和 PyTorch 2.9.1 CPU 版。以下命令在仓库根目录的 PowerShell 中运行。

~~~powershell
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt
& $python -B -c "from src.data import TorontoCIFAR10; TorontoCIFAR10('data', train=True, download=True)"
& $python -B -m refinement_v3 test
~~~

数据准备命令下载并校验官方 CIFAR-10。已有数据可将官方解包后的 cifar-10-batches-py/ 放到 data/ 下。62项测试中有一项读取真实数据。results/split_indices.npz 保存固定的45,000/5,000分层训练验证索引，results/split.json 保存从训练集计算的归一化统计量。模型复评使用官方独立的10,000张测试图像。

所附权重来自组合增强基础 CNN 的原定可视化种子 20261105，对应上方精选图。以下命令使用既有加载器进行 CPU FP32 复评，验收结果为 **8,557 / 10,000，85.57%**。环境变量 LAB3_DATA 可指定包含 cifar-10-batches-py/ 的目录。

~~~powershell
@'
import csv
import os
from pathlib import Path

import torch

from analysis_code.canonical import infer, load_checkpoint, sha
from refinement_v3.data import _normalise
from src.data import TorontoCIFAR10

torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)
path = Path("weights/baseline_both/20261105/best.pt")
with Path("submission_results/per_seed.csv").open(encoding="utf-8", newline="") as handle:
    reference = next(row for row in csv.DictReader(handle)
                     if (row["stage"], row["run"], row["seed"]) == ("2", "baseline_both", "20261105"))
if sha(path) != reference["checkpoint_sha256"]:
    raise ValueError("Checkpoint SHA-256 differs from the frozen evaluation")
model, checkpoint = load_checkpoint(path)
dataset = TorontoCIFAR10(os.environ.get("LAB3_DATA", "data"), train=False, download=False)
stats = checkpoint["normalization"]
if dataset.classes != stats["classes"]:
    raise ValueError("CIFAR-10 class order differs from the checkpoint")
images = _normalise(dataset.data, stats["train_mean"], stats["train_std"])
predictions = infer(model, images).argmax(1)
correct = int((predictions == torch.tensor(dataset.targets)).sum())
print(f"CPU FP32: {correct}/{len(dataset)} ({correct / len(dataset):.2%})")
'@ | & $python -B -
~~~

## 复现实验矩阵

src/ 和 configs/experiment.json 对应第一阶段。refinement/ 和 refinement/experiment.json 对应第二阶段。refinement_v3/ 保存修正版第三阶段训练评估代码、配置和62项测试。analysis_code/ 包含 FP32 复评、校准、鲁棒性和派生结果核验脚本。阶段二、三使用 BF16 训练，按验证准确率选择检查点，同分时比较验证损失。两套训练各自从60轮开始，以验证指标统一决定是否延长至80轮，早停耐心为12轮。

以下命令适用于已准备数据的独立检出目录。第一阶段写入 refinement/legacy_reproduction/。

~~~powershell
& $python -B -m src train
& $python -B -m src evaluate
~~~

第二阶段先对当前检出的输入建立本地快照。snapshot 在数据到位后执行一次。产物写入 refinement/runs/suite/。

~~~powershell
& $python -B -m refinement snapshot
& $python -B -m refinement train
& $python -B -m refinement evaluate
~~~

第三阶段的完整矩阵写入 refinement_v3/runs/suite_corrected/。以下命令适用于该目录尚无历史存档的检出副本。

~~~powershell
& $python -B -c "import os; os.environ['LAB3_SUITE']='suite_corrected'; from refinement_v3.corrected import train; train()"
& $python -B -c "import os; os.environ['LAB3_SUITE']='suite_corrected'; from refinement_v3.corrected import evaluate; evaluate()"
~~~

完成第二、三阶段的训练和评估后，或取得同样目录结构的完整正式存档后，可以重新计算39组结果。运行前须具备18＋21份权重、训练历史、测试结果、逐图预测和封存清单。复算输出放在独立目录，公开的精选结果位于 submission_results/。

~~~powershell
& $python -B -m analysis_code.final_acceptance recompute --out reproduced/fp32
& $python -B -m analysis_code.finish_results --out reproduced/fp32
& $python -B -m analysis_code.robustness_corrected --out reproduced/fp32/robustness
~~~

全量历史 verify、protect、qualitative 和 zcost 子命令依赖完整本地存档及对应清单。公开检出在准备数据后可运行单元测试和复评所附权重。公开的逐种子表可用于复核聚合结果。完整模型复评需要取得存档或按上述流程生成训练产物。重训结果的逐位一致性还依赖硬件和库版本。
