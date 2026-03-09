# OpTC-URAS 用户操作指南 (User Guide)

本文档旨在帮助用户从零开始配置、运行并评估 **OpTC-URAS** 项目。

---

## 1. 数据集准备与配置

本项目支持标准的 OpTC 数据集（JSONL 格式）或 ECAR 格式数据。

### 1.1 数据路径设置
在配置文件 `configs/ecar.yaml` 中，`data` 部分定义了明确的训练、验证和测试数据来源：

```yaml
data:
  optc:
    # 基础目录
    data_dir: "data/raw"
    cache_dir: "data/cache"
    
    # [关键] 显式指定数据集路径
    # 训练集：启用数据增强 (Augmentation) 和随机采样
    train_path: "data/raw/train.jsonl"
    train_prefix: "train"
    
    # 验证集：全量保留 (无截断/采样)，用于模型评估
    val_path: "data/raw/val.jsonl"
    val_prefix: "val"
    
    # 测试集：全量保留 (无截断/采样)，用于最终推理
    test_path: "data/raw/test.jsonl"
    test_prefix: "test"
    
    # 聚合窗口大小（分钟）
    window_minutes: 15
```

### 1.2 数据预处理逻辑 (Preprocessing Logic)
`preprocess.py` 脚本会根据配置文件中的定义，对不同数据集采取不同的处理策略：

1.  **Training Set (训练集)**: 
    *   **处理策略**: **增强与采样 (Augmentation & Sampling)**。
    *   **逻辑**: 先收集窗口内的所有事件，若超过 `max_events`，则基于总事件量进行增强（1x/3x/5x/10x），并在增强后进行全局随机采样。
    *   **目的**: 保证训练数据的分布多样性，避免简单截断丢失关键信息。
2.  **Validation & Test Set (验证与测试集)**:
    *   **处理策略**: **全量保留 (Full/Infinite)**。
    *   **逻辑**: 设置 `max_events=None`，不做任何截断或丢弃。
    *   **目的**: 确保评估和测试的客观性，真实反映模型在完整流量下的表现。

---

## 2. 核心参数配置详解

所有可调参数均在 `configs/ecar.yaml` 中。以下是关键参数的详细解释：

### 2.1 联邦学习与隐私 (FL & Privacy)
```yaml
training:
  federated_learning: true   # true=开启联邦模拟，false=中心化训练（调试用）

model:
  feature_dp:
    enabled: true            # 是否开启特征差分隐私
    noise_sigma: 0.01        # [关键] 添加到特征上的高斯噪声强度。值越大，隐私越强，但检测精度可能下降。
    clip_C: 10.0             # 特征向量的裁剪阈值（L2 Norm），防止异常值破坏隐私预算。
```

### 2.2 模型架构 (Model Architecture)
```yaml
model:
  step1:
    lora_rank: 4             # [关键] LoRA 适配器的秩。
                             # 作用：控制 Step 1 微调的自由度。
                             # 推荐值：4 或 8。太小会导致欠拟合，太大会导致联邦场景下的灾难性遗忘。
    num_subspaces: 4         # URAS 划分的子空间数量。
    
  teacher:
    temp: 0.1                # InfoNCE 温度系数。越小，模型对相似度的区分越敏锐。
    augment_mask_p: 0.2      # 数据增强：随机 Mask 掉 20% 的特征。
    augment_noise_std: 0.01  # 数据增强：添加 0.01 的高斯噪声。
```

### 2.3 异常检测 (Detector)
```yaml
model:
  scd_loss:
    lambda_d: 1.0            # 风格-内容解耦损失权重
    lambda_r: 1.0            # 重建损失权重
    lambda_v: 25.0           # [关键] 异常分数损失权重。控制模型对异常样本的敏感度。
    
  atc:
    alpha_conf: 0.0          # 自适应阈值：置信度权重
    alpha_unc: 0.0           # 自适应阈值：不确定性权重
    alpha_risk: 0.0          # 自适应阈值：风险权重
    
  view_sensitivities:        # [工程参数] 视图风险权重
    process: 1.0
    file: 1.0
    network: 1.0             # 如果希望对网络攻击更敏感，可将此值调大（如 2.0）。
```

---

## 3. 执行流程 (Execution Workflow)
项目代码经过重构，支持模块化的分阶段执行。

### 第一步：预处理 (Preprocessing)
读取原始日志，生成 `.pkl` 缓存文件。
```bash
# 自动处理 Train (增强+采样) 和 Val/Test (全量)
python preprocess.py --config configs/ecar.yaml
```

### 第二步：分阶段训练与推理
您可以按顺序执行以下命令，也可以单独运行某个阶段进行调参：

#### 阶段 1: 教师模型预训练 (Teacher Training)
```bash
python main.py train_teacher --config configs/ecar.yaml
```
*   **输入**: Train Set (部分), Val Set
*   **输出**: `teacher_checkpoint.pt`

#### 阶段 2: 学生模型/联邦学习训练 (Student Training)
```bash
python main.py train_student --config configs/ecar.yaml
```
*   **输入**: Teacher Checkpoint, Train Set, Val Set
*   **输出**: `step1_checkpoint.pt`, `student_checkpoint.pt`

#### 阶段 3: 检测器训练 (Detector Training)
```bash
python main.py train_detector --config configs/ecar.yaml
```
*   **输入**: Student Checkpoints, Train Set (用于拟合分布)
*   **输出**: `detector_checkpoint.pt`, 统计阈值

#### 阶段 4: 测试与推理 (Test/Inference)
```bash
python main.py test --config configs/ecar.yaml
```
*   **输入**: 所有 Checkpoints, Test Set
*   **输出**: `detection_results.csv`

> **Note**: 如果您希望像以前一样一键跑通所有流程，可以使用 `all` 模式：
> ```bash
> python main.py all --config configs/ecar.yaml
> ```

### 第三步：评估 (Evaluation)
计算 Precision, Recall, F1, AUC 等指标。
```bash
python evaluate.py results4/detection_results.csv
```
*输出*：控制台打印指标，并生成 `detection_metrics.json`。
