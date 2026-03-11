# OpTC-URAS 用户操作指南 (User Guide)

本文档旨在帮助用户从零开始配置、运行并评估 **OpTC-URAS** 项目。

**Update (2026-03)**: 指南已更新以反映 **Step1A 深度语义驱动架构** 的配置与操作。

---

## 1. 数据集准备与配置

本项目支持标准的 OpTC 数据集（JSONL 格式）或 ECAR 格式数据。

### 1.1 数据路径设置
在配置文件 `configs/final_production.yaml` 中，`data` 部分定义了明确的训练、验证和测试数据来源：

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
    
    # 验证集与测试集
    val_path: "data/raw/val.jsonl"
    val_prefix: "val"
    test_path: "data/raw/test.jsonl"
    test_prefix: "test"
    
    # [Update] 聚合窗口大小（分钟）
    # 推荐使用 5 分钟窗口，以减少背景噪声对攻击语义的稀释
    window_minutes: 5
```

### 1.2 数据预处理逻辑 (Preprocessing Logic)
`preprocess.py` 脚本会根据配置文件中的定义，对不同数据集采取不同的处理策略：

1.  **全量窗口 (Full Window)**: 
    *   **策略**: 不再对窗口内的总事件数进行截断 (`max_events=Inf`)。
    *   **目的**: 确保攻击序列（即使出现在窗口末尾）不会被丢弃。
2.  **语义提取 (Semantic Extraction)**:
    *   **策略**: 深度解析原始日志，提取 `Payload`, `Command Line`, `IP:Port`, `Registry Key` 等高价值语义字段，并保留在预处理结果中。
    *   **目的**: 为 Step 1 的可学习语义编码器提供丰富的上下文。

---

## 2. 核心参数配置详解

所有可调参数均在 `configs/final_production.yaml` 中。以下是关键参数的详细解释：

### 2.1 模型架构 (Step 1A Semantic)
```yaml
model:
  # [Update] Slot 长度（秒）
  # 5min 窗口划分为 10 个 30s Slot，用于时序 Transformer 建模
  slot_seconds: 30
  
  # [Update] 残差质量注入强度
  # 范围 [0.0, 1.0]。控制质量权重对主语义特征的影响程度。
  # 0.5 是一个平衡点，既利用了质量信息，又防止其过度主导。
  quality_injection_lambda: 0.5
    
  step1:
    num_subspaces: 4         # URAS 划分的子空间数量。
```

### 2.2 联邦学习与隐私 (FL & Privacy)
```yaml
training:
  federated_learning: true   # true=开启联邦模拟
```

### 2.3 异常检测 (Detector)
```yaml
model:
  scd_loss:
    lambda_d: 10.0           # 风格-内容解耦损失权重
    lambda_r: 10.0           # 重建损失权重
    lambda_v: 1.0            # 异常分数损失权重
    gamma: 1.0               # [Update] 修正为 1.0，避免梯度爆炸
```

---

## 3. 执行流程 (Execution Workflow)
项目代码经过重构，支持模块化的分阶段执行。

### 第一步：预处理 (Preprocessing)
读取原始日志，生成语义丰富的切片数据。
```bash
# 读取 configs/final_production.yaml
python preprocess.py --config configs/final_production.yaml
```

### 第二步：分阶段训练与推理
您可以按顺序执行以下命令，也可以单独运行某个阶段进行调参：

#### 阶段 1: 教师模型预训练 (Teacher Training)
```bash
python main.py train_teacher --config configs/final_production.yaml
```

#### 阶段 2: 学生模型/联邦学习训练 (Student Training)
```bash
python main.py train_student --config configs/final_production.yaml
```

#### 阶段 3: 检测器训练 (Detector Training)
```bash
python main.py train_detector --config configs/final_production.yaml
```

#### 阶段 4: 测试与推理 (Test/Inference)
```bash
python main.py test --config configs/final_production.yaml
```

> **Note**: 一键跑通所有流程：
> ```bash
> python main.py all --config configs/final_production.yaml
> ```

### 第三步：评估 (Evaluation)
计算 Precision, Recall, F1, AUC 等指标。
```bash
python evaluate.py experiments/result1/detection_results.csv
```
*输出*：控制台打印指标，并生成 `detection_metrics.json`。
