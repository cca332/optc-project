# OpTC-URAS 代码架构导读 (Code Walkthrough)

本文档深入解析项目代码结构，帮助开发者理解各个模块的功能及相互联系。

---

## 1. 顶层目录结构

```text
src/optc_uras/
├── data/       # 数据层：负责读取、解析和封装数据
├── features/   # 特征层：负责从原始日志提取统计特征和行为特征
├── models/     # 模型层：包含 Step 1, Teacher, Student, Detector 的核心网络定义
├── federated/  # 联邦层：实现 Client/Server 交互、差分隐私和安全聚合逻辑
└── utils/      # 工具层：日志、配置、类型定义
```

---

## 2. 核心文件详解

### 2.1 数据流起点 (`data/` & `features/`)

*   **`data/ecar.py`**: 
    *   **功能**: 原始 JSONL 读取器。
    *   **核心类**: `EcarJsonlReader` (流式读取), `WindowAggregator` (时间窗口聚合)。
*   **`features/deterministic_aggregator.py`**:
    *   **功能**: 将聚合后的日志转换为数值特征向量。
    *   **机制**: 使用 Hash Hashing 处理高维字段（如文件名），使用 Vocabulary 处理低维字段（如 Event Type）。
*   **`features/quality.py`**:
    *   **功能**: 计算视图质量指标（Validity, Completeness, Entropy）。
    *   **作用**: 为 Step 1 的加权融合提供依据。

### 2.2 模型核心 (`models/`)

*   **`models/step1.py` (Fusion)**:
    *   **核心类**: `Step1Model`。
    *   **逻辑**: 
        1.  **Feature Extraction**: 调用 Aggregator 提取特征。
        2.  **Quality Weighting**: 根据质量指标计算视图权重 $w_v$。
        3.  **Router (LoRA)**: 使用带有 LoRA 适配器的 MLP 将特征映射到潜在空间。
        4.  **Gated Fusion**: 融合多视图特征生成 $z$。
    *   **关键点**: `LoRALayer` 的实现允许在冻结 Base 参数的同时进行微调。

*   **`models/teacher.py` (Phase 1)**:
    *   **核心类**: `TeacherModel`。
    *   **逻辑**: 实现了 `augment` (Masking/Jitter) 和 `forward_contrastive` (InfoNCE Loss)。
    *   **作用**: 预训练出鲁棒的特征提取器，指导 Student。

*   **`models/student.py` (Phase 2)**:
    *   **核心类**: `StudentHeads`。
    *   **逻辑**: 包含多个 Projection Head，用于将 $z$ 映射到多个子空间。
    *   **作用**: 联邦学习的主体，学习 Teacher 的分布。

*   **`models/detector.py` (Phase 3)**:
    *   **核心类**: `AnomalyDetector`。
    *   **逻辑**: 包含 `SCD` 模块（解耦 Style/Content）和 `ATC` 逻辑（计算动态阈值）。

### 2.3 联邦架构 (`federated/`)

*   **`federated/client.py`**:
    *   **功能**: 模拟单个 Client 的行为。
    *   **流程**: `train_epoch()` -> 计算 Loss -> 更新本地 Student 和 Step 1 (LoRA) -> 添加 DP 噪声 -> 返回更新。
*   **`federated/server.py`**:
    *   **功能**: 模拟中心服务器。
    *   **流程**: `aggregate()` -> 接收多个 Client 的更新 -> 执行 FedAvg (加权平均) -> 广播新参数。
*   **`federated/dp.py`**:
    *   **功能**: 差分隐私工具。
    *   **实现**: `dp_features()` 实现了高斯机制（Gaussian Mechanism）的噪声添加。

---

## 3. 全流程串联 (`main.py`)

`main.py` 是整个项目的入口脚本，经过重构后，它采用了模块化设计，通过 `argparse` 子命令分发任务。主要包含以下独立函数：

1.  **`run_train_teacher()` (Phase 1)**:
    *   加载 `train` 和 `val` 数据集。
    *   初始化并训练 `TeacherModel` (InfoNCE)。
    *   保存 `teacher_checkpoint.pt`。

2.  **`run_train_student()` (Phase 2)**:
    *   加载 Teacher Checkpoint。
    *   初始化 `Step1Model` (Feature Fusion) 和 `StudentHeads`。
    *   执行联邦学习循环 (FedAvg) 或中心化训练。
    *   保存 `step1_checkpoint.pt` 和 `student_checkpoint.pt`。

3.  **`run_train_detector()` (Phase 3)**:
    *   加载 Student Checkpoints。
    *   在训练集上提取特征 (URAS)。
    *   训练 `AnomalyDetector` (SCD Loss)。
    *   计算并保存统计阈值 (Threshold)。

4.  **`run_test()` (Inference)**:
    *   加载所有训练好的模型。
    *   读取 `test` 数据集 (Test Prefix)。
    *   执行完整推理链路，输出 `detection_results.csv`。

5.  **`main()`**:
    *   解析命令行参数 `mode` (`train_teacher` | `train_student` | `train_detector` | `test` | `all`)。
    *   根据模式调用上述对应函数。
