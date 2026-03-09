# OpTC-URAS: Unified Representation & Anomaly Surveillance

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

本项目是 **OpTC-URAS** 的官方实现代码库。这是一个面向企业级系统日志（如 OpTC, ECAR）的**联邦隐私保护异常检测框架**。它采用创新的“三阶段”流水线，在保证数据隐私（不上传原始日志）的前提下，从分散的多视图数据（进程、文件、网络）中学习鲁棒的行为表征，并实现高精度的异常检测。

---

## 📚 核心方法 (Methodology)

本框架由三个核心阶段组成，旨在解决系统日志检测中的**多源异构性**、**数据隐私**和**分布漂移**问题：

### Step 1: 同源多子视图表征提取与融合 (Multi-View Fusion)
> **代码位置**: [`src/optc_uras/models/step1.py`](src/optc_uras/models/step1.py)

针对单一主机产生的多维度日志（View），我们设计了一个质量感知的融合模块：
*   **确定性特征提取**: 使用哈希统计聚合 (Deterministic Aggregator) 和固定随机投影 (Fixed Random Projection) 将原始日志转为低维向量。
*   **质量加权 (Quality-Awareness)**: 自动计算每个视图的“完整性”、“有效性”和“信息熵”，生成可靠性权重 $w$。
*   **路由对齐 (Routing & Alignment)**: 通过可学习的 Router 和对齐基矩阵，将不同视图映射到统一语义空间。
    *   *Update*: 引入 **LoRA (Low-Rank Adaptation)** 对 Router/Fusion 进行轻量级微调，有效缓解联邦学习中的灾难性遗忘。
*   **门控融合 (Gated Fusion)**: 基于视图间的一致性（Pearson相关系数）动态门控，生成最终的样本级 Latent Representation $z$。

### Step 2: 隐私感知联邦预训练 (Federated Pretraining)
> **代码位置**: [`src/optc_uras/models/student.py`](src/optc_uras/models/student.py), [`src/optc_uras/federated/`](src/optc_uras/federated/)

采用 **Teacher-Student** 蒸馏架构进行联邦自监督学习：
*   **Teacher (Central)**: 在少量公开/脱敏数据上使用 **InfoNCE 对比学习**（配合 Masking/Jitter 数据增强）进行预训练，随后冻结作为知识引导。
*   **Student (Federated)**: 分布在各 Client 端，通过联邦平均 (FedAvg) 协同训练。
*   **隐私保护 (Privacy)**:
    *   **Feature DP**: 对上传的特征添加高斯噪声并裁剪。
    *   **Gradient DP**: 对模型更新梯度进行差分隐私处理。
    *   **Secure Aggregation**: 模拟安全聚合协议，确保服务器无法窥探单客户端更新。
*   **URAS**: 最终生成的统一表征 (Unified Representation) 是 Student 多个子空间投影的拼接。

### Step 3: 风格-内容解耦异常检测 (SCD Detection)
> **代码位置**: [`src/optc_uras/models/detector.py`](src/optc_uras/models/detector.py)

在良性数据上训练的异常检测器：
*   **SCD (Style-Content Disentanglement)**: 将 URAS 表征解耦为“风格” (Style, 稳定的良性模式) 和“内容” (Content, 样本特异性噪声)。
*   **异常评分**: 基于 Style 向量的马氏距离 (Mahalanobis Distance) 计算异常分数。
*   **ATC (Adaptive Threshold Correction)**: 推理时根据样本的**置信度** (Confidence)、**不确定性** (Uncertainty) 和**隐私风险** (Risk) 动态调整检测阈值。
*   **可解释性 (Interpreter)**: 提供归因分析，指出导致异常的具体视图（如“主要是 Network 异常”）。

---

## 📂 项目结构 (Project Structure)

符合顶级会议（CVPR/ICLR/CCS）标准的工程结构：

```text
optc-project/
├── configs/            # 实验配置文件 (YAML)
│   └── ecar.yaml       # 默认 ECAR 数据集配置
├── src/                # 核心源码包 (optc_uras)
│   └── optc_uras/
│       ├── models/     # 模型定义
│       │   ├── step1.py       # Step 1 融合模型
│       │   ├── detector.py    # Step 3 SCD 检测器
│       │   ├── teacher.py     # Teacher 模型
│       │   └── ...
│       ├── features/   # 特征工程 (Aggregators, Quality)
│       ├── federated/  # 联邦学习 (Client, Server, DP)
│       ├── data/       # 数据加载与处理
│       └── utils/      # 通用工具
├── main.py             # [主入口] 全流程训练与推理脚本
├── preprocess.py       # [工具] 数据预处理脚本
├── evaluate.py         # [工具] 评估指标计算脚本
├── requirements.txt    # 依赖列表
└── README.md           # 项目文档
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

```bash
# 建议使用 Python 3.8+
pip install -r requirements.txt
```

### 2. 数据预处理

将原始 JSONL 日志转换为模型可读的时间槽切片数据。支持真实 OpTC 数据格式或 ECAR 格式。

```bash
# 默认读取 configs/ecar.yaml 中配置的路径
python preprocess.py --config configs/ecar.yaml
```

### 3. 运行全流程 (Pipeline)

项目支持模块化分阶段执行，也支持一键运行：

```bash
# 阶段 1: 教师模型预训练 (Pretrain Teacher)
python main.py train_teacher --config configs/ecar.yaml

# 阶段 2: 学生模型/联邦学习训练 (Train Student)
python main.py train_student --config configs/ecar.yaml

# 阶段 3: 检测器训练 (Train Detector)
python main.py train_detector --config configs/ecar.yaml

# 阶段 4: 测试与推理 (Test/Inference)
python main.py test --config configs/ecar.yaml

# (可选) 一键顺序执行所有阶段
python main.py all --config configs/ecar.yaml
```

*   **输出**: 运行结束后，结果会保存在 `results*/` 目录下，包含模型 Checkpoints、训练日志以及 `detection_results.csv`。

### 4. 评估指标 (Evaluation)

使用提供的评估脚本，计算准确率、AUC、F1 等核心指标。脚本内置了 Ground Truth 对齐逻辑（支持 UTC/EDT 时间自动转换）。

```bash
# 评估生成的检测结果 (默认支持 UTC 时间戳)
python evaluate.py results4/detection_results.csv
```

---

## ⚙️ 配置说明 (Configuration)

所有实验参数均在 `configs/ecar.yaml` 中集中管理。关键参数说明：

### 基础配置
*   **`data.optc`**: 数据路径与缓存设置。
*   **`model.target_dim`**: Step 1 输出维度。
*   **`model.num_subspaces`**: Step 2 子空间数量。

### 核心算法配置 (Advanced)
*   **LoRA (Step 1)**:
    *   `model.step1.lora_rank`: LoRA 适配器的秩 (Rank)，推荐 4 或 8。
*   **Teacher (Phase 1)**:
    *   `model.teacher.temp`: InfoNCE 温度系数。
    *   `model.teacher.augment_mask_p`: 数据增强 Masking 概率。
*   **Privacy (FL)**:
    *   `model.feature_dp.noise_sigma`: 特征差分隐私噪声强度。
*   **Detector (Step 3)**:
    *   `model.scd_loss`: SCD 损失函数的各项权重 ($\lambda_d, \lambda_r, \lambda_v$)。
    *   `model.atc`: 自适应阈值校正的系数 (`alpha_conf`, `alpha_unc`, `alpha_risk`)。
    *   `model.view_sensitivities`: 视图敏感度权重 (Process/File/Network)，用于计算风险分数。

---

## 📊 结果示例

`detection_results.csv` 输出格式如下：

| timestamp | host | anomaly_score | adaptive_threshold | is_anomaly | top_attribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2019-09-25T14:15:00 | SysClient0051 | 12.45 | 5.21 | 1 | ('process', 0.82) |
| 2019-09-25T14:30:00 | SysClient0201 | 0.32 | 5.10 | 0 | None |

---

## 📜 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。
