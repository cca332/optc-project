# OpTC-URAS: Unified Representation & Anomaly Surveillance

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

本项目是 **OpTC-URAS** 的官方实现代码库。这是一个面向企业级系统日志（如 OpTC, ECAR）的**联邦隐私保护异常检测框架**。它采用创新的“三阶段”流水线，在保证数据隐私（不上传原始日志）的前提下，从分散的多视图数据（进程、文件、网络）中学习鲁棒的行为表征，并实现高精度的异常检测。

**Update (2026-03)**: 系统已升级为 **Step1A 深度语义驱动架构**，引入了可学习事件编码器、时序 Transformer 和残差质量注入机制，显著提升了对复杂攻击序列的检测能力。

---

## 📚 核心方法 (Methodology)

本框架由三个核心阶段组成，旨在解决系统日志检测中的**多源异构性**、**数据隐私**和**分布漂移**问题：

### Step 1: 语义驱动的多视图表征提取与融合 (Semantic-driven Multi-View Fusion)
> **代码位置**: [`src/optc_uras/models/step1.py`](src/optc_uras/models/step1.py)

针对单一主机产生的多维度日志（View），我们设计了一个深度语义驱动的融合模块：

*   **语义特征提取 (Semantic Feature Extraction)**:
    *   **预处理 (Step0)**: 采用 **5min 全量窗口**，无截断提取 `Payload`, `Command Line`, `IP:Port` 等丰富语义字段。
    *   **可学习编码 (A2)**: 使用 **Learnable Event Encoder** 将离散事件映射为语义向量，替代旧的统计哈希。
    *   **时序建模 (A3)**: 引入 **Slot Temporal Transformer**，捕捉 Slot 序列中的长程时序依赖。
    *   **视图摘要 (A4)**: 使用 **Masked Attention Pooling** 生成视图级摘要向量。
*   **残差质量注入 (Residual Quality Injection)**:
    *   基于统计指标（完整性、有效性、熵）计算质量权重 $w$。
    *   采用 **$1/V$ 残差注入公式** $\tilde{s} = (1 + \lambda(w - 1/V)) \bar{s}$，在保证语义主导的前提下，轻量级地调节视图可信度。
*   **路由对齐 (Routing & Alignment)**: 通过可学习的 Router 和对齐基矩阵，将不同视图映射到统一语义空间。
*   **门控融合 (Gated Fusion)**: 基于视图间的一致性动态门控，生成最终的样本级 Latent Representation $z$。

### Step 2: 隐私感知联邦预训练 (Federated Pretraining)
> **代码位置**: [`src/optc_uras/models/student.py`](src/optc_uras/models/student.py), [`src/optc_uras/federated/`](src/optc_uras/federated/)

采用 **Teacher-Student** 蒸馏架构进行联邦自监督学习：
*   **Teacher (Central)**: 在少量公开/脱敏数据上使用 **InfoNCE 对比学习**进行预训练。
*   **Student (Federated)**: 分布在各 Client 端，通过联邦平均 (FedAvg) 协同训练。
*   **隐私保护 (Privacy)**:
    *   **Feature DP**: 对上传的特征添加高斯噪声并裁剪。
    *   **Gradient DP**: 对模型更新梯度进行差分隐私处理。
    *   **Secure Aggregation**: 模拟安全聚合协议。

### Step 3: 风格-内容解耦异常检测 (SCD Detection)
> **代码位置**: [`src/optc_uras/models/detector.py`](src/optc_uras/models/detector.py)

在良性数据上训练的异常检测器：
*   **SCD (Style-Content Disentanglement)**: 将 URAS 表征解耦为“风格” (Style) 和“内容” (Content)。
*   **异常评分**: 基于 Style 向量的马氏距离计算异常分数。
*   **ATC (Adaptive Threshold Correction)**: 根据样本置信度动态调整阈值。

---

## 📂 项目结构 (Project Structure)

符合顶级会议（CVPR/ICLR/CCS）标准的工程结构：

```text
optc-project/
├── configs/            # 实验配置文件 (YAML)
│   └── final_production.yaml  # 生产环境配置 (5min window)
├── src/                # 核心源码包 (optc_uras)
│   └── optc_uras/
│       ├── models/     # 模型定义
│       │   ├── step1.py       # [核心] Step 1 语义融合模型
│       │   ├── detector.py    # Step 3 SCD 检测器
│       │   └── ...
│       ├── features/   # 特征工程 (New: SemanticEncoder, SlotAggregator)
│       ├── federated/  # 联邦学习
│       ├── data/       # 数据加载与处理 (New: Rich Semantic Extraction)
│       └── utils/      # 通用工具
├── main.py             # [主入口] 全流程训练与推理脚本
├── preprocess.py       # [工具] 数据预处理脚本
└── README.md           # 项目文档
```

---

## 🚀 快速开始 (Quick Start)

### 从零开始运行（完整步骤）

数据与路径约定（在 `configs/final_production.yaml` 中已配置）：

| 用途 | 路径 | 说明 |
|------|------|------|
| 训练集 | `data/raw/train.jsonl` | 支持 `.json` / `.jsonl` / `.json.gz` |
| 验证集 | `data/raw/val.jsonl` | 可选；可与训练集同文件，按时间划分 |
| 测试集 | `data/test/AIA-51-75.ecar-last.json` | 按需修改为你的测试文件路径 |
| 预处理缓存 | `data/cache/` | 自动生成，无需手动创建 |
| 输出结果 | `experiments/result1/` | 模型与检测结果 |

若训练集文件名或路径不同，请修改配置文件中的 `data.optc.train_path`（及可选的 `val_path`、`test_path`）。

**按顺序执行：**

```bash
# 1. 安装依赖（在项目根目录下执行）
pip install -r requirements.txt

# 2. 数据预处理：将原始日志转为 5min 窗口切片并写入 data/cache/
python preprocess.py --config configs/final_production.yaml

python scripts/make_fast_cache.py --config configs/final_production.yaml --num_workers 0

# 3. 全流程：Teacher 预训练 → Student 联邦训练 → Detector 训练 → 测试
python main.py all --config configs/final_production.yaml

# 4. 评估检测结果（可选）
python evaluate.py experiments/result1/detection_results.csv
```

如需分阶段运行，可使用：`python main.py train_teacher --config ...`、`train_student`、`train_detector`、`test`。

---

### 1. 环境准备

```bash
# 建议使用 Python 3.8+
pip install -r requirements.txt
```

### 2. 数据预处理

将原始 JSONL 日志转换为语义丰富的 5min 窗口切片数据。

```bash
# 读取 configs/final_production.yaml 中配置的路径
python preprocess.py --config configs/final_production.yaml
```

### 3. 运行全流程 (Pipeline)

```bash
# 一键顺序执行所有阶段 (Teacher -> Student -> Detector -> Test)
python main.py all --config configs/final_production.yaml
```

*   **输出**: 结果保存在 `experiments/result1/` 目录下。

### 4. 评估指标 (Evaluation)

```bash
python evaluate.py experiments/result1/detection_results.csv
```

---

## ⚙️ 配置说明 (Configuration)

所有实验参数均在 `configs/final_production.yaml` 中集中管理。关键参数说明：

### 基础配置
*   **`data.optc.window_minutes`**: **5** (推荐，原为 15)。
*   **`model.slot_seconds`**: **30** (推荐，原为 60)。
*   **`model.target_dim`**: Step 1 输出维度。

### 核心算法配置 (Advanced)
*   **Step 1A (Semantic)**:
    *   `model.quality_injection_lambda`: 残差质量注入强度 (0.0-1.0)。
*   **Detector (Step 3)**:
    *   `model.scd_loss`: SCD 损失权重。
    *   `model.atc`: 自适应阈值系数。

---

## 📜 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。
