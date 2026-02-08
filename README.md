# OpTC URAS (skeleton repo)

这是一个**仓库骨架**，把「模型方法.docx」里的三步式框架落成可扩展代码结构：

- **Step1**：同源多子视图表征提取与融合（质量权重 + 路由对齐 + 门控融合）
- **Step2**：隐私感知联邦表征预训练（URAS + ASD + AT-InfoNCE + DP + Secure Aggregation）
- **Step3**：benign-only 异常检测（SCD + Mahalanobis + 分位数阈值 + ATC + 可选解释）

> 重要口径：最小样本粒度是 **host × 15min**；slots 仅用于内部表征与质量统计，不对外输出更细粒度定位。

## 快速开始（骨架可跑通 toy 示例）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 生成 toy processed 数据与 splits（不依赖真实 OpTC 原始日志）
python scripts/preprocess.py --config configs/default.yaml --toy

# 2) 运行“联邦预训练”的单机模拟（toy）
python scripts/train_pretrain.py --config configs/default.yaml

# 3) 拟合检测器 + 推理
python scripts/fit_detector.py --config configs/default.yaml
python scripts/infer.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```

## 真实数据接入你需要做什么？

1. 在 `src/optc_uras/data/raw_reader.py` 中实现 `RawReader`（把你们的 PIDSMaker 预处理输出读成统一事件结构）。
2. 在 `scripts/preprocess.py` 里选择/注册你的 reader，并输出到 `processed/`（建议 npz 或 pt）。
3. 保持 Step1/Step2/Step3 的**接口契约**不变，你可以自由替换内部实现。

## 目录结构（约定）

- `src/optc_uras/features/`：确定性/不可学习的特征与统计（slots、聚合器、质量指标、固定随机投影）
- `src/optc_uras/models/`：可学习主干（路由器、对齐基、门控交互、Teacher/Student、Losses）
- `src/optc_uras/federated/`：FedAvg + DP +（可替换的）SecureAgg
- `src/optc_uras/detection/`：SCD、分布拟合、阈值与 ATC、漂移、解释
- `scripts/`：四个可复现实验入口（preprocess / train_pretrain / fit_detector / infer+evaluate）
- `configs/`：字段树对应的 YAML（默认 + 消融覆盖）

## 许可证

仅骨架示例；你可以按项目需要替换为内部许可证。
