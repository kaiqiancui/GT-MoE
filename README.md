# GT-MoE: 基于声誉的专家选择与自适应激励（RD-ESI）MoE路由机制

## 1. 项目简介

本项目旨在设计、实现并通过从头训练一个定制的小型混合专家（MoE）Transformer模型来验证“**基于声誉的动态专家选择与自适应激励（Reputation-based Dynamic Expert Selection with Adaptive Incentives, RD-ESI）**”新型路由机制的有效性，具体的算法信息参考 `map.md` 文件。

与当前主流MoE路由机制（如Top-K门控）相比，RD-ESI引入了动态声誉评分 $R_i(t)$、负载感知 $L_i(t)$ 以及探索奖励和声誉衰减等自适应激励策略，以期优化专家选择过程，改善负载均衡，促进专家特化，并缓解马太效应。

核心目标：通过在 **WikiText dataset** 上进行实验，主要验证RD-ESI博弈论算法的性能和核心机制，而非构建一个领先水平的大型模型。

详细的算法设计、公式、实现方案、训练策略和评估指标请参考项目研究方案文档。

## 2. 项目文件结构

以下是本项目的文件结构：

```
.
├── README.md                   # 项目介绍和指南 (本文档)
├── map.md                      # 详细的算法设计与实现方案
├── requirements.txt            # Python 依赖包列表
├── configs/                    # 存放实验配置文件
│   ├── rd_esi_custom_model_config.yaml
│   ├── top_k_custom_model_config.yaml
│   └── expert_choice_custom_model_config.yaml
├── src/                        # 主要源代码目录
│   ├── __init__.py
│   ├── main.py                 # 项目主入口脚本
│   ├── rd_esi/                 # RD-ESI 机制的核心实现
│   │   ├── __init__.py
│   │   ├── router.py           # RDESIRouter 类，整合 RD-ESI 逻辑
│   │   ├── reputation_scorer.py # 动态声誉评分 R_i(t) 的计算与更新
│   │   ├── load_manager.py     # 专家负载 L_i(t) 的计算与追踪
│   │   ├── mitigation_strategies.py # 探索奖励和声誉衰减等缓解马太效应的策略
│   │   └── utils.py            # RD-ESI 相关的辅助函数
│   ├── models/                 # 模型定义与实现
│   │   ├── __init__.py
│   │   ├── custom_moe_layer.py # 自定义MoE层，集成RD-ESI路由器和专家网络
│   │   └── custom_transformer_moe.py # 自定义小型MoE Transformer模型架构
│   ├── data_utils/             # 数据加载与预处理工具
│   │   ├── __init__.py
│   │   ├── wikitext_loader.py # 加载和处理 WikiText dataset
│   │   └── tokenizer_utils.py  # 分词器相关工具
│   ├── training/               # 训练相关的脚本和工具
│   │   ├── __init__.py
│   │   ├── trainer.py          # 自定义训练流程
│   │   └── training_utils.py   # 学习率调度、混合精度、梯度累积等辅助函数
│   ├── evaluation/             # 评估相关的脚本和工具
│   │   ├── __init__.py
│   │   ├── metrics.py          # 实现 PPL 及 MoE 特定指标
│   │   └── evaluator.py        # 执行模型评估的脚本
│   └── baselines/              # 基线路由机制的实现
│       ├── __init__.py
│       ├── top_k_gating_custom.py # 标准 Top-K 门控在自定义模型上的实现
│       └── expert_choice_routing_custom.py # 专家选择路由在自定义模型上的实现
└── results/                    # 存储实验结果、日志、图表和训练好的模型权重
    ├── rd_esi_run/
    ├── top_k_baseline_run/
    └── expert_choice_run/
```

## 3. 各代码文件作用详解

### `src/rd_esi/`
* `router.py`: 定义 **`RDESIRouter`** 类，实现 RD-ESI 的核心路由逻辑，计算选择分数 $\text{SelectionScore}_i(x,t) = g_i(x) + \beta \cdot R_i(t) - \gamma \cdot L_i(t) + \text{ExplorationBonus}_i(x,t)$。
* `reputation_scorer.py`: 实现动态声誉评分 $R_i(t)$ 的EMA计算。
* `load_manager.py`: 实现专家负载 $L_i(t)$ 的量化（如队列长度或近期使用频率EMA）。
* `mitigation_strategies.py`: 实现探索奖励和声誉衰减机制。
* `utils.py`: RD-ESI 相关的辅助函数。

### `src/models/`
* `custom_moe_layer.py`: 包含一个 **`RDESIRouter`** 实例和一组专家网络，在其 `forward` 方法中调用 RD-ESI 路由器，分派令牌给选定专家，并组合专家输出。
* `custom_transformer_moe.py`: 定义一个小型 MoE Transformer 模型架构，堆叠标准 Transformer 组件和自定义的 **`CustomMoELayer`**。

### `src/data_utils/`
* `wikitext_loader.py`: 负责加载和预处理 **WikiText dataset**，进行数据清洗、分词和批处理。
* `tokenizer_utils.py`: 管理分词器的加载和使用。

### `src/training/`
* `trainer.py`: 实现标准的从头训练循环，管理模型的训练、验证步骤，优化器和学习率调度，处理RD-ESI状态的更新逻辑。
* `training_utils.py`: 包含训练辅助功能，如学习率预热/衰减策略、混合精度训练的设置、梯度累积的实现等。

### `src/evaluation/`
* `metrics.py`: 定义和计算评估指标，包括困惑度(PPL)和MoE特定指标。
* `evaluator.py`: 在测试集上运行模型评估，调用 `metrics.py` 计算指标。

### `src/baselines/`
* `top_k_gating_custom.py`: 在自定义模型架构上实现标准Top-K门控路由。
* `expert_choice_routing_custom.py`: 在自定义模型架构上实现专家选择路由。

### `src/main.py`
项目主入口，通过配置文件启动训练或评估流程。

## 4. 数据预处理

由于数据集较大，我们提供了一个一次性预处理步骤，将数据集进行分词并保存到磁盘。这样可以显著加快后续的训练运行，因为它避免了每次训练时重新分词数据。

在开始第一次训练之前，请从项目根目录运行以下命令。确保指定您打算用于训练的配置文件。

```bash
python src/preprocess_data.py --config_file configs/rd_esi_small.yaml
```

这个过程可能会花费相当长的时间，具体取决于您的机器的CPU核心数和磁盘速度，但它只需要执行一次。处理后的数据将保存到配置文件中由 `processed_data_path` 指定的目录（默认为 `./processed_data/`）。

## 5. 数据集

本项目使用 **WikiText dataset** 作为训练和评估数据集。WikiText 是一个大规模的高质量语言建模数据集，由Salesforce研究团队创建。

### WikiText 数据集特点
* **来源**: 从维基百科上精选的高质量文章
* **规模**: WikiText-103包含约103M个单词，WikiText-2包含约2M个单词
* **语言**: 英语文本
* **格式**: 已经过清洗和预处理，保留了文章结构和标点符号
* **适用性**: 特别适合语言模型训练，比传统的Penn Treebank数据集更大更多样化

### 数据处理
`src/data_utils/wikitext_loader.py` 实现了 WikiText 数据集的加载和处理逻辑:
1.  使用 HuggingFace Datasets 库加载数据
2.  使用数据集自带的训练集、验证集和测试集划分
3.  对文本进行分词处理
4.  创建适用于语言模型训练的批次数据

## 5. 安装与环境设置

1.  克隆本项目。
2.  创建并激活Python虚拟环境（推荐使用conda）：
    ```bash
    conda create -n gt_moe_env python=3.9
    conda activate gt_moe_env
    ```
3.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

## 6. 运行实验

### 6.1 训练模型

以下是训练不同路由机制模型的命令：

#### RD-ESI 模型训练
```bash
python src/main.py --config_file configs/rd_esi_custom_model_config.yaml --mode train --output_dir results/rd_esi
```

#### Top-K 基线模型训练
```bash
python src/main.py --config_file configs/top_k_custom_model_config.yaml --mode train --output_dir results/top_k
```

#### Expert Choice 基线模型训练
```bash
python src/main.py --config_file configs/expert_choice_custom_model_config.yaml --mode train --output_dir results/expert_choice
```

### 6.2 评估模型

如果您已经训练好模型并想单独评估它们，可以使用以下命令：
```bash
python src/main.py --config_file configs/rd_esi_custom_model_config.yaml --mode eval --output_dir results/rd_esi_eval
```

### 6.3 运行比较实验

比较不同路由机制的性能：
```bash
python src/experiments/run_comparison.py --config_dir configs --output_dir results/comparison --mechanisms rd_esi top_k expert_choice
```

## 7. 配置文件说明

配置文件位于 `configs/` 目录下，包含三个主要配置文件：
* `rd_esi_custom_model_config.yaml`: RD-ESI 路由机制的配置
* `top_k_custom_model_config.yaml`: Top-K 路由机制的配置
* `expert_choice_custom_model_config.yaml`: Expert Choice 路由机制的配置

每个配置文件包含以下主要部分：
* `model`: 模型架构参数（隐藏层大小、层数、头数等）和MoE配置（专家数量、路由参数等）
* `tokenizer`: 分词器配置
* `data`: 数据加载参数（批量大小、序列长度等）
* `trainer`: 训练相关参数（优化器、学习率、训练步数等）
* `evaluator`: 评估相关参数

## 8. 实验结果分析

实验结果将保存在 `results/` 目录下，包括：
* 训练日志
* 评估指标（PPL、专家负载分布等）
* 模型检查点
* 可视化图表

可以使用以下命令生成比较图表：
```bash
python src/experiments/plot_results.py --results_dir results/comparison
```


* 感谢所有为本项目做出贡献的研究者和开发者
* 特别感谢 WikiText 数据集的创建者和维护者