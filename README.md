# GT-MoE: 激励相容的混合专家路由机制设计

[![Open Source License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/kaiqiancui/GT-MoE.svg?style=social)](https://github.com/kaiqiancui/GT-MoE)

本项目为博弈论课程期末大作业

**项目作者**: [崔凯乾](https://github.com/kaiqiancui), [杜健](https://github.com/dj2717166816)

---

## 摘要

本项目旨在探索并解决大语言模型中**混合专家（Mixture of Experts, MoE）**架构的核心挑战。尽管MoE已成为突破“规模法则”计算瓶颈的关键范式，其路由机制普遍面临的**负载失衡**与**训练不稳定**问题，仍限制了模型潜能的完全发挥。

我们创新地提出，MoE路由的本质是一个**具有信息不对称的动态委托代理博弈 (Dynamic Principal-Agent Game)**。现有挑战的根源在于，路由策略未能有效解决博弈中的**逆向选择 (Adverse Selection)** 与**道德风险 (Moral Hazard)** 问题。

为此，我们设计并实现了一种新颖的混合式专家路由机制——**RD-ESI (Reputation-driven Differentiable-heuristic Search for Experts with Incentive)**。RD-ESI 作为一套**动态激励合约**，将完美贝叶斯均衡（PBE）理论的序贯理性与贝叶斯更新思想，转化为一个可计算、可优化的工程实现，旨在引导系统走向一个兼顾性能与效率的博弈均衡。

## 目录
- [研究动机：为何 MoE 路由是一个难题？](#研究动机为何-moe-路由是一个难题)
- [我们的视角：将路由重构为动态博弈](#我们的视角将路由重构为动态博弈)
- [RD-ESI 机制：理论的实践方案](#rd-esi-机制理论的实践方案)
- [代码架构](#代码架构)
- [核心实现：梯度与启发式的协同](#核心实现梯度与启发式的协同)
- [快速开始](#快速开始)
- [引用](#引用)

## 研究动机：为何 MoE 路由是一个难题？

1.  **从“大力出奇迹”到“分而治之”**
    -   传统的“稠密”大模型遵循规模法则，参数量越大，性能越强。但这导致了天文数字般的训练与推理成本。
    -   MoE 架构如同一个“专家委员会”，它通过一个“分诊台”（路由器）仅激活部分“专科医生”（专家网络）来处理任务，实现了在巨大模型容量下的计算成本节约。

2.  **路由器的困境：充满不确定性的决策**
    表面上，路由是一个简单的优化问题，但其决策环境充满了深度的不确定性与激励冲突：
    -   **严重的信息不对称**:
        -   **逆向选择**: 分配任务**前**，路由器无法确切知道哪个专家对当前任务的**真实能力**最强。
        -   **道德风险**: 分配任务**后**，路由器无法完美监督专家为完成任务所付出的**真实努力**。
    -   **普遍的激励失衡**:
        -   **负载失衡**: 缺乏有效激励，导致少数“明星专家”被过度使用，大量专家被闲置，如同“公地悲剧”。
        -   **马太效应**: 强者愈强、弱者愈弱的循环最终导致模型“表征崩溃”，限制了整体性能。

## 我们的视角：将路由重构为动态博弈

为了系统性地解决上述挑战，我们转变了研究思路：**不再是设计一个“路由器”，而是设计一套引导所有参与者高效协作的“游戏规则”**。

我们将 MoE 路由问题建模为一个**动态委托代理博弈**：
-   **委托人 (Principal)**: 路由器，目标是最大化系统长期总体性能。
-   **代理人 (Agents)**: 专家网络，其行为受路由机制的激励所引导。

在此框架下，我们追求的理想状态是**完美贝叶斯均衡 (Perfect Bayesian Equilibrium, PBE)**。在 PBE 状态下，路由器能基于对专家的**理性信念**（如声誉）做出**最优选择**，并且该信念能根据专家的历史表现，通过**贝叶斯法则**进行动态更新。

## RD-ESI 机制：理论的实践方案

直接求解 PBE 在计算上是不可行的。因此，我们设计了 **RD-ESI** 机制，作为 PBE 理论的一个工程近似实现。RD-ESI 的核心是一份动态激励合约，即专家选择分数 `SelectionScore`：
![RD-ESI 激励合约公式](assets/math.jpg)
<!-- $$\text{Score}_i(x,t) = \underbrace{g_i(x; w)}_{\substack{\text{解决} \\ \text{逆向选择}}} + \underbrace{\beta R_i(t)}_{\substack{\text{缓解} \\ \text{道德风险}}} - \underbrace{\gamma L_i(t)}_{\substack{\text{内部化} \\ \text{负外部性}}} + \underbrace{\text{Bonus}_i}_{\substack{\text{平衡} \\ \text{探索-利用}}}$$ -->

合约的每个组成部分都旨在应对一个具体的博弈挑战：
-   **基础门控分数 $g_i(x; w)$**: 一个可微分的神经网络，通过任务 $x$ 与专家 $i$ 的特征匹配，直接评估其**潜在能力**，以解决**逆向选择**。
-   **声誉奖励 $\beta R_i(t)$**: 对专家历史表现的奖励。由于好表现通常源于高努力，这激励了专家付出**真实努力**，缓解**道德风险**。声誉的更新模拟了**贝叶斯信念更新**：
    $R_i(t) = (1-\alpha) \cdot R_i(t-1) + \alpha \cdot \text{perf}_i(t-1)$
-   **负载惩罚 $\gamma L_i(t)$**: 对导致系统拥堵的决策施加惩罚，将个体选择对全局造成的**负外部性内部化**。
-   **探索奖励 $\text{Bonus}_i$**: 为选择次数较少的专家提供额外奖励，确保系统能持续探索，以获得对所有专家更准确的长期信念，平衡**探索-利用**。

此外，我们引入了标准的**辅助损失 $L_{\text{aux}}$**。从博弈论视角看，它扮演了**元激励约束**的角色，通过施加独立的梯度信号，迫使基础门控网络 $g_i(x;w)$ 自身学习符合系统全局利益（如负载均衡）的路由倾向。

## 代码架构

本项目采用模块化的代码结构，易于理解和扩展。
```
GT-MoE/
├── src/
│   ├── models/
│   │   ├── moe_layer.py       # CustomMoELayer, 集成路由器和专家网络
│   │   └── moe_transformer.py # 实验用的 Transformer 模型架构
│   │
│   ├── rd_esi/
│   │   ├── router.py          # RDESIRouter, 核心路由逻辑和动态激励合约
│   │   ├── reputation.py      # ReputationScorer, 声誉计算模块
│   │   ├── load_manager.py    # LoadManager, 负载追踪模块
│   │   └── mitigation.py      # 探索奖励等策略模块
│   │
│   ├── configs/               # 存放实验配置文件 (如 .yaml)
│   ├── data/                  # 数据集加载与预处理脚本
│   └── trainer.py             # 核心训练与评估流程
│
├── main.py                    # 启动实验的主入口脚本
├── requirements.txt           # 项目依赖
└── README.md                  # 本文档
```

## 核心实现：梯度与启发式的协同

RD-ESI 机制的一个关键特性是**双系统更新机制**，它结合了梯度优化与启发式规则。

1.  **可微分组件 (梯度驱动)**:
    -   总损失由主任务损失和 MoE 辅助损失构成：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main\_task}} + \lambda_{\text{aux}} \cdot \mathcal{L}_{\text{aux}}$。
    -   梯度通过 $\mathcal{L}_{\text{total}}$ 反向传播，同时优化模型参数以完成主任务，并优化路由器的 `gate_projector` ($g_i(x;w)$) 以学习负载均衡。

2.  **启发式状态 (无梯度更新)**:
    -   声誉 $R_i(t)$、负载 $L_i(t)$ 等状态被注册为 PyTorch 的 `buffer`，它们在训练过程中保持状态，但不参与梯度计算。
    -   在每个训练步的 `forward` 传播之后，我们会调用一个在 `@torch.no_grad()` 上下文中运行的 `update_states()` 方法，根据当前批次的结果，通过算法逻辑更新这些启发式状态。

这种设计实现了**梯度优化驱动的长期稳健学习**与**启发式规则驱动的快速动态适应**的有机结合。

## 快速开始

1.  **克隆仓库**
    ```bash
    git clone https://github.com/kaiqiancui/GT-MoE.git
    cd GT-MoE
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

3.  **准备数据集**
    请根据 `src/data/` 中的说明下载并预处理 C4 数据集。

4.  **开始训练**
    通过指定配置文件来启动训练任务。
    ```bash
    python main.py --config src/configs/rd_esi_default.yaml
    ```
    您可以在 `src/configs/` 目录下创建自己的配置文件，以调整模型架构、超参数和 RD-ESI 机制的各项权重。

## 引用

如果您在研究中发现本项目对您有帮助，请考虑引用：
```bibtex
@misc{cui2025gtmoe,
  author       = {崔凯乾 and 杜健},
  title        = {GT-MoE: 激励相容的混合专家路由机制设计},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/kaiqiancui/GT-MoE](https://github.com/kaiqiancui/GT-MoE)}}
}
