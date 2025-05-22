# GT-MoE
博弈论课程设计：基于拍卖机制的MoE

## 代码结构
```
gt_moe_auction/
├── models/
│   ├── experts.py           # 定义专家模块 (例如 FFN)
│   ├── router.py            # 定义拍卖路由器模块 (AuctionRouter)
│   ├── moe_layer.py         # 定义MoE层 (组合了路由器和专家)
│   └── main_model.py        # 示例：使用MoE层(或多个MoE层)的Transformer模型
├── utils/
│   ├── distributed.py       # (待实现) 分布式训练相关的工具 (专家并行)
│   ├── metrics.py           # (待实现) 负载均衡指标计算 (CV, Gini 指数等)
│   └── logging.py           # (待实现) 日志记录工具
├── configs/
│   └── model_config.yaml    # (示例) 模型超参数、lambda_load等配置
├── data/
│   └── ...                  # (你的数据集和预处理脚本)
├── train.py                 # (概念) 主训练脚本
├── evaluate.py              # (概念) 评估脚本
└── README.md
```

## 基于动态拍卖的路由 (Dynamic Auction-Based Routing)

- 概念: 将输入令牌（token）视为“竞标者”，专家网络视为“卖方”，而MoE的路由机制（router）则扮演“拍卖师”的角色 。
- 机制: 令牌提交“出价”（bid），该出价可以反映其处理的紧迫性、数据类型或从特定专家处获得的预期价值。专家可以战略性地宣布其当前的可用容量、处理成本或期望的价格 。路由机制根据拍卖规则（例如，Vickrey-Clarke-Groves (VCG) 拍卖）分配令牌，以最大化社会福利（例如，所有参与者效用之和）或整体吞吐量，同时确保公平的价格和负载均衡 。
- 预期影响: 改进专家利用率和负载均衡；降低延迟；更好的专家特化；在需求超过供应时，以更具原则性的方式（例如，最高出价者）决定哪些令牌得到处理，而不是随机丢弃或简单的先进先出。

## 数据集
gsarti/clean_mc4_it
## 性能指标:
- 困惑度 (Perplexity, PPL)：语言建模任务的标准指标 。

- 任务特定准确率：在与LLM相关的下游任务上进行评估，例如GLUE基准测试中的选定任务（如MNLI, QQP, SST-2）或SQuAD问答任务 。第一份文件还提及了F1分数、Matthews相关系数 (MCC)、Pearson/Spearman相关系数、BLEU、ROUGE和METEOR等用于通用NLP任务的指标 。


## 效率指标:
- 训练吞吐量：训练期间每秒处理的令牌数 。
- 推理延迟：为给定输入长度生成输出所需的时间（ms/token或ms/sample） 。
- 每令牌FLOPs (针对选定专家)：用于衡量计算成本 。


## MoE特定指标:
- 专家负载分布：使用诸如每个专家处理的令牌数的变异系数 (CV) 或专家分配分布的熵等指标 。
- 专家利用率：一段时间内被积极使用的专家的百分比 。
- 令牌丢失率：如果适用（某些MoE系统在专家超负荷时会丢弃令牌） 。
- 专家激活稀疏度：每个令牌/层激活的专家数量 。
- 路由器Logits分析 。
