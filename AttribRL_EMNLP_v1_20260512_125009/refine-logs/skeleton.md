# Skeleton — AttribRL

## State A（reviewer 当前认知）

EMNLP/NLP 社区当前默认：
- A1: tool-use RL 用 outcome reward + GRPO 即可教会 LLM agent 使用工具（Search-R1, R1-Searcher, ReTool, Tool Zero 的 paradigm）
- A2: 若 outcome 颗粒太粗，可加 step-level / invocation correctness PRM（TRM, StepTool, AgentPRM）
- A3: discovery-utilization gap 是 LLM 的 *诊断现象*（Engländer 2026 anchor），与训练侧目标无关
- A4: counterfactual reward 已在 RLHF/RAG faithfulness 中被验证（CST, CoRM-RAG, Causal Rewards），但只用于静态 response 或 retrieved passage 维度
- A5: attention attribution 不可靠（NeurIPS 2025 Attention Bias Optimization），不能作为 utilization 证据

## State B（reviewer 阅读后必须的认知）

读完 AttribRL paper，reviewer 必须相信：
- B1: outcome reward 与 *observation→action utilization* 在统计上系统性解耦——97.54% vs 0.53% gap 不是个例
- B2: 存在一个**self-supervised、可训练、抗 hacking** 的 utilization reward signal——counterfactual replacement KL + lexical span overlap 的双源组合
- B3: 该 reward 与 outcome reward 通过 λ schedule 混合后，Qwen2.5-7B 在 5 benchmark 上 conversion rate 从 ~18% 提升到 ~67%，final task success 同时 +X pp，**且无 verbatim copy hacking**
- B4: 该 reward 与 CST/AgenTracer/TRM/ToolRL 有清晰、可证伪的差异化——object of optimization、causal vs invocation、training vs diagnosis 三轴分离
- B5: 该方法在 ≥2 model size (Qwen-7B, Qwen-14B)、≥3 seeds 下稳健；reproducibility checklist 全填；fallback 是 DiscUseBench diagnostic paper

## Skeleton Path（5 个非可跳过步骤）

**Step 1**: 用 14-model × 5-benchmark *DiscUseBench* 把 anchor 现象规模化——这是 motivation 的硬证据基础。若 reader 不接受这个事实，所有 method 设计无意义。

**Step 2**: 定义 utilization 三元 (Discovery / Use / Conversion) 的 *causal* 而非 LLM-judge 版本——用 counterfactual observation replacement。这一步是后续 reward design 的概念跳板，缺失则 reward 设计无依据。

**Step 3**: 提出 AttribRL 的 *双源* 设计——counterfactual KL（causal）+ lexical span overlap（cheap）+ λ curriculum 混合。强调"双源是 verbatim-copy hacking 的结构性防御"，不是 A+B 拼装。

**Step 4**: 形式化 reward 的 *theoretical guarantees*——(a) 在 Pearl backdoor 假设下，counterfactual replacement KL 是 ATE 的 sound proxy；(b) lexical hack 的 *negative control*（irrelevant-observation injection）下，pure-lexical agent 必失败；(c) λ schedule 的 multi-objective Pareto 解释。

**Step 5**: 实证: ≥2 model size × ≥3 seeds × 5 benchmark 的 utilization × outcome 双指标提升；ablation 完整（4 个 reward variant + 3 个 schedule）；reward hacking audit（verbatim copy ratio < 5%, delayed-utilization captured）；fallback 是 DiscUseBench 自身作为 Findings paper。
