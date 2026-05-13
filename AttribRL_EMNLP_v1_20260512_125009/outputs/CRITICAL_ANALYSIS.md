# Landscape 批判清单 — AgentRead

**Date**: 2026-05-12
**Direction**: AgentRead: Diagnosing and Closing the Discovery-to-Utilization Gap in Tool-Augmented LLMs
**Model**: gpt-5.5 (xhigh reasoning), via Codex MCP
**Thread ID**: 019e17c2-d657-7c51-9a97-a817899d3399
**Anchor**: /Users/zpy/LLM_project/idea_paper/boardSearch/outputs/archive_2026-05-11_agentread-design_done/DESIGN_DOC_AgentRead.md

---

## Critique Manifest

1. **CRITIQUE-01** — *Unverified Assumption*
   - **Description**: 领域默认 "tool returned the needed answer" 等同于 "agent had usable evidence"。但 discovery 可能只是答案字符串出现在 observation 中，不代表它在模型可解析、可信、可定位的形式中出现。若不区分 `answer-present`、`evidence-readable`、`evidence-actionable`，Discovery→Use gap 可能被错误归因给模型忽略工具，而不是工具输出结构本身。
   - **Affected**: P01 (Anchor)、TRACE、SciCrafter、DORA Explorer；所有使用 discovery/use 诊断但未建模 observation 可用性的工作。
   - **Why Exploitable**: EMNLP diagnostic + metric paper — 重新定义 Discovery 层级，证明现有 discovery rate 高估了真正可用证据暴露率。

2. **CRITIQUE-02** — *Unverified Assumption*
   - **Description**: Tool-use RL 默认 outcome reward 足以间接鼓励读取/利用 tool output。从未被直接检验：模型可能通过 prior / shortcut / format imitation / reward hacking 得到正确答案，而非因 observation 改变后续 action。
   - **Affected**: ToolRL, Search-R1, R1-Searcher/++, ReTool, Tool Zero。
   - **Why Exploitable**: EMNLP training-analysis paper — 证明 outcome improvement 与 observation-utilization improvement 解耦，暴露当前 tool-use RL 的核心盲区。

3. **CRITIQUE-03** — *Unverified Assumption*
   - **Description**: Step-level/process reward 工作默认 "good step for task completion" 近似等价于 "step is grounded in previous observation"。但 invocation correctness / progress estimate / trajectory utility 都可以不依赖 observation。一个 step 可能格式正确、方向正确，却完全没用刚返回的 evidence。
   - **Affected**: TRM, StepTool, AgentPRM, iTool, RLTR, Auto-Process-Supervision, TRACE。
   - **Why Exploitable**: EMNLP process-supervision critique paper — 把 step quality 拆成 `progress` 与 `attribution-to-observation` 两个正交轴，重测已有 PRM 结论。

4. **CRITIQUE-04** — *Unverified Assumption*
   - **Description**: 现有诊断默认 Discovery→Use gap 是统一机制（attention sink / overconfidence / SFT artifact）导致。但不同任务可能对应完全不同机制：search QA 是 evidence selection，code tool 是 state update，math tool 是 numerical trust，MCP retrieval 是 tool-choice-to-use cascade。
   - **Affected**: P01, SciCrafter, TRACE, DORA Explorer；以及把 gap 当作单一 headline metric 的诊断工作。
   - **Why Exploitable**: EMNLP mechanistic diagnosis paper — 按 task-tool-observation 类型分解 failure mode。

5. **CRITIQUE-05** — *Incorrect Generalization*
   - **Description**: RLHF/RM 中的 counterfactual reward 原用于静态 response 或 pairwise preference。被直接推广到 agent trajectory 会失效——trajectory 中 observation/action/tool call/state transition 有时序依赖和 policy-induced confounding。
   - **Affected**: "Beyond Reward Hacking: Causal Rewards" (P23), Counterfactual RM Multimodal (P24)；以及将 causal reward 直接视为可迁移到 tool-agent RL 的论证。
   - **Why Exploitable**: EMNLP methodological paper — 指出静态 counterfactual RM 不能直接定义 trajectory utilization。

6. **CRITIQUE-06** — *Incorrect Generalization*
   - **Description**: RAG faithfulness 的 perturbation/counterfactual evaluation 原始对象是 retrieved passages 与 final answer 的关系。Tool-agent 轨迹不是单步 retrieval——tool output 会影响后续 search query / tool choice / intermediate reasoning / termination。RAG 的 passage-deletion 直接迁移会漏掉 delayed 和 multi-hop utilization。
   - **Affected**: Sufficient Context, FaithfulRAG, CoRM-RAG, FaithEval, Mindful-RAG。
   - **Why Exploitable**: EMNLP evaluation paper — 证明 RAG-style attribution 在 agent setting 下系统性低估或误判 use，需要 trajectory-aware counterfactuals。

7. **CRITIQUE-07** — *Incorrect Generalization*
   - **Description**: Process reward model 原本适合有明确 intermediate labels 或可验证 progress 的任务。泛化到 open-ended tool agents 时，progress signal 由 rollout outcome 反推，无法保证 step 与 observation 存在因果关系。
   - **Affected**: TRM, StepTool, AgentPRM, iTool, RLTR, Auto-Process-Supervision。
   - **Why Exploitable**: EMNLP reward-model audit paper — 重评 PRM 在 tool agents 中学到的是 task prior、format prior，还是 observation grounding。

8. **CRITIQUE-08** — *Experimental Flaw*
   - **Description**: 多数实验只报 final success/accuracy，不报 Discovery/Use/Conversion 联合分布。会混三种系统：不会发现、发现但不用、使用但推理失败。Outcome RL 的 gain 可能来自更频繁搜索而非更好利用 search 结果。
   - **Affected**: ToolRL, Search-R1, R1-Searcher/++, ReTool, Tool Zero, MCP-Zero, RAG-MCP。
   - **Why Exploitable**: EMNLP benchmark paper — DiscUseBench 式协议重切已发表 gain，揭示 success metric 掩盖的 failure composition。

9. **CRITIQUE-09** — *Experimental Flaw*
   - **Description**: LLM judge 被用来判定 "use"，但 judge 常把 lexical overlap / answer agreement / plausible rationale 误判为 evidence utilization。无 counterfactual removal / evidence swapping / adversarial observation，judge 无法区分 "真的用了 observation" 与 "事后解释看起来像用了"。
   - **Affected**: P01 (LLM-judge use definition), TRACE, SciCrafter；RAG faithfulness 中依赖 judge 的工作。
   - **Why Exploitable**: EMNLP evaluation-reliability paper — 系统审计 LLM judge 的 use 判定，提出 causal use protocol。

10. **CRITIQUE-10** — *Experimental Flaw*
    - **Description**: 现有 tool-agent benchmarks 没有控制 parametric knowledge leakage。若模型本来知道答案，工具 observation 的存在与否不会改变 final answer，仍可能被算作 "successful tool use"。同时低估 utilization gap 与高估 tool-augmented performance。
    - **Affected**: Search-R1, R1-Searcher/++, ToolRL, ReTool；P01 未显式隔离 memorized-answer cases。
    - **Why Exploitable**: EMNLP controlled-benchmark paper — counterfactual answer swaps / fresh facts / synthetic hidden-state tasks 隔离 true observation dependence。

11. **CRITIQUE-11** — *Cross-Domain Misfit*
    - **Description**: Tool retrieval 从 IR 借用 Recall@k；但 tool agents 真正瓶颈可能是选择、调用、读取、整合和后续行动。高 Recall@k 不保证 high downstream use。
    - **Affected**: MCP-Zero, RAG-MCP；所有只用 Recall@k 评价 tool/MCP retrieval 的 claims。
    - **Why Exploitable**: EMNLP agent-evaluation paper — 证明 retrieval-side metrics 与 downstream utilization weakly correlated。

12. **CRITIQUE-12** — *Cross-Domain Misfit*
    - **Description**: Attention attribution 从 interpretability 迁入 agent diagnosis 时默认 saliency/attention 可解释 "模型用了什么"。但 Attention Bias Optimization 已显示 saliency 过度归权无关 context；long trajectory 中 attention 还受 recency / format / system prompt / tool schema 干扰。
    - **Affected**: Attention Bias Optimization (P27), LoGra (P28)；任何把 saliency/attention 当作 use evidence 的 agent analysis。
    - **Why Exploitable**: EMNLP interpretability-for-agents paper — 检验 attention/influence attribution 与 counterfactual observation dependence 的一致性边界。

13. **CRITIQUE-13** — *Cross-Domain Misfit*
    - **Description**: RAG faithfulness 默认 source documents 是静态、可替换、可删除的 evidence units；tool observations 则常是 stateful / procedural / interactive 的中间状态。删除一个 tool output 可能改变后续 tool calls，而非简单影响 final answer。
    - **Affected**: Sufficient Context, FaithfulRAG, CoRM-RAG, Mindful-RAG。
    - **Why Exploitable**: EMNLP cross-domain adaptation paper — 定义 agent-specific faithfulness failure。

14. **CRITIQUE-14** — *Experimental Flaw*
    - **Description**: 混合 reward 的 λ schedule 基本未被系统研究。λ 太低，模型继续忽略 observation；λ 太高，模型可能 copy tool output / 牺牲 reasoning / 产生 attribution gaming。没有 reward trade-off 曲线，任何 "utilization improves performance" 都可能只是特定 λ 的偶然结果。
    - **Affected**: 全线 tool-use RL；step-level reward 工作；任何未来混合 outcome+process/utilization reward 的方法 claim。
    - **Why Exploitable**: EMNLP training-dynamics paper — 系统刻画 outcome / utilization / lexical attribution reward 之间的 Pareto frontier 和 failure regime。
