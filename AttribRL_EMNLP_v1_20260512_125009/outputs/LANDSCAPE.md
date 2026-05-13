# 文献全景图：AgentRead — Diagnosing and Closing the Discovery-to-Utilization Gap in Tool-Augmented LLMs

**Date**: 2026-05-11
**Papers analyzed**: 35
**Sources**: web search (arXiv, OpenReview, ACL Anthology), arxiv_fetch.py
**Venue context**: EMNLP（user-specified for screening）
**Anchor design doc**: `/Users/zpy/LLM_project/idea_paper/boardSearch/outputs/archive_2026-05-11_agentread-design_done/DESIGN_DOC_AgentRead.md`

---

## Executive Summary

围绕 "AgentRead：tool-augmented LLM agent 的 discovery–utilization 落差" 主题，过去 12 个月的文献已形成 6 个清晰主题，但**几乎完全错过了 anchor 现象本身**。最贴近的诊断性论文 **Engländer et al. (arXiv:2604.17609) "Agents Explore but Agents Ignore"** 首次在 Terminal-Bench 与 AppWorld 上量化了 "discover 但 ignore" 现象（gpt-oss-120b 97.54% discovery vs 0.53% use），但只覆盖 2 个 benchmark、未做训练侧 fix。

工具使用 RL 方向（**Search-R1 / R1-Searcher / ReTool / ToolRL / Tool Zero**）压倒性地选择 **outcome-only reward 或 invocation-correctness reward**，没有任何一个工作把 reward 信号锚到 *observation→action* 的因果关系上。最贴近的 step-level reward 工作（**StepTool / TRM / AgentPRM / iTool**）依赖人工标注或 rollout 估计的 "progress"，与 utilization 概念互补但不直接解决。

Counterfactual / 因果 reward 方向（**Beyond Reward Hacking (Causal Rewards), Counterfactual Reward Model**）在 RLHF 偏好对齐场景中已被验证；attention attribution 方向（**Attention Bias Optimization** NeurIPS 2025）则指出现有显著性方法对无关 token 误分配权重——但**这两条线从未被搬到 tool-augmented agent 的 RL 训练循环里**。这正是 AttribRL 的 niche 所在：counterfactual + lexical 双源 attribution reward + GRPO，自监督、无需新标注、直接 reward "用了 observation"。

亮点 gap：(G1) 没有 14-model × 5-bench 的 discovery/use/conversion 三维表（DiscUseBench 唯一），(G2) 没有 RL 方法以 utilization 为优化目标，(G3) counterfactual 信号从未作为训练 reward 喂给 tool agent，(G7) 缺乏 self-supervised 的 observation–action 因果信号。EMNLP 2025 已接收 ≥4 篇 tool-use RL 主题论文，证明此 venue 高度匹配；EMNLP 2026 Findings 也是 Plan B（diagnostic-only paper）的天然落点。

---

## Paper Table

| ID | Paper | Authors | Year | Venue | Method | Key Result | Relevance | Source |
|----|-------|---------|------|-------|--------|------------|-----------|--------|
| P01 | Agents Explore but Agents Ignore: LLMs Lack Environmental Curiosity | Engländer, Althammer, Üstün, Gallé, Sherborne | 2026 | arXiv:2604.17609 (preprint) | Solution Injection + discovery@k / interaction@k | gpt-oss-120b: 97.54% discovery vs 0.53% use; Terminal-Bench 79-81%/37-50%; AppWorld 90%+/<7% | **ANCHOR — 现象证据** | web |
| P02 | Empowering LLM Tool Invocation with Tool-call Reward Model (TRM) | (Anonymous OpenReview) | 2025 | OpenReview LnBEASInVr | Process reward model for tool invocation; PPO/GRPO integration | 3B TRM 10K samples; consistently > outcome-only RL | **直接竞品 — invocation correctness** | web |
| P03 | ToolRL: Reward is All Tool Learning Needs | Qian, Acikgoz et al. | 2025 | NeurIPS 2025 (arXiv:2504.13958) | Principled reward design for tool selection/application | +17% over base, +15% over SFT | 主流 baseline | web |
| P04 | Search-R1: Training LLMs to Reason and Leverage Search Engines with RL | Jin, Zeng et al. | 2025 | arXiv:2503.09516 | Outcome RL + retrieved token masking + multi-turn search | +41% Qwen2.5-7B over RAG baselines | infra/baseline | web |
| P05 | R1-Searcher | Song et al. | 2025 | arXiv:2503.05592 | Two-stage outcome-only RL for search | > GPT-4o-mini baseline RAG | baseline | web |
| P06 | R1-Searcher++ | (extension) | 2025 | arXiv:2505.17005 | Dynamic knowledge acquisition via RL | (improved over R1-Searcher) | baseline | web |
| P07 | StepTool: Step-grained RL Framework for Tool Learning | Yu et al. | 2024 | arXiv:2410.07745 | Step-grained reward shaping + optimization | > SFT and RL baselines on Pass Rate & Recall | step reward, 与 TRM 重叠 | web |
| P08 | ReTool: RL for Strategic Tool Use | Feng, Huang et al. | 2025 | arXiv:2504.11536 | Interleaved code execution + outcome RL | 32B 67% accuracy, 400 steps (vs 1080 for text-only) | baseline | web |
| P09 | iTool: Reinforced Fine-Tuning with Dynamic Deficiency Identification | (EMNLP 2025) | 2025 | EMNLP 2025 Main 701 | Deficiency-aware RL | (improvement over SFT/RL) | EMNLP precedent | web |
| P10 | Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch | (anonymous) | 2025 | Findings EMNLP 2025 (arXiv:2511.01934) | Pure RL cold-start for tool-use | (gains over SFT cold-start) | EMNLP precedent | web |
| P11 | TRACE: Trajectory-Aware Comprehensive Evaluation for Deep Research Agents | Chen, Jiang, Liu, Zhang, Guo, King | 2026 | WWW '26 (arXiv:2602.21230) | Hierarchical Trajectory Utility Function | Process efficiency + cognitive quality + evidence grounding | eval framework | web |
| P12 | DORA Explorer: Improving the Exploration Ability of LLMs Without Training | Gurjar, Ishmam, Marino | 2026 | arXiv:2604.17244 | Diversity-Oriented Ranking; training-free | MAB + TALES; > CoT/ToT exploration | exploration-side | web |
| P13 | Can Current Agents Close the Discovery-to-Application Gap? (SciCrafter) | (anonymous) | 2026 | arXiv:2604.24697 | 4-capacity decomposition (gap-ID / discovery / consolidation / application) | Frontier models plateau ~26%; knowledge application is biggest gap | **Most adjacent prior on "discovery vs application"** | web |
| P14 | AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise & Progress | (anonymous) | 2026 | WWW 2026 (arXiv:2511.08325) | MC-rollout based PRM | 8× compute-efficient | step reward | web |
| P15 | Reinforcement Learning for LLM Agent Planning (RLTR) | (anonymous) | 2025 | EMNLP 2025 Industry 116 | RL with Tool-use Rewards | (improvement) | EMNLP precedent | web |
| P16 | Enhancing LLM Agents with Automated Process Supervision | (anonymous) | 2025 | EMNLP 2025 Main 506 | Auto process supervision | (improvement) | EMNLP precedent | web |
| P17 | Sufficient Context: A New Lens on RAG Systems | Joren, Zhang et al. | 2025 | OpenReview Jjr2Odj8DJ | Sufficient context classifier + selective generation | +2-10% correct fraction | context-side | web |
| P18 | Lost in the Middle: How Language Models Use Long Contexts | Liu et al. | 2023 (foundational) | TACL | U-shape long-context attention curve | 30% degradation when info in middle | foundational | web |
| P19 | FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful RAG | (anonymous) | 2025 | arXiv:2506.08938 | Fact-level conflict modeling | (improved faithfulness) | RAG side | web |
| P20 | CoRM-RAG: Counterfactual Risk Minimization | (anonymous) | 2026 | arXiv:2605.01302 | Cognitive Perturbation Protocol | counterfactual-robust RAG | counterfactual RAG | web |
| P21 | FaithEval | (anonymous) | 2024 | (preprint) | Faithfulness benchmark across context types | SOTA models still fail on counterfactual context | eval | web |
| P22 | Mindful-RAG | (anonymous) | 2025 | (preprint) | Intent & context-aware generation | resilience to perturbation | RAG side | web |
| P23 | Beyond Reward Hacking: Causal Rewards for LLM Alignment | (anonymous) | 2025 | arXiv:2501.09620 | Counterfactual invariance for reward model | mitigates spurious correlations | **causal reward, related concept** | web |
| P24 | Counterfactual Reward Model Training (multimodal RL) | (anonymous) | 2025 | arXiv:2508.19567 | Causal inference + multimodal representation | unsupervised bias-resilient reward | counterfactual reward | web |
| P25 | Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use | (anonymous) | 2026 | arXiv:2605.02964 | Adversarial benchmark | quantifies tool-agent reward hacking | reward hacking | web |
| P26 | Reward Shaping to Mitigate Reward Hacking in RLHF | (anonymous) | 2025 | arXiv:2502.18770 | Shaped reward forms | reduced hacking | reward hacking | web |
| P27 | The Fragile Truth of Saliency: Attention Bias Optimization | (anonymous) | 2025 | NeurIPS 2025 | Bias-injected saliency via attention | causal saliency; existing methods over-attribute irrelevant context | **attention attribution** | web |
| P28 | Large-Scale Training Data Attribution with Efficient Influence Functions (LoGra) | (anonymous) | 2025 | ICLR 2025 / OpenReview jZw0CWXuDc | Gradient projection for IF | scalable TDA | attribution | web |
| P29 | BFCL: Berkeley Function Calling Leaderboard V1-V4 | Patil et al. | 2024-2025 | OpenReview 2GmDdhBdDk | AST + multi-turn + agentic eval | de-facto FC standard; Llama 3.1 405B 0.885 | benchmark | web |
| P30 | ToolBench (ToolLLM) | Qin et al. | 2023-2024 | ICLR 2024 | Multi-step API benchmark | foundational | benchmark | known |
| P31 | API-Bank | Li et al. | 2023 | EMNLP 2023 | Mixed API benchmark | foundational | benchmark | known |
| P32 | Terminal-Bench: Benchmarking Agents on Hard Realistic Tasks | (anonymous) | 2026 | ICLR 2026 (arXiv:2601.11868) | Containerized CLI eval; canary strings | Used in Anchor paper | benchmark | web |
| P33 | AppWorld | Trivedi et al. | 2024 | ACL 2024 | Controllable apps & people | Used in Anchor paper | benchmark | known |
| P34 | MCP-Zero: Active Tool Discovery for Autonomous LLM Agents | (anonymous) | 2025 | arXiv:2506.01056 | Active tool discovery, 4400+ MCP | reduces context overhead | tool selection | web |
| P35 | RAG-MCP: Mitigating Prompt Bloat via RAG for Tool Selection | (anonymous) | 2025 | arXiv:2505.03275 | Retrieval-augmented MCP | scalable discovery | tool selection | web |

---

## Thematic Analysis

### Theme 1: 工具使用 RL — Outcome-only 主导
**Status**: active
**Dominant approach**: outcome reward + GRPO/PPO on multi-turn tool trajectories
**Papers**: P03 (ToolRL), P04 (Search-R1), P05 (R1-Searcher), P06 (R1-Searcher++), P08 (ReTool), P10 (Tool Zero)

主流路径——把 tool calls 当作 token、把任务成功当作 sparse outcome reward 喂 GRPO/PPO。Search-R1 / R1-Searcher 在 retrieval-search 子场景里证明了 outcome-only 路线在 7B 规模就能 +20–41% 超过 RAG 基线；ToolRL 把这条线推广到通用 tool-use 并明确指出 outcome reward 的局限（fine-grained 信号缺失），但其"解决方案"是丰富 reward type/scale/granularity，没有把粒度推到 observation 维度。ReTool 与 Tool Zero 证明 outcome-only RL 在 32B 与 cold-start 设置下都可行。**共识**：outcome reward 简单且 generalize；**未解争议**：所有方法都把 "agent 读没读 tool output" 视为黑盒——只看最终 task 成败。

### Theme 2: Step-level / Process Reward — 弥补 outcome 颗粒
**Status**: active
**Dominant approach**: 训练 process reward model (PRM) 或定义 step-grained shaping 项，与 outcome 一起 GRPO
**Papers**: P02 (TRM), P07 (StepTool), P09 (iTool), P14 (AgentPRM), P15 (RLTR), P16 (EMNLP-Main-506)

主张："coarse outcome reward 会引发 gradient conflict——对的 tool call 因为答错被惩罚"。TRM 训练 3B 的 process reward model 对 invocation correctness 打分；AgentPRM 用 MC rollouts 估计 step 的 promise & progress；StepTool 用规则化的步级 reward shaping。**共识**：步级信号有效。**争议**：是用标注数据训 PRM、还是用 rollout 估 progress、还是用规则化 shaping。**关键空缺**：所有 step-level reward 都 reward "this step is good for finishing the task"（invocation 或 progress），没有任何一个 reward "this action depended on the previous observation"。AttribRL 的 niche 正在此。

### Theme 3: Agent 诊断 / Metacognitive Failure — 新兴
**Status**: emerging
**Dominant approach**: 注入已知 solution / 解构 capacity 维度 / 测量 trajectory 内部 quality
**Papers**: P01 (Agents Explore but Agents Ignore — Anchor), P11 (TRACE), P12 (DORA Explorer), P13 (SciCrafter)

近 6 个月（2026 Q1-Q2）涌现的 4 篇都在敲同一个鼓：current agents 在 trajectory 内部存在 *无法被 outcome metric 捕获* 的失败模式。Anchor (P01) 是最直接的 discovery–utilization gap 证据；SciCrafter 把它泛化为 4-capacity（知识 gap 识别 / 实验性发现 / 巩固 / 应用），实验显示 "knowledge application is biggest gap"；TRACE 提出 evidence grounding 作为 trajectory utility 子项；DORA Explorer 处理 exploration 多样性。**共识**：outcome metric 隐藏退化、需要 trajectory 内部指标。**未解**：(a) 这些诊断是否只是 prompt 没写好？(b) RL 是否可以把这些诊断 metric 转成 training signal？这是 AttribRL 的转化空间。

### Theme 4: Context Utilization & RAG Faithfulness — 工具 agent 之外的兄弟线索
**Status**: mature
**Dominant approach**: faithfulness 评测 / 充分性分类器 / 反事实扰动测试
**Papers**: P17 (Sufficient Context), P18 (Lost in the Middle), P19 (FaithfulRAG), P20 (CoRM-RAG), P21 (FaithEval), P22 (Mindful-RAG)

RAG 社区已经独立地发展出 "context utilization vs context sufficiency" 概念，并用 counterfactual perturbation（CoRM-RAG）和 fact-level conflict（FaithfulRAG）做评测/对齐。**重要诊断**：SOTA 模型在 sufficient context 下高表现，但在 insufficient context 下不会 abstain；RAG 反而增加 overconfidence。**与本工作关系**：RAG faithfulness 关心 "answer 是否 grounded in context"，AgentRead 关心 "action 是否 grounded in observation"——形式同构，目标互通，但 RAG 侧的方法（faithfulness 评测、counterfactual perturbation）从未被借到 tool agent 的 RL 训练循环。**这是清晰的 cross-domain transfer 空间**。

### Theme 5: Counterfactual / Causal Reward 与 Attention Attribution — 远房表亲
**Status**: emerging
**Dominant approach**: 在 reward model 训练中加入 counterfactual 不变性 / 用 attention 偏置实现因果 saliency
**Papers**: P23 (Causal Rewards), P24 (CF RM Multimodal), P25 (Reward Hacking Benchmark), P26 (Reward Shaping), P27 (Attention Bias Optimization), P28 (LoGra Influence Functions)

P23 (Beyond Reward Hacking) 证明 counterfactual invariance 能让 RLHF 奖励模型摆脱 spurious correlation；P24 把同思路搬到 multimodal bias mitigation。P27 (NeurIPS 2025) 用 attention bias 重新计算 saliency 并指出"现有方法对无关 token 错误归权"，与 P01 中的 attention 失锚假设直接呼应。**共识**：counterfactual 是 ground-truth-free 的 robust signal。**未解**：counterfactual 操作在 *trajectory-level RL*（每步一个 observation）能否稳定 train——业界没人做过。AttribRL 的最大不确定性也在此（design doc §12 自陈"要么 first work 要么 first failed work"）。

### Theme 6: Tool Selection / MCP Retrieval — 上游 vs 下游
**Status**: emerging
**Dominant approach**: 把 tool list 当语料、用 retrieval 检索相关 tool
**Papers**: P34 (MCP-Zero), P35 (RAG-MCP), P29 (BFCL 评测 retrieval 子项)

MCP 生态 4400+ servers 推动了 tool-retrieval / tool-to-agent retrieval 研究。这条线 *上游*（先检索出对的 tool）与 AgentRead *下游*（已经读到了 tool output 是否用上）正交但互补。当前几乎所有 tool-retrieval 工作只评 Recall@k，不评下游 utilization。

---

## Gap Identification Matrix

| Gap ID | Gap Description | Evidence (papers) | Gap Type | Confidence |
|--------|----------------|-------------------|----------|------------|
| G1 | 缺乏 14-model × 5-benchmark 规模的 discovery / use / conversion 三维实证表；Anchor paper 只覆盖 2 个 benchmark、不分 model 尺寸/家族 | P01, P13, P11 | missing diagnostic | HIGH |
| G2 | 现有 tool-use RL（outcome / invocation reward）从未把 reward 锚到 *tool output utilization*；TRM/ToolRL/Search-R1 都不奖励"用上了" | P02, P03, P04, P07, P08, P10 | overlooked formulation | HIGH |
| G3 | Counterfactual reward signal 已在 RLHF 偏好对齐（P23/P24）和 RAG faithfulness 评测（P20/P22）证明有效，但从未作为 trajectory-level training reward 喂给 tool agent | P23, P24, P20, P22 | cross-domain transfer | HIGH |
| G4 | Process reward models（AgentPRM/StepTool）依赖标注或 MC rollouts 估 progress；无 *self-supervised observation-action causality* signal | P02, P07, P14 | overlooked formulation | MEDIUM |
| G5 | "Discovery rate / Use rate / Conversion rate" 三元定义只在 P01 出现且只用 LLM judge；缺乏 *causal*（counterfactual）的严格"use"定义 | P01, P11, P13 | missing diagnostic | HIGH |
| G6 | 缺乏对 discovery–utilization gap 的*机制学*分析（是 attention 失锚？overconfidence？SFT artifact？）；P27 指出 attention attribution 对无关 token 误归权但未对接 agent observation | P01, P27, P17 | untested assumption | MEDIUM |
| G7 | Lexical attribution (rouge-L) 便宜但易被 verbatim copy 攻击；counterfactual 严谨但贵——无工作*组合二者*作为防御性 reward 设计 | P02, P03, P23 | resolution opportunity | HIGH |
| G8 | RL reward 中的 λ schedule（动态权重）在 RLHF 已有研究但未应用到 "outcome + utilization" 混合 reward 的 tool-agent 训练 | P03, P26 | untested assumption | LOW |
| G9 | EMNLP/ACL 社区已有 ≥4 篇 tool-use RL paper 但全部聚焦 reward design 或 step-level supervision，从未碰 utilization 这一定义 | P09, P10, P15, P16, P31 | missing diagnostic | MEDIUM |
| G10 | RAG faithfulness 的 counterfactual perturbation 评测（CoRM-RAG / FaithEval）从未被搬到 tool agent trajectory 上 | P20, P21, P22, P01 | cross-domain transfer | HIGH |
| G11 | Tool-retrieval 工作（MCP-Zero / RAG-MCP）只评 Recall@k，没有 downstream utilization 评估；上下游脱节 | P34, P35, P29 | missing diagnostic | MEDIUM |
| G12 | Plan B "diagnostic-only" 方向：14-model × 5-bench DiscUseBench 本身缺位，且 anchor paper 没释 data；EMNLP Findings/Main 都接此类纯诊断 paper | P01, P11, P13 | missing diagnostic | HIGH |

---

## Trajectory Analysis

**Top recurring authors / labs**:

1. **Cheng Qian (UIUC, ToolRL)** — 持续在 agentic RL post-training 方向；W26-W27 内连续放出 3 篇相关 preprint。方向：reward design generalization → cross-task tool agent。
2. **Peter Jin (UIUC, Search-R1)** — Search-R1 codebase 已成为 EMNLP/NeurIPS 2025 多个 tool-use RL 工作的事实基础设施（veRL fork）。方向：tool/search RL infra → 多 modal RL。
3. **Leon Engländer (Cohere/EPFL, Anchor)** — 2026 起进入 agent diagnostics / environmental curiosity 主题；与 Sherborne / Üstün 在 Cohere Labs 内部协作。方向：诊断 → 干预（潜在）。
4. **Irwin King (CUHK, TRACE)** — Deep research agent eval；TRACE 之后预计转向 trajectory utility 训练。

**Coauthor clusters**:

- **Cluster A (Berkeley/UIUC tool-use RL)**: Patil (BFCL) + Jin (Search-R1) + Qian (ToolRL) + Acikgoz — outcome RL + benchmark side。**正在收敛**到 *step-level reward* 主题。
- **Cluster B (Cohere/Edinburgh agent diagnostics)**: Engländer + Althammer + Sherborne — pure diagnostics，目前不做训练。**预计 6-9 个月内会做训练侧 fix**——这是直接对手 timeline。
- **Cluster C (NeurIPS 2025 attribution)**: P27 (Attention Bias Optimization) + P28 (LoGra) — 通用 LLM 解释性。**与 agent 社区零交集**——这是 AgentRead 的桥接价值。
- **Cluster D (China-mainland RAG faithfulness)**: P19 FaithfulRAG, P20 CoRM-RAG, P22 Mindful-RAG — counterfactual perturbation 工具链已成熟，但**未上 RL trajectory**。

**关键趋势**：
- Outcome RL（Cluster A 主流）→ Step-level reward / PRM（已迁移）→ **Utilization reward** （下一站，AttribRL 押注）
- Agent diagnostics（Cluster B）→ Diagnostic benchmarks → **Training-side intervention** （Cluster B 6-9 月内会做，AgentRead 必须抢节奏）

**Implication for AgentRead**：
- 与 Cluster A 形成清晰差异化（utilization 而非 invocation/progress）
- 与 Cluster B 形成 *方法/诊断* 互补（如果 Cluster B 抢先发诊断 paper，AgentRead 仍可用方法贡献区分；如果 Cluster B 抢先发训练 paper，AgentRead 用 counterfactual self-supervision 角度区分）
- 抢 Cluster C 的工具上车 agent 社区（attention attribution + counterfactual）是 **桥接红利**

---

## References

**主参考列表（35 篇）已在 Paper Table 中给出 ID 与 URL/arXiv 编号。** 关键链接：

- P01 Anchor: https://arxiv.org/abs/2604.17609
- P02 TRM: https://openreview.net/forum?id=LnBEASInVr
- P03 ToolRL: https://arxiv.org/abs/2504.13958
- P04 Search-R1: https://arxiv.org/abs/2503.09516
- P05 R1-Searcher: https://arxiv.org/abs/2503.05592
- P07 StepTool: https://arxiv.org/abs/2410.07745
- P08 ReTool: https://arxiv.org/abs/2504.11536
- P11 TRACE: https://arxiv.org/abs/2602.21230
- P12 DORA Explorer: https://arxiv.org/abs/2604.17244
- P13 SciCrafter: https://arxiv.org/abs/2604.24697
- P17 Sufficient Context: https://openreview.net/forum?id=Jjr2Odj8DJ
- P20 CoRM-RAG: https://arxiv.org/abs/2605.01302
- P23 Causal Rewards: https://arxiv.org/abs/2501.09620
- P25 Reward Hacking Benchmark: https://arxiv.org/abs/2605.02964
- P27 Attention Bias Optimization: NeurIPS 2025
- P29 BFCL: https://gorilla.cs.berkeley.edu/leaderboard.html
- P32 Terminal-Bench: https://arxiv.org/abs/2601.11868
- P34 MCP-Zero: https://arxiv.org/abs/2506.01056
