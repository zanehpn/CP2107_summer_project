# Generated Research Ideas (Raw) — AgentRead

**Direction**: AgentRead: Diagnosing and Closing the Discovery-to-Utilization Gap in Tool-Augmented LLMs
**Date**: 2026-05-12
**Model**: gpt-5.5 (xhigh reasoning), via Codex MCP
**Landscape source**: outputs/LANDSCAPE.md (35 papers, 12 gaps, 6 themes)
**Ideas generated**: 11
**Codex thread ID**: 019e17c2-d657-7c51-9a97-a817899d3399
**Anchor design doc**: /Users/zpy/LLM_project/idea_paper/boardSearch/outputs/archive_2026-05-11_agentread-design_done/DESIGN_DOC_AgentRead.md

---

## IDEA-01: AttribRL — Counterfactual-Lexical Attribution Rewards for Tool-Augmented Agents

- **Anchored Critique**: CRITIQUE-02 + CRITIQUE-03 + CRITIQUE-05 + CRITIQUE-07 + CRITIQUE-14。当前 RL/PRM 只奖励 outcome 或 step quality；AttribRL 把 utilization 显式写进 reward。
- **Thesis**: We show that tool-augmented agents can close the Discovery-to-Utilization gap by optimizing a self-supervised dual-source attribution reward under GRPO.
- **Gap addressed**: G2, G3, G4, G5, G7, G8
- **Core mechanism**: 对每个 trajectory 构造两个 attribution signals — (i) counterfactual reward 衡量删/换 observation 后 action 与 final answer 的变化；(ii) lexical reward 衡量后续 reasoning/answer 是否引用 observation 中的关键 evidence span。组合为 utilization reward，与 outcome reward 通过 λ schedule 混合，避免单 lexical copy-hack 与单 counterfactual 高成本。
- **Non-obvious because**: Skeptic — "outcome reward 自然学会用工具"；Rebuttal — 97.54% discovery vs 0.53% use 证伪此假设，AttribRL 直接优化此缺失变量。
- **Theorem/Conjecture Scaffold**:
  - $R(\tau) = R_{\text{out}}(\tau) + \lambda_t R_{\text{attr}}(\tau)$, with $R_{\text{attr}} = \alpha R_{\text{cf}} + (1-\alpha) R_{\text{lex}}$
  - $R_{\text{cf}} = \mathrm{KL}\big(\pi_\theta(\cdot|h_t,o_t) \,\|\, \pi_\theta(\cdot|h_t,\tilde{o}_t)\big)$
  - GRPO update: $\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_i[\min(r_i A_i, \text{clip}(r_i,1-\epsilon,1+\epsilon)A_i)]$ where $A_i$ from group-normalized mixed reward.
- **ML 子领域客观规律**:
  - GRPO/PPO clipping → 控制 policy drift；实验需报告 KL/reward hacking/conversion 曲线
  - Counterfactual identifiability (Pearl backdoor) → observation replacement 近似 do-intervention；需 oracle-known + answer-swapped controls
  - Reward learning consistency → mixed reward 可能引入 spurious optimum；必须 sweep λ Pareto curve
- **Contribution type**: new method + new formulation + empirical
- **Risk**: HIGH — 训练侧结果可能受算力限制，但 3B/7B QLoRA + DiscUseBench subset 可得 publishable evidence
- **Effort**: 6-9 person-weeks
- **Closest work**: ToolRL (Qian et al. NeurIPS 2025) — delta：ToolRL 奖励 final outcome (+ fine-grained reward types)；AttribRL 奖励 observation→action attribution 并显式测 Discovery/Use/Conversion。
- **Plan B 关联**: 若训练失败可退至 IDEA-02 (DiscUseBench) 单独发表

---

## IDEA-02: DiscUseBench — A Diagnostic Benchmark for Discovery, Use, and Conversion in Tool Agents [Plan B]

- **Anchored Critique**: CRITIQUE-01 + CRITIQUE-04 + CRITIQUE-08 + CRITIQUE-09
- **Thesis**: We show that tool-agent success hides distinct failure modes by measuring Discovery / Use / Conversion across 14 models and 5 benchmarks.
- **Gap addressed**: G1, G5, G6, G12
- **Core mechanism**: 不训练模型，统一 instrumentation — 记录 tool output 是否包含 answer、后续 action 是否使用该 evidence、最终答案是否正确，并按 task type / model family / tool format 分解 conversion bottleneck。
- **Non-obvious because**: Skeptic — "只是 benchmark 不是方法"；Rebuttal — Anchor P01 显示单点 gap 极端但缺规模化 evidence；EMNLP 接受强诊断论文，尤其当它推翻 success-only evaluation。
- **Theorem/Conjecture Scaffold**:
  - Empirical hypothesis: $\text{Success} \ne \text{Discovery} \times \text{Use}$（条件期望意义下）
  - More specifically: models with similar final accuracy will show statistically different conversion rates under controlled discovery conditions
- **ML 子领域客观规律**:
  - Measurement reliability → inter-judge agreement、bootstrap CI 设计 judge protocol
  - Causal decomposition → success = Discovery × Use × Reasoning，报告 conditional rates
  - Error taxonomy validity → clustering / mutual information 检验 failure modes 是否稳定 cross-benchmark
- **Contribution type**: diagnostic + empirical benchmark
- **Risk**: LOW — instrumentation + careful evaluation；不依赖训练成功
- **Effort**: 4-6 person-weeks
- **Closest work**: "Agents Explore but Agents Ignore" (Engländer et al. 2026, P01) — delta：从 2 benchmarks 扩到 5 benchmarks × 14 models，加入 mechanism analysis 与更细粒度 use definition。

---

## IDEA-03: Causal AttribRL — Explicit Trajectory Causal Models Replace Lexical Attribution

- **Anchored Critique**: CRITIQUE-05 + CRITIQUE-12
- **Thesis**: We show that explicit causal trajectory attribution is a more faithful reward signal than lexical overlap or raw attention by modeling observation intervention effects over future actions.
- **Gap addressed**: G3, G5, G6, G7
- **Core mechanism**: 把 trajectory 建模为 SCM (history $H_t$, observation $O_t$, action $A_{t+1}$, final answer $Y$)。reward 不看 surface overlap，而看 $do(O_t = \tilde{O}_t)$ 对 $A_{t+1:T}$ 与 $Y$ 的 average treatment effect。
- **Non-obvious because**: Skeptic — "attention 或 lexical 已够便宜"；Rebuttal — attention 高亮无关 context，lexical 可被 copy hack；causal effect 虽贵但可作 high-precision teacher 或 reward calibration signal。
- **Theorem/Conjecture Scaffold**:
  - $U_t = \mathbb{E}_{\tilde{o}\sim q(\tilde{O})}\big[D\big(p_\theta(A_{t+1:T},Y|H_t,O_t), p_\theta(A_{t+1:T},Y|H_t,\tilde{o})\big)\big]$
  - Conjecture：当 $U_t$ 上升而 $R_{\text{out}}$ 不变时，conversion rate 比 lexical-only reward 提升更稳健
- **ML 子领域客观规律**:
  - Pearl intervention / backdoor → 控制 $H_t$ 后估计 $O_t \to A_{t+1}$ causal effect
  - Attention faithfulness critique (Jain & Wallace) → require comparing attention with counterfactual output change
  - Variance-cost tradeoff → intervention sampling 数量决定 estimator variance；sweep 1/2/4 counterfactuals
- **Contribution type**: new method + theoretical formulation
- **Risk**: HIGH — causal score 成本较高，可能只适合 reward model teacher 而非 online RL reward
- **Effort**: 7-10 person-weeks
- **Closest work**: "Beyond Reward Hacking: Causal Rewards" (P23) — delta：从 static RM counterfactual invariance 转向 sequential tool-agent trajectory SCM 上的 causal utilization reward。

---

## IDEA-04: From Tool Recall to Tool Utilization — Joint Optimization for MCP Retrieval Agents

- **Anchored Critique**: CRITIQUE-11
- **Thesis**: We show that high tool Recall@k does not imply downstream utilization by optimizing retrieval with a conversion-aware objective.
- **Gap addressed**: G11, G2
- **Core mechanism**: retriever reward 从 "gold tool in top-k" 改为 "retrieved tool caused useful downstream action"。训练或 rerank 时加入 utilization-conditioned labels：tool 被检索 → 被调用 → output 被使用 → final answer 正确。
- **Non-obvious because**: Skeptic — "Tool retrieval 是 IR 问题，Recall@k 足够"；Rebuttal — agent setting 中 Recall@k 只是入口指标；真正瓶颈是 retrieved tool 是否进入 causal trajectory。
- **Theorem/Conjecture Scaffold**:
  - Empirical hypothesis: $\rho(\text{Recall@k}, \text{TaskSuccess}) < \rho(\text{ToolConversion@k}, \text{TaskSuccess})$
  - Reranking score: $s(q,t) = s_{\text{ret}}(q,t) + \beta \cdot \hat{P}(\text{Use}=1|q,t,h)$
- **ML 子领域客观规律**:
  - Learning-to-rank consistency → ranking loss must align with downstream utility
  - Off-policy evaluation → logged trajectories 估计 tool utility 用 IPS correction
  - IR metric mismatch → optimize nDCG/Recall 只在 relevance is final objective；此处 relevance is intermediate
- **Contribution type**: new formulation + empirical
- **Risk**: MEDIUM — 需要 MCP-style tasks，但 small tool library 可模拟
- **Effort**: 5-7 person-weeks
- **Closest work**: MCP-Zero (P34) / RAG-MCP (P35) — delta：从 Recall@k 评估转为 downstream utilization 与 conversion 评估。

---

## IDEA-05: TrajFaith — Agent-Specific Faithfulness Evaluation for Tool Trajectories

- **Anchored Critique**: CRITIQUE-06 + CRITIQUE-13
- **Thesis**: We show that RAG-style faithfulness metrics misjudge tool agents by introducing trajectory-aware sufficiency and necessity tests.
- **Gap addressed**: G10, G5
- **Core mechanism**: 定义三类 trajectory faithfulness — local necessity / delayed necessity / sequential sufficiency。不只是删除 passage，而是替换某步 observation 并 replay 后续 policy，观察 later tool calls / intermediate states / final answer 是否变化。
- **Non-obvious because**: Skeptic — "tool output 本质也是 retrieved context"；Rebuttal — tool output 改变的是 future action distribution，不只是 final answer text；RAG metric 漏掉 delayed use。
- **Theorem/Conjecture Scaffold**:
  - Empirical hypothesis: RAG faithfulness score 与 trajectory faithfulness score 在 25-40% successful tool-agent trajectories 上 disagree
  - Trajectory necessity: $N_t = \mathbb{1}[Y(O_t) \ne Y(\tilde{O}_t) \lor A_{t+1:T}(O_t) \ne A_{t+1:T}(\tilde{O}_t)]$
- **ML 子领域客观规律**:
  - Causal mediation → observation 通过 intermediate actions 影响 answer，evaluation must include mediators
  - Sequential decision theory → action dependency is history-conditioned not document-conditioned
  - RAG faithfulness perturbation → useful baseline；experiments must include delayed-effect cases
- **Contribution type**: diagnostic + new evaluation formulation
- **Risk**: MEDIUM — 无训练；需 careful trajectory replay infra
- **Effort**: 5-8 person-weeks
- **Closest work**: CoRM-RAG (P20) — delta：从 retrieved-document counterfactuals 转向 sequential tool trajectory counterfactuals。

---

## IDEA-06: LeakProof-Use — Controlled Benchmarks for Parametric-Knowledge Isolation in Tool Agents

- **Anchored Critique**: CRITIQUE-10
- **Thesis**: We show that apparent tool use is inflated by parametric knowledge leakage through answer-swapped and freshness-controlled tool tasks.
- **Gap addressed**: G5, G6
- **Core mechanism**: 构造三种 controlled instances — fresh synthetic facts / answer-swapped known facts / hidden-state tool outputs。若 model 在 observation 被替换后仍输出原答案 → parametric shortcut，非 tool utilization。
- **Non-obvious because**: Skeptic — "大模型知道答案也是能力"；Rebuttal — 对 tool-agent paper 而言，问题不是是否答对，而是是否因 tool observation 答对；这决定 RL reward 是否真训练了 tool use。
- **Theorem/Conjecture Scaffold**:
  - Empirical hypothesis: $\text{Use}_{\text{standard}} > \text{Use}_{\text{leak-controlled}}$ 且 gap 随 parametric model strength 增大
- **ML 子领域客观规律**:
  - Dataset contamination theory → memorization inflates generalization estimates
  - Counterfactual evaluation → answer-swapping creates intervention on evidence while preserving prompt structure
  - Calibration analysis → stronger models rely more on prior confidence，report confidence-conditioned use
- **Contribution type**: diagnostic benchmark
- **Risk**: LOW — 小规模可完成，negative result 同样强
- **Effort**: 3-5 person-weeks
- **Closest work**: Sufficient Context (P17) — delta：从 RAG answerability 转为 tool-agent parametric leakage isolation。

---

## IDEA-07: Can LLM Judges Detect Tool Utilization? A Stress Test of Use Metrics

- **Anchored Critique**: CRITIQUE-09
- **Thesis**: We show that LLM judges systematically overestimate tool utilization by testing them on counterfactual, adversarial, and rationale-swapped trajectories.
- **Gap addressed**: G5, G12
- **Core mechanism**: 构建 judge stress suite — same answer/different evidence、copied irrelevant evidence、correct answer with unused tool、wrong answer with correct evidence、post-hoc rationale injection。比较 LLM judge 与 counterfactual oracle labels。
- **Non-obvious because**: Skeptic — "LLM-as-judge 在 NLP eval 成熟"；Rebuttal — use 判定不是 quality judgment 而是 causal attribution；judge 看完整 transcript 易被 post-hoc rationalization 欺骗。
- **Theorem/Conjecture Scaffold**:
  - Empirical hypothesis: $\text{FPR}_{\text{judge}}(\text{Use})$ 随 lexical overlap 单调上升，even when counterfactual oracle says Use = 0
- **ML 子领域客观规律**:
  - Evaluator bias → judge models correlate with surface similarity / answer agreement
  - Causal attribution → use labels need intervention-based validation
  - Inter-rater reliability → Cohen's κ / Krippendorff's α across judges and oracle
- **Contribution type**: diagnostic + evaluation audit
- **Risk**: LOW — 少量 API budget 即可；适合 EMNLP Findings
- **Effort**: 3-4 person-weeks
- **Closest work**: FaithEval (P21) — delta：从 faithfulness judge audit 转向 tool-utilization causal judge audit。

---

## IDEA-08: Failure Modes of the Discovery-to-Utilization Gap — A Mechanism Decomposition Study

- **Anchored Critique**: CRITIQUE-04
- **Thesis**: We show that the Discovery-to-Utilization gap decomposes into distinct mechanisms by clustering failures across attention, confidence, tool format, and reasoning state.
- **Gap addressed**: G6, G12
- **Core mechanism**: 对失败 trajectory 提取 mechanism probes — pre-tool confidence、post-tool answer shift、observation position、tool-output entropy、evidence span length、copy rate、self-correction behavior。supervised taxonomy + clustering 验证失败模式是否稳定。
- **Non-obvious because**: Skeptic — "Gap 已被 P01 发现，机制分析只是补充"；Rebuttal — 无机制分解，训练方法会把不同病因混成同一个 reward；直接影响 AttribRL 的 reward design。
- **Theorem/Conjecture Scaffold**:
  - Empirical hypothesis: 至少 3 个 separable modes 解释多数 non-use cases — prior-overconfidence / observation-unreadability / delayed-integration failure
  - Formal test: cluster stability $\text{AMI} > \tau$ across model families and benchmarks
- **ML 子领域客观规律**:
  - Representation clustering validity → AMI / silhouette 检 taxonomy 稳定性
  - Calibration theory → overconfidence predicts resistance to observation update
  - Causal mediation → observation may fail at perception / trust / integration
- **Contribution type**: diagnostic + mechanism analysis
- **Risk**: MEDIUM — 分析复杂但不依赖训练
- **Effort**: 5-7 person-weeks
- **Closest work**: SciCrafter (P13) — delta：从 capacity decomposition 转为 tool-observation utilization mechanism decomposition。

---

## IDEA-09: Observation Quality as the Hidden Bottleneck in Tool-Agent Utilization

- **Anchored Critique**: CRITIQUE-01
- **Thesis**: We show that a large fraction of non-use is caused by observation quality rather than agent incapacity by rewriting tool outputs into controlled evidence formats.
- **Gap addressed**: G5, G6
- **Core mechanism**: 对同一 tool result 生成多种 observation formats — raw dump / highlighted evidence / structured JSON / minimal answer sentence / conflicting noisy context。固定 answer content，改变 readability/salience/structure，测 conversion rate。
- **Non-obvious because**: Skeptic — "agent 应该能读 raw output"；Rebuttal — 若同一 evidence 在 structured format 下 conversion 大幅升，说明 gap 部分来自 tool interface，而非 reasoning。
- **Theorem/Conjecture Scaffold**:
  - Empirical hypothesis: $\text{Conversion}(\text{structured}) > \text{Conversion}(\text{raw})$ under matched Discovery + answer content
- **ML 子领域客观规律**:
  - Lost in the Middle → evidence position/salience affect utilization
  - Information bottleneck → structured outputs reduce irrelevant entropy
  - HCI representation principle → format changes decision quality even with identical information
- **Contribution type**: empirical diagnostic + intervention
- **Risk**: MEDIUM — task-dependent；negative result still informative
- **Effort**: 4-6 person-weeks
- **Closest work**: Lost in the Middle (P18) — delta：从 long-context retrieval placement 转向 tool-output readability and conversion。

---

## IDEA-10: What Do Tool Process Reward Models Actually Reward?

- **Anchored Critique**: CRITIQUE-07
- **Thesis**: We show that existing tool PRMs reward plausible progress more than true observation dependence by evaluating them on counterfactual step pairs.
- **Gap addressed**: G4, G6
- **Core mechanism**: 构造 matched step pairs — 看起来等价"good for task"，但一个依赖真实 observation，另一个使用替换/无关 observation。测试 TRM-style 或 open PRMs 是否能区分 attribution-grounded step。
- **Non-obvious because**: Skeptic — "PRM 已评 step quality，自然包含 grounding"；Rebuttal — step quality 是 outcome-correlated property，grounding 是 causal property；二者在 counterfactual pairs 中可分离。
- **Theorem/Conjecture Scaffold**:
  - Empirical hypothesis: $P(R_{\text{PRM}}(s_{\text{grounded}}) > R_{\text{PRM}}(s_{\text{ungrounded}})) \approx 0.5$ on matched counterfactual pairs
- **ML 子领域客观规律**:
  - Bradley-Terry reward modeling → pairwise preference consistency testable on controlled pairs
  - Causal invariance → true grounding reward should change under observation intervention
  - Shortcut learning → models exploit format/progress cues unless adversarial controls used
- **Contribution type**: diagnostic + reward-model audit
- **Risk**: MEDIUM — 需找或 approximate existing PRMs；可 proxy
- **Effort**: 5-7 person-weeks
- **Closest work**: TRM (P02) — delta：从 training invocation correctness PRM 转向 auditing whether PRM detects observation-action causality。

---

## IDEA-11: λ Is the Method — Reward-Mixing Schedules for Utilization-Aware Tool RL

- **Anchored Critique**: CRITIQUE-14
- **Thesis**: We show that utilization-aware RL succeeds or fails mainly through reward-mixing dynamics by mapping the Pareto frontier between outcome accuracy and attribution fidelity.
- **Gap addressed**: G8, G7
- **Core mechanism**: 不提新 reward，系统比较 fixed λ / linear warmup / curriculum λ / adaptive λ。核心指标不是单点 accuracy 而是 accuracy-use-copying 三方 Pareto frontier。
- **Non-obvious because**: Skeptic — "λ 只是 hyperparameter"；Rebuttal — 在 utilization RL 中 λ 决定模型是继续忽略工具 / 真用 / 还是 hack 成复制；它是 training dynamics 的核心变量。
- **Theorem/Conjecture Scaffold**:
  - Objective: $R_t = R_{\text{out}} + \lambda_t R_{\text{use}}$
  - Adaptive schedule: $\lambda_{t+1} = \lambda_t + \eta(\text{TargetUse} - \widehat{\text{Use}}_t)$
  - Conjecture: adaptive λ achieves higher conversion at matched final accuracy than fixed λ
- **ML 子领域客观规律**:
  - Multi-objective optimization → mixed rewards trace Pareto frontier
  - Curriculum learning → early high outcome reward stabilizes policy；later use reward changes attribution
  - PPO/GRPO KL control → λ changes advantage scale；experiments must normalize rewards and track KL
- **Contribution type**: empirical + training dynamics
- **Risk**: MEDIUM — training compute nontrivial；3B model + short trajectories + QLoRA 可行
- **Effort**: 5-8 person-weeks
- **Closest work**: Search-R1 (P04) — delta：not another tool-RL recipe；systematic reward-mixing dynamics for utilization。
