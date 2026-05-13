# Screening Report — AgentRead × EMNLP

**Direction**: AgentRead: Diagnosing and Closing the Discovery-to-Utilization Gap in Tool-Augmented LLMs
**Venue**: EMNLP (Main + Findings)
**Date**: 2026-05-12
**Ideas screened**: 3 (IDEA-01 AttribRL, IDEA-02 DiscUseBench, IDEA-05 TrajFaith)
**Composite weights**: novelty=0.25, venue=0.35, strategic=0.20, feasibility=0.20
**Anchor design doc**: /Users/zpy/LLM_project/idea_paper/boardSearch/outputs/archive_2026-05-11_agentread-design_done/DESIGN_DOC_AgentRead.md
**Reviewer model**: gpt-5.5 (xhigh) via Codex MCP
**Thread ID**: 019e17cb-7b3e-7ea0-97ea-4f94759c0fb6

---

## Executive Summary

3 个候选 idea 全部跨过 ABANDON 线（≥5.0），但都未达 PROCEED 线（≥7.0），均归为 **PROCEED WITH CAUTION**。三方 reviewer 共识：EMNLP main 难度较高，**Findings 是稳健落点**，main 需要更强 empirical scale (≥2 model size, ≥3 seeds, repro checklist) 与更硬的 formalism（特别是 AttribRL 的 counterfactual validity）。

**关键发现**：
1. **AttribRL (IDEA-01)** — 概念上最 ambitious，5 个 reviewer 各打 Borderline/Weak Accept；最严重的方法学风险是 *KL(π|h,o) || π(h,o̅))* 测的是 distribution shift sensitivity 而非 causal necessity。需补 delayed-necessity 与 irrelevant-observation negative control。
2. **DiscUseBench (IDEA-02)** — 主要风险是相对 Anchor P01 的 incrementality；reviewers 警告"不要卖 leaderboard，要卖 conversion laws"。如执行得当 Findings 稳，Main 取决于机制发现强度。
3. **TrajFaith (IDEA-05)** — 概念清晰且 reviewer 接受度最高 (Findings/Accept)，但独立 paper 体量略小；建议要么扩为独立 Findings submission，要么内嵌为 DiscUseBench 的 evaluation 子模块。

**意外发现**：
- 文献交叉验证到 2 个 close prior 此前未在 landscape 中：**CST (Counterfactual Simulation Training, arXiv:2602.20710)** 与 **AgenTracer**。AttribRL 必须明确 frame 与 CST 的差别（object of optimization：CST 优化 CoT predictability；AttribRL 优化 policy action 对 observation 的 causal dependency）。
- **AgentSHAP** (OpenReview zSKpJF2lTU 2026) 和 **DFAH (Determinism-Faithfulness Agent Harness, arXiv:2601.15322)** 是 DiscUseBench/TrajFaith 的直接竞品，必须在 related work 区分。

**复合分数前后位置**: DiscUseBench (6.79) > TrajFaith (6.71) > AttribRL (6.54)。但**战略优先级**仍为 AttribRL → DiscUseBench → TrajFaith（基于 anchor design doc 与 user 明确指令"主投 AttribRL"）。AttribRL 复合分数偏低是因 Feasibility 罚分（HIGH risk + 2×4090 算力约束），并非概念弱。

---

## 共通 prior（影响所有 3 个 idea）

| Paper | Year | Status | 对每个 idea 的影响 |
|-------|------|--------|--------------------|
| Anchor "Agents Explore but Agents Ignore" (arXiv:2604.17609) | 2026 | preprint | DiscUseBench 直接 baseline；AttribRL 的 motivation 来源；TrajFaith 的现象证据 |
| TRM (OpenReview LnBEASInVr) | 2026 | OpenReview | AttribRL 必须 ablate；DiscUseBench 必须 evaluate |
| ToolRL (NeurIPS 2025) | 2025 | published | AttribRL 必须 baseline |
| **CST (Counterfactual Simulation Training, arXiv:2602.20710)** ⚠️ NEW | 2026 | preprint | AttribRL 必须明确区分 (CoT predictability vs action dependency) |
| **AgenTracer (OpenReview l05DseqvuD)** ⚠️ NEW | 2026 | OpenReview | AttribRL/TrajFaith 必须区分 (failure attribution vs training reward / faithfulness metric) |
| **AgentSHAP (OpenReview zSKpJF2lTU)** ⚠️ NEW | 2026 | OpenReview | DiscUseBench 必须 contrast |
| **DFAH (arXiv:2601.15322)** ⚠️ NEW | 2026 | preprint | TrajFaith 必须 contrast |

---

## Per-Idea Reports

### Idea 1: IDEA-01 AttribRL — Counterfactual-Lexical Attribution Rewards [PROCEED WITH CAUTION]

#### Module A — Novelty Assessment

- **Novelty Score**: 7.5/10
- **Core Claims**:
  1. Observation→action causality as reward target — Novelty **HIGH** — Closest: TRM (invocation correctness) / CST (CoT counterfactual predictability)
  2. Counterfactual + lexical self-supervised attribution reward + GRPO — Novelty **MEDIUM-HIGH** — Closest: CST (counterfactual reward 2026)
  3. Sparse K=5 garbage/random-doc counterfactual closes discovery-utilization gap — Novelty **MEDIUM** — Closest: AgenTracer counterfactual replay
- **Closest Prior**:

| Paper | Year | Overlap | Key Delta |
|-------|------|---------|-----------|
| Agents Explore but Agents Ignore (Anchor P01) | 2026 | HIGH | Anchor 是 diagnostic 不训练；AttribRL 把 gap 转成 reward 训练 |
| TRM | 2026 | HIGH | TRM 奖励 invocation correctness；AttribRL 奖励 observation utilization |
| ToolRL (NeurIPS '25) | 2025 | HIGH | ToolRL 奖励 outcome；AttribRL 奖励 observation→action 因果 |
| **CST (2602.20710)** | 2026 | MEDIUM-HIGH | CST 优化 CoT 让 simulator 可预测；AttribRL 优化 policy 让 action 依赖 observation |
| AgenTracer | 2026 | MEDIUM | AgenTracer 用 counterfactual 做失败归因；AttribRL 用作 reward signal |
| StepTool / AgentPRM / PORTool | 2024-25 | MEDIUM | step-level reward 缺 observation→action attribution |

- **Key differentiator**: 把"是否真的用了 observation"从事后诊断变成可训练的 dense process reward signal
- **Suggested positioning**: 不要 frame 成"又一个 tool-use RL"；frame 成 "**post-training LMs to condition actions on environmental evidence**"。主线必须是 Anchor gap 的 first training response，并把 CST/TRM/ToolRL 明确作为三条 closest axes 在 related work 与 method 对照表中显示。

#### Module B — Venue Simulation (EMNLP)

##### Reviewer 1 — Methodological Rigorist
- **Tier**: 1/2
- **Strengths**: (1) Reward objective 指向 tool-agent 核心失败模式，比 outcome-only 或 invocation correctness 贴近 utilization；(2) 与 GRPO、λ curriculum、process reward literature 对接清楚，有可能成为 reusable reward component。
- **Critical Weaknesses**: (1) KL 高不等于"正确使用 observation"，模型可能对 garbage 过敏；(2) R_lex 的 key span 来源不清，若由 LLM 抽取则不是完全 self-supervised；(3) 缺少与 TRM+ToolRL、StepTool、AgentPRM、PORTool 的组合/替换 ablation。
- **Verdict**: **Borderline** (=5)
- **Flip to Accept**: 证明 R_attr 与 human-labeled observation-use 有相关性，加入 irrelevant-observation negative controls 和 reward hacking audit。

##### Reviewer 2 — Empirical Pragmatist
- **Tier**: 2
- **Strengths**: (1) BFCL/ToolBench/API-Bank/WebArena-Lite/SWE-Bench-Lite 覆盖面强；(2) Qwen-7B conversion 18%→67% 若复现，足够 headline。
- **Critical Weaknesses**: (1) training 只写 Qwen2.5-7B；EMNLP main 需 ≥2 model size、multi-seed、CI/repro checklist；(2) 14 model evaluation 不能替代 2-size training validation；(3) WebArena/SWE-Bench-Lite 的 trajectory replay 成本、随机性、tool nondeterminism 需要置信区间。
- **Verdict**: **Weak Reject for Main / Weak Accept for Findings** (=5)
- **Flip to Accept**: 补 Qwen-14B 或 32B + 一个 Llama/DeepSeek 7B family，至少 3 seeds，公开 veRL fork、reward logs、failed-run traces。

##### Reviewer 3 — Theoretician/Reformulator
- **Tier**: 2
- **Strengths**: (1) 试图把 observation utilization formalize 成 policy sensitivity，是有价值的 causal-NLP bridge；(2) 与 CST 的 object of optimization 区分清楚。
- **Critical Weaknesses**: (1) KL(π(a|h,o)|π(a|h,õ)) 是 interventional sensitivity，不是充分的 causal necessity；(2) Counterfactual observation 用 garbage/random doc 可能破坏 support 导致 OOD artifact；(3) delayed utilization 没被 reward 捕捉。
- **Verdict**: **Borderline** (=5)
- **Flip to Accept**: 加入 local/delayed necessity 指标，并证明 counterfactual construction 保持 task-validity 或用 matched distractor controls。

##### Meta Review
- **核心争议**: 一致认可问题与方向重要；争议集中在 R_attr 是否真测量 utilization 而非 distribution shift sensitivity。
- **最终裁决**: **Findings / Weak Accept**；main 需要更强 theory + ≥2 size RL evidence
- **冲 Outstanding 路径**: diagnosis → causal reward → multi-domain post-training → interpretable failure taxonomy，公开 trajectories/reward model
- **Top 3 风险**:
  1. Reward hacking — 模型学会对任意 observation 敏感
  2. Empirical shortfall — 只 7B 单模型会被 main 卡
  3. Baseline risk — TRM/ToolRL/PORTool/AgentPRM 不全则差异化不足

**Venue Score**: (5+5+5)/3 = **5.0** (取严格平均；若按 Findings/Weak Accept 加权约 5.5)

#### Module C — Strategic Fit

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Longevity | 8/10 | tool-augmented LM 是 LM 后训练核心；observation utilization 是基本面问题 |
| Roadmap Viability | 9/10 | Paper 1 AttribRL → Paper 2 Causal AttribRL → Paper 3 cross-modal observation utilization → Paper 4 large-scale agent post-training |
| Application Grounding | 8/10 | 直接影响 tool agent deployment（BFCL/MCP/coding agents） |
| Execution Uniqueness | 5/10 | user 有完整 design doc + W1 pilot；但 Cluster A/B 团队（UIUC, Cohere/Edinburgh）也在做相邻，timeline 紧 |
| Iteration Readiness | 6/10 | pilot 周期 1 周；full training 2×4090 × 11d；可快速发现训练发散 |

**Strategic Score**: **7.2/10**
**Strategic recommendation**: 高 longevity + 高 roadmap，是用户应押重注的 idea；但 execution uniqueness 偏低反映 Cluster B (Cohere/Edinburgh) 6-9 个月内可能发表 training-side fix——必须抢节奏。建议 6-8 周内完成 main experiments。

#### Composite Score

| Component | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Novelty | 7.5/10 | 0.25 | 1.875 |
| Venue | 5.5/10 | 0.35 | 1.925 |
| Strategic | 7.2/10 | 0.20 | 1.440 |
| Feasibility | 6.5/10 | 0.20 | 1.300 |
| **Composite** | | | **6.54** |

**Recommendation**: **PROCEED WITH CAUTION**（战略优先级 #1）

---

### Idea 2: IDEA-02 DiscUseBench — Diagnostic Benchmark [PROCEED WITH CAUTION]

#### Module A — Novelty Assessment

- **Novelty Score**: 6.8/10
- **Core Claims**:
  1. Discovery / Use / Conversion taxonomy — **MEDIUM-HIGH** — Closest: Anchor P01 (discovery/interaction@k)
  2. 14 model × 5 benchmark diagnostic map — **MEDIUM** — Closest: Anchor P01 (4 model × 2 bench)
  3. Counterfactual observation replacement + leakage isolation + mechanism decomposition — **HIGH if executed rigorously**, otherwise MEDIUM
- **Closest Prior**:

| Paper | Year | Overlap | Key Delta |
|-------|------|---------|-----------|
| Anchor P01 | 2026 | VERY HIGH | DiscUseBench 在 P01 基础上扩 grid + causal use + leakage controls |
| T-Eval | 2024 | MEDIUM | T-Eval 是 step-by-step capability eval，不是 D/U/C taxonomy |
| TRACE (WWW 2026) | 2026 | MEDIUM | TRACE trajectory utility 不专门 discovery-utilization |
| **AgentSHAP** ⚠️ NEW | 2026 | MEDIUM | tool importance attribution；不是 D/U/C conversion |
| DFAH ⚠️ NEW | 2026 | LOW-MEDIUM | determinism/evidence faithfulness；不是 D/U/C taxonomy |

- **Key differentiator**: 把 "agent saw it" 与 "agent used it" 用 causal counterfactual protocol 拆开，而非只看 final success
- **Suggested positioning**: 不要卖 "大 leaderboard"；卖对 success-only evaluation 的机制性反例。novelty 必须来自 causal use + leakage controls + mechanism analysis 三件套，而非规模

#### Module B — Venue Simulation (EMNLP)

##### Reviewer 1 — Diagnostic Critic
- **Tier**: 1/2
- **Strengths**: (1) 正面挑战 pass@1/success-only paradigm，符合 EMNLP 对 analysis paper 的偏好；(2) Conversion 指标解释 "会发现但不用"。
- **Critical Weaknesses**: (1) 若结论只是复现 Anchor 的 ignore gap，main novelty 不够；(2) attention failure / overconfidence / readability 需要可操作判定，不能只 LLM judge narrative；(3) AttribRL 放进 14 models 可能让 benchmark 看起来服务自家方法。
- **Verdict**: **Weak Accept** (=6)
- **Flip to Accept**: 给出至少 2 个 Anchor 没发现的新机制结论，并用 intervention 证明因果。

##### Reviewer 2 — Benchmark Specialist
- **Tier**: 2
- **Strengths**: (1) 5 类任务覆盖 function calling/API/web/code；(2) Answer-swapped controls 是必要加分项。
- **Critical Weaknesses**: (1) 14×5×multi-seed 的 closed-model 成本和 API drift；(2) GPT-5.5/Claude Opus 4.7/Gemini 3 Pro 等版本必须冻结；(3) WebArena-Lite/SWE-Bench-Lite 的 replay 需要 deterministic harness。
- **Verdict**: **Borderline** (=5)
- **Flip to Accept**: 发布 frozen task set、dockerized tools、raw trajectories、API date/version、bootstrap CI、open-model-only degraded subset。

##### Reviewer 3 — Field Skeptic
- **Tier**: 2/3
- **Strengths**: (1) Taxonomy 清晰，可成为后续 tool-agent training 的 evaluation substrate；(2) mechanism decomposition 扎实则 Findings 合适。
- **Critical Weaknesses**: (1) "只是 benchmark 没方法"的风险，Anchor 已经定义 discovery/interaction；(2) 14 models 快速过期，静态 leaderboard 长期价值有限；(3) 需要明确 NLP contribution，非通用 agent engineering。
- **Verdict**: **Weak Reject for Main / Borderline-Weak Accept for Findings** (=5)
- **Flip to Accept**: paper 中心从 leaderboard 改为 empirical law: 什么 environment / tool schema / observation format 系统性导致 conversion collapse。

##### Meta Review
- **核心争议**: R1 认为 diagnostic insight 有价值；R3 认为缺方法贡献且 Anchor 太近；R2 担心可复现性。
- **最终裁决**: **Findings / Weak Accept**；main 需机制发现非常强
- **冲 Outstanding 路径**: 提出可复现的 "conversion laws"——observation salience、schema affordance、parametric leakage 与 use failure 的定量关系
- **Top 3 风险**:
  1. Incrementality over Anchor
  2. Closed-model reproducibility + API drift
  3. Counterfactual replacement protocol 被质疑为 artifact

**Venue Score**: (6+5+5)/3 = **5.33**（按 Findings/Weak Accept 校准为 5.5）

#### Module C — Strategic Fit

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Longevity | 7/10 | D/U/C taxonomy 持久但 14 model leaderboard 会快速过期 |
| Roadmap Viability | 6/10 | Paper 1 DiscUseBench → Paper 2 mechanism taxonomy；空间收窄 |
| Application Grounding | 7/10 | 影响 tool agent eval methodology |
| Execution Uniqueness | 6/10 | instrumentation 上 user 有 design doc 优势；Cluster B 有先发优势 |
| Iteration Readiness | 8/10 | 纯 inference，每个 setting 几小时 |

**Strategic Score**: **6.8/10**
**Strategic recommendation**: 作为 AttribRL 的 Plan B 兼数据基础设施，价值极高；独立 paper 而言 Findings 是稳妥落点。

#### Composite Score

| Component | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Novelty | 6.8/10 | 0.25 | 1.700 |
| Venue | 5.5/10 | 0.35 | 1.925 |
| Strategic | 6.8/10 | 0.20 | 1.360 |
| Feasibility | 9.0/10 | 0.20 | 1.800 |
| **Composite** | | | **6.79** |

**Recommendation**: **PROCEED WITH CAUTION**（战略优先级 #2，Plan B）

---

### Idea 3: IDEA-05 TrajFaith — Trajectory-Aware Faithfulness for Tool Agents [PROCEED WITH CAUTION]

#### Module A — Novelty Assessment

- **Novelty Score**: 7.2/10 (independent Findings paper) / 6.5/10 (as IDEA-02 submodule)
- **Core Claims**:
  1. RAG-style faithfulness systematically misjudges tool agents — **MEDIUM-HIGH** — Closest: CoRM-RAG
  2. Local necessity / delayed necessity / sequential sufficiency taxonomy — **HIGH** — No exact prior
  3. Replay-based trajectory faithfulness under nondeterminism — **MEDIUM** — Closest: DFAH
- **Closest Prior**:

| Paper | Year | Overlap | Key Delta |
|-------|------|---------|-----------|
| FaithfulRAG | 2025 | MEDIUM | document/fact-level，非 trajectory replay |
| CoRM-RAG | 2026 | MEDIUM | counterfactual RAG，非 observation-action dependency |
| Mindful-RAG | 2024 | LOW-MEDIUM | RAG failure analysis |
| **DFAH** ⚠️ NEW | 2026 | MEDIUM | determinism/evidence faithfulness；TrajFaith 强 sequential necessity |
| AgenTracer | 2026 | MEDIUM-HIGH | counterfactual replay for failure；TrajFaith 评 faithful dependence |
| TRACE | 2026 | MEDIUM | trajectory utility；TrajFaith 隔离 observation 必要性/充分性 |

- **Key differentiator**: faithfulness 从 document-answer alignment 升级到 observation→future-action→final-answer sequential causal dependency
- **Suggested positioning**: 提出 agent faithfulness 的 formal taxonomy，并展示它改变 model ranking、case interpretation、training reward diagnosis（不要只说 "RAG metric 不适用"）

#### Module B — Venue Simulation (EMNLP)

##### Reviewer 1 — NLP Empiricist
- **Tier**: 2
- **Strengths**: (1) tool-augmented LM evaluation/faithfulness 的 NLP 相关性；(2) local/delayed/sequential breakdown 解释长轨迹 failure。
- **Critical Weaknesses**: (1) 需 ≥2 benchmark × ≥2 size × multi-seed；(2) replay subsequent policy 对 closed model 成本高；(3) 需 human agreement calibration。
- **Verdict**: **Weak Accept if complete / Borderline otherwise** (=5.5)
- **Flip to Accept**: human annotation calibration、seed sensitivity、temp-0 vs temp>0、CI。

##### Reviewer 2 — Method Reviewer
- **Tier**: 1/2
- **Strengths**: (1) 与 RAG faithfulness 差异清楚；(2) 三类 metric 概念完整覆盖 immediate / latent / multi-step。
- **Critical Weaknesses**: (1) replacement→replay 易混 policy instability；(2) sequential sufficiency 定义需避免 brute-force ablation；(3) 与 AgenTracer/TRACE/DFAH/AgentSHAP 必须同表对比。
- **Verdict**: **Weak Accept** (=6)
- **Flip to Accept**: formal assumptions、counterfactual validity tests、cost-reduced approximation 与 exact replay 的 consistency analysis。

##### Reviewer 3 — Analysis Reviewer
- **Tier**: 1/2
- **Strengths**: (1) Analysis potential 极强，case studies 适合 EMNLP；(2) 可解释 final answer faithful 但 trajectory unfaithful。
- **Critical Weaknesses**: (1) 若仅作 IDEA-02 子组件贡献会稀释；(2) 需展示 metric 改变实际结论；(3) delayed necessity 可视化和 taxonomy 必须做深。
- **Verdict**: **Accept for Findings** (=7)
- **Flip to Accept**: 20-30 个 annotated cases，RAG metrics vs TrajFaith 的 systematic disagreement matrix。

##### Meta Review
- **核心争议**: 认可必要性；争议在 replay counterfactual 稳定性 + 独立成 paper 体量。
- **最终裁决**: **Findings / Accept**；main 需更大规模 + 更硬 formalism + 证明 metric 改变领域结论
- **冲 Outstanding 路径**: agent faithfulness 标准定义 = formal taxonomy + reproducible harness + human validation + 与 RAG metrics 的 strong disagreement analysis
- **Top 3 风险**:
  1. Replay nondeterminism 被误当作 unfaithfulness
  2. 与 AgenTracer/TRACE/DFAH 的边界不够硬
  3. 独立 paper 体量不足 / 子模块时埋没 novelty

**Venue Score**: (5.5+6+7)/3 = **6.17**（Meta 偏 Findings/Accept 调到 6.2）

#### Module C — Strategic Fit

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Longevity | 7/10 | trajectory faithfulness 是较持久概念 |
| Roadmap Viability | 6/10 | Paper 1 TrajFaith → Paper 2 training-side faithful reward，与 AttribRL 重叠 |
| Application Grounding | 6/10 | 主要影响 evaluation methodology |
| Execution Uniqueness | 5/10 | Cluster D（RAG faithfulness）有先发；user 独特在 tool replay |
| Iteration Readiness | 7/10 | 无训练 |

**Strategic Score**: **6.2/10**

#### Composite Score

| Component | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Novelty | 7.2/10 | 0.25 | 1.800 |
| Venue | 6.2/10 | 0.35 | 2.170 |
| Strategic | 6.2/10 | 0.20 | 1.240 |
| Feasibility | 7.5/10 | 0.20 | 1.500 |
| **Composite** | | | **6.71** |

**Recommendation**: **PROCEED WITH CAUTION**（建议作为独立 Findings 投稿 OR 内嵌 DiscUseBench 的 evaluation 子模块）

---

## Strategic Composition Decision (战略组合决策)

复合分数前后位置 DiscUseBench (6.79) > TrajFaith (6.71) > AttribRL (6.54)。

但**战略优先级**为：
1. **AttribRL** (#1 战略) — anchor design doc 主轴；HIGH risk / HIGH reward；冲 Main / Outstanding 候选
2. **DiscUseBench** (#2 战略) — AttribRL 的 Plan B + 训练数据基础设施；LOW risk；Findings 稳
3. **TrajFaith** (#3 战略) — 独立 Findings 候选 OR 作 DiscUseBench evaluation 子模块

**Phase 4 idea-refine 建议**: 重点 refine **AttribRL** 与 **DiscUseBench**，TrajFaith 暂缓为 Phase 5 final report 中"建议同步推进的副线 paper"。
