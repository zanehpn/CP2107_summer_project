# AutoVibeIdea Pipeline Log — AgentRead

**Topic**: AgentRead: Agents Read but Don't Use — Diagnosing and Closing the Discovery-to-Utilization Gap in Tool-Augmented LLMs
**Venue**: EMNLP
**Reviewer Model**: gpt-5.5
**Anchor Design Doc**: `/Users/zpy/LLM_project/idea_paper/boardSearch/outputs/archive_2026-05-11_agentread-design_done/DESIGN_DOC_AgentRead.md`
**Start**: 2026-05-11 23:50:49 +08:00

---

## Pipeline Configuration

- REFINE_TOP_N = 2
- DEFAULT_VENUE = EMNLP (user-specified)
- 4 阶段：lit-survey → idea-gen → idea-screen → idea-refine（含 Phase 5.5 Deep Expansion + Phase 1.4.TE Theory-Experiment Alignment）

## Anchor 事实（来自 DESIGN_DOC_AgentRead.md）

- gpt-oss-120b: 97.54% discovery vs 0.53% use → conversion rate 0.5%
- AttribRL：counterfactual + lexical 双源 attribution reward + GRPO
- DiscUseBench: 14 model × 5 bench (BFCL/ToolBench/API-Bank/WebArena-Lite/SWE-Bench-Lite)
- Plan B: diagnostic-only paper
- 4 周计划，2×4090，约 21 GPU-day

---

## Checkpoint Log

### Checkpoint 1 — Literature Survey ✅ (2026-05-12 00:08)

- **状态**: 完成
- **输出**: `outputs/LANDSCAPE.md`, `outputs/LANDSCAPE.json`
- **统计**: 35 篇论文 / 6 主题 / 12 gap
- **覆盖方向**:
  - D1 Tool-use RL: TRM (P02), ToolRL (P03), Search-R1 (P04), R1-Searcher (P05/P06), StepTool (P07), ReTool (P08), iTool (P09), Tool Zero (P10)
  - D2 Agent diagnostics: Anchor "Agents Explore but Ignore" (P01), TRACE (P11), DORA (P12), SciCrafter (P13)
  - D3 Context utilization: Sufficient Context (P17), Lost in Middle (P18), FaithfulRAG/CoRM-RAG/FaithEval/Mindful-RAG (P19-P22)
  - D4 Counterfactual/Attribution: Causal Rewards (P23), CF Multimodal (P24), Reward Hacking Bench (P25), Attention Bias Optimization (P27), LoGra (P28)
  - D5 Benchmarks: BFCL/ToolBench/API-Bank/Terminal-Bench/AppWorld (P29-P33), MCP-Zero/RAG-MCP (P34-P35)
- **关键高置信 gap**:
  - G1 14-model×5-bench DiscUseBench 缺位
  - G2 现有 RL 无 utilization reward
  - G3 counterfactual 信号未上 tool agent RL
  - G7 lexical+counterfactual 组合防御未被尝试
  - G10 RAG counterfactual perturbation 未上 trajectory
  - G12 Plan B diagnostic-only paper 路径开放
- **EMNLP 适配度**: 高（P09/P10/P15/P16/P31 均 EMNLP 系列接收 tool-use RL）
- **决策**: 立即进入 Phase 2 idea-gen，以 Anchor design doc 中的 AttribRL 为主候选

### Checkpoint 2 — Idea Generation ✅ (2026-05-12 00:35)

- **状态**: 完成
- **输出**: `outputs/CRITICAL_ANALYSIS.md`, `outputs/IDEAS_RAW.md`, `outputs/IDEAS_FILTERED.md`
- **Codex thread**: 019e17c2-d657-7c51-9a97-a817899d3399 (gpt-5.5 xhigh)
- **Phase 2a**: 14 个 CRITIQUE 条目（5 Unverified Assumption, 3 Incorrect Generalization, 4 Experimental Flaw, 3 Cross-Domain Misfit）
- **Phase 2b**: 11 个 raw ideas，每个 anchor 到 1+ CRITIQUE 且含 Theorem Scaffold + ML 客观规律列表（R3 满足）
- **Filter outcome**: 11 → 6 surviving（Risk 分布 2H/2M/2L）
- **Top 2**:
  1. **IDEA-01 AttribRL** (He 19/20) — counterfactual+lexical attribution reward + GRPO；anchor design doc 直系；CRITIQUE-02/03/05/07/14 五条同时消化
  2. **IDEA-02 DiscUseBench** (He 17/20) — Plan B 诊断 paper；CRITIQUE-01/04/08/09 四条消化；14 model × 5 bench 三维表
- **5 个 eliminated ideas** 被建议合并为 IDEA-01/02 的 subsections（IDEA-07 judge audit + IDEA-08 mechanism + IDEA-09 obs quality + IDEA-10 PRM audit + IDEA-11 λ schedule）
- **决策**: 立即进入 Phase 3 screening，top 2 全部投 EMNLP

### Checkpoint 3 — Idea Screening ✅ (2026-05-12 01:10)

- **状态**: 完成
- **输出**: `outputs/SCREENING_REPORT.md`, `outputs/SCREENING_RANKED.md`
- **Codex thread**: 019e17cb-7b3e-7ea0-97ea-4f94759c0fb6 (gpt-5.5 xhigh)
- **Screened**: 3 ideas (IDEA-01 AttribRL, IDEA-02 DiscUseBench, IDEA-05 TrajFaith)
- **Composite (Novelty 0.25 + Venue 0.35 + Strategic 0.20 + Feasibility 0.20)**:
  - IDEA-02 DiscUseBench: **6.79** (Plan B Findings)
  - IDEA-05 TrajFaith: **6.71** (Findings/Accept potential)
  - IDEA-01 AttribRL: **6.54** (主战略 idea)
- **3 个 ideas 全部 PROCEED WITH CAUTION**（无 PROCEED；无 ABANDON）
- **关键 reviewer 共识**:
  - EMNLP main 难度较高；Findings 是稳健落点
  - AttribRL 最大风险：KL 高 ≠ causal necessity；needs delayed-necessity + irrelevant-observation negative control
  - DiscUseBench 最大风险：incrementality over Anchor P01
  - TrajFaith 最大风险：与 AgenTracer/DFAH 边界 + 独立 paper 体量
- **新发现的 close priors**（lit-survey 阶段漏掉）:
  - **CST (Counterfactual Simulation Training, arXiv:2602.20710)** — AttribRL 必须显式区分
  - **AgenTracer (OpenReview l05DseqvuD)** — counterfactual replay for failure attribution
  - **AgentSHAP (OpenReview zSKpJF2lTU)** — tool importance attribution
  - **DFAH (arXiv:2601.15322)** — agent determinism/faithfulness harness
- **决策**:
  - Phase 4 refine top 2 = AttribRL（socratic-auto 模式）+ DiscUseBench（标准 ≤3 轮模式）
  - TrajFaith defer（建议作为 DiscUseBench evaluation 子模块 OR 独立 Findings 投稿）
  - 战略覆写：AttribRL 复合分最低但 anchor design doc 优先级 #1

### Checkpoint 4 — Idea Refinement ✅ (2026-05-12 01:55)

- **状态**: 完成（双 paper 路径）
- **输出**:
  - `refine-logs/FINAL_PROPOSAL.md` (AttribRL, canonical)
  - `refine-logs/FINAL_PROPOSAL_DiscUseBench.md`
  - `refine-logs/REVIEW_SUMMARY.md`, `REFINEMENT_REPORT.md`, `score-history.md`
  - `refine-logs/socratic-turn-{0,1,2}-{questions,answers}.md` + `socratic-final-review.md`
  - `refine-logs/discusebench-skeleton.md`
- **AttribRL (Socratic 3-turn, gpt-5.5 xhigh thread 019e17d4-...)**:
  - Score evolution: 6.54 (screening) → **7.62/10** (REVISE) → +1.08 ✓ exceeded ≥7.5 target
  - Method upgrades through dialogue: gated min mixture / sigmoid normalization + absolute floor + null-distractor calibration / two-stage fn-tok + arg-tok KL / D_t judge v2 evidence-availability / Lemma 1.a verbatim defense / 6+3+1 variant ablation matrix
  - Post-revisions: drop adaptive α; delayed KL → diagnostic; V9 → related baseline; Pearl framing → soft motivation
- **DiscUseBench (single codex review, thread 019e17e5-...)**:
  - Score: 6.65 (REVISE) → post-simplification est 7.2
  - Key revisions: Use def 3-layer (no closed-model KL dep); 14×5 → 8×4; pre-registered H1/H2; NLP-anchored title "Diagnosing Evidence-to-Action Conversion in Language Agents"; TrajFaith merger; AttribRL 不作 baseline
- **决策**: 进入 Phase 5 final report

### Checkpoint 5 — Final Report ✅ (2026-05-12 02:05)

- **状态**: 完成；pipeline 结束
- **输出**: `outputs/IDEA_DISCOVERY_REPORT.md`（aggregates all phases + manifest + strategic recommendations + pivot decision tree + long-term roadmap）
- **总耗时**: ≈2 小时（5 phases）
- **Codex MCP calls**: 4 unique threads；全程 model_reasoning_effort=xhigh
- **API budget remaining**: $30 - ≈$2 used so far on screening/refinement = $28 for W1 pilot

