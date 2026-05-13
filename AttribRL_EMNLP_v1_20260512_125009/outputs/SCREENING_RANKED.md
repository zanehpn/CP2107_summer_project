# Screening Results: Ranked Ideas — AgentRead × EMNLP

**Direction**: AgentRead: Diagnosing and Closing the Discovery-to-Utilization Gap in Tool-Augmented LLMs
**Venue**: EMNLP (Main + Findings)
**Date**: 2026-05-12
**Ideas screened**: 3
**Reviewer model**: gpt-5.5 (xhigh) via Codex MCP

## Rankings

| Rank | Idea | Novelty | Venue | Strategic | Feasibility | Composite | Recommendation |
|------|------|---------|-------|-----------|-------------|-----------|----------------|
| 1 (composite) | IDEA-02 DiscUseBench | 6.8 | 5.5 | 6.8 | 9.0 | **6.79** | PROCEED WITH CAUTION |
| 2 (composite) | IDEA-05 TrajFaith | 7.2 | 6.2 | 6.2 | 7.5 | **6.71** | PROCEED WITH CAUTION |
| 3 (composite) | IDEA-01 AttribRL | 7.5 | 5.5 | 7.2 | 6.5 | **6.54** | PROCEED WITH CAUTION |

> **战略优先级覆写**：尽管 AttribRL 复合分数最低，但 anchor design doc 与 user 明确指令将 AttribRL 设为 main paper。Phase 4 refinement 顺序为 **AttribRL > DiscUseBench > TrajFaith (defer)**。

---

## Rank 1 (composite): IDEA-02 DiscUseBench — Diagnostic Benchmark [PROCEED WITH CAUTION]

### Module A: Novelty
- **Score**: 6.8/10
- **Key differentiator**: 把 "agent saw it" 与 "agent used it" 用 causal counterfactual protocol 拆开，而非只看 final success
- **Closest prior**: Anchor "Agents Explore but Agents Ignore" (P01, 2026)；delta = 5 bench × 14 model + causal use definition + leakage controls + mechanism analysis
- **⚠️ New prior discovered**: AgentSHAP (OpenReview zSKpJF2lTU 2026) — tool importance attribution；DFAH (arXiv:2601.15322) — determinism/faithfulness harness
- **Positioning**: 不卖大 leaderboard，卖对 success-only evaluation 的机制性反例

### Module B: Venue Simulation (EMNLP)
- R1 Diagnostic Critic: **Weak Accept** — "正面挑战 success-only paradigm"；担心若仅复现 Anchor ignore gap，main novelty 不足
- R2 Benchmark Specialist: **Borderline** — closed-model reproducibility 风险（API drift, 版本冻结）
- R3 Field Skeptic: **Weak Reject (main) / Borderline-Weak Accept (Findings)** — "只是 benchmark"风险，Anchor 太近
- **Meta**: Findings / Weak Accept；冲 Outstanding 需提"conversion laws"
- **Top risk**: Incrementality over Anchor + closed-model reproducibility + counterfactual replacement artifact

### Module C: Strategic Fit
- Longevity 7/10 — D/U/C taxonomy 持久，leaderboard 易过期
- Roadmap Viability 6/10 — 空间收窄
- Application Grounding 7/10 — eval methodology 影响大
- Execution Uniqueness 6/10 — Cluster B 有先发威胁
- Iteration Readiness 8/10 — 纯 inference

---

## Rank 2 (composite): IDEA-05 TrajFaith — Trajectory-Aware Faithfulness [PROCEED WITH CAUTION]

### Module A: Novelty
- **Score**: 7.2/10 (independent) / 6.5 (submodule)
- **Key differentiator**: faithfulness 从 document-answer alignment 升到 observation→future-action→final-answer sequential causal dependency
- **Closest prior**: CoRM-RAG (P20)、DFAH (新发现)、AgenTracer (新发现)
- **Positioning**: 提出 agent faithfulness formal taxonomy + 证明 metric 改变 model ranking/case interpretation/training reward diagnosis

### Module B: Venue Simulation (EMNLP)
- R1 NLP Empiricist: **Weak Accept if complete / Borderline otherwise** — 需 ≥2 benchmark × ≥2 size × multi-seed + human agreement
- R2 Method Reviewer: **Weak Accept** — 三类 metric 概念完整；需 formal assumptions + counterfactual validity tests
- R3 Analysis Reviewer: **Accept for Findings** — Analysis potential 极强
- **Meta**: Findings / Accept；冲 Outstanding 需 formal taxonomy + reproducible harness + human validation
- **Top risk**: Replay nondeterminism 误判 + 与 AgenTracer/TRACE/DFAH 边界 + 体量

### Module C: Strategic Fit
- Longevity 7/10
- Roadmap Viability 6/10 — Paper 2 与 AttribRL 重叠
- Application Grounding 6/10
- Execution Uniqueness 5/10 — Cluster D 有先发
- Iteration Readiness 7/10

---

## Rank 3 (composite, but #1 战略优先级): IDEA-01 AttribRL [PROCEED WITH CAUTION]

### Module A: Novelty
- **Score**: 7.5/10
- **Key differentiator**: 把"是否真用了 observation"从事后诊断转为可训练的 dense process reward
- **Closest prior**: TRM, ToolRL, CST (新发现), AgenTracer (新发现)
- **⚠️ Critical positioning**: CST (arXiv:2602.20710) 是最 close counterfactual-reward prior — 必须在 related work 用对照表明确"CST 优化 CoT predictability；AttribRL 优化 policy 对 observation 的 causal dependency"
- **Positioning**: 不 frame 成又一个 tool-use RL；frame 成 "post-training LMs to condition actions on environmental evidence"

### Module B: Venue Simulation (EMNLP)
- R1 Methodological Rigorist: **Borderline** — KL 高 ≠ "正确使用"；需 irrelevant-observation negative controls + reward hacking audit
- R2 Empirical Pragmatist: **Weak Reject (main) / Weak Accept (Findings)** — 只 Qwen-7B 单 size 不够；需补 14B 或 32B + 多 family + 3 seeds
- R3 Theoretician: **Borderline** — KL 是 interventional sensitivity 非 causal necessity；delayed utilization 未捕捉；garbage observation 破坏 support
- **Meta**: Findings / Weak Accept；冲 Outstanding 需 diagnosis→causal reward→multi-domain→failure taxonomy
- **Top risk**: Reward hacking + 单 size empirical shortfall + baseline 不全

### Module C: Strategic Fit
- Longevity 8/10 — 基本面问题
- Roadmap Viability 9/10 — Paper 1→4 自然延伸
- Application Grounding 8/10 — 直接影响 tool agent deployment
- Execution Uniqueness 5/10 — Cluster A/B 紧逼，timeline 紧
- Iteration Readiness 6/10 — pilot 1 周 + full 11d

---

## Detailed Per-Idea Reports

详见 outputs/SCREENING_REPORT.md。

---

## Next Steps

### Phase 4 idea-refine 顺序（按战略优先级）

1. **AttribRL (IDEA-01)** — **必须** refine。优先模式 `socratic-auto` 因方法层 reviewer 弱点（causal necessity / OOD artifact / delayed utilization）需深度对话补足。目标：从 Borderline 提升到 Accept (composite ≥7.5)。
2. **DiscUseBench (IDEA-02)** — **必须** refine。优先模式 `standard`（≤3 轮 review）因 benchmark paper 更多是协议与覆盖度问题。目标：从 Weak Accept 稳固到 Findings/Accept，可能争 Main。
3. **TrajFaith (IDEA-05)** — **defer**。建议作为：
   - 选项 A：独立 Findings 投稿（小尺度）
   - 选项 B：内嵌 DiscUseBench 的 evaluation 子模块（推荐——保护战略资源）
   - 在 Final Report 中标记为副线

### 前置补救（refine 前完成）
- AttribRL：在 related work 中补 CST 与 AgenTracer；引入 irrelevant-observation negative control 与 delayed-necessity reward 项
- DiscUseBench：引入 AgentSHAP、DFAH 对照表；冻结 closed-model 版本（GPT-5.5-2026-04, Claude Opus 4.7-2026-03, Gemini 3 Pro-2026-04 等）
- TrajFaith：决定独立投稿 vs 内嵌

### For PROCEED WITH CAUTION ideas（全部 3 个）
- 全部都未到 PROCEED 阈值 (7.0)；refine 后 re-score
- 失败 fallback：DiscUseBench (Plan B Findings) 至少有稳健落点
