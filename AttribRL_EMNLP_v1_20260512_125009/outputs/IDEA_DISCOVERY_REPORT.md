# Idea Discovery Report — AgentRead

**研究方向**: AgentRead: Diagnosing and Closing the Discovery-to-Utilization Gap in Tool-Augmented LLMs
**目标 Venue**: EMNLP（Main + Findings 双投策略）
**Anchor Design Doc**: `/Users/zpy/LLM_project/idea_paper/boardSearch/outputs/archive_2026-05-11_agentread-design_done/DESIGN_DOC_AgentRead.md`
**Pipeline 执行日期**: 2026-05-11 23:50 → 2026-05-12 01:55 (≈2 hr)
**Reviewer Model**: gpt-5.5 (xhigh reasoning effort) via Codex MCP

---

## Executive Summary

围绕 anchor 事实 **gpt-oss-120b 97.54% discovery vs 0.53% use** (Engländer et al. arXiv:2604.17609)，本管道从 35-paper literature survey 出发，经 14-critique landscape critique、11 raw ideas → 6 filtered → 3 screened → 2 refined。

**最终推荐**：
- **Main paper (HIGH risk / HIGH reward)**: **AttribRL** — counterfactual-lexical 双源 attribution reward + GRPO (gated min mixture)，score 7.62/10 (REVISE → Findings/Accept potential, Main 取决实证)
- **Plan B safety net (LOW risk)**: **DiscUseBench** — Diagnosing Evidence-to-Action Conversion in Language Agents，8 model × 4 benchmark + 2 pre-registered conversion laws + linguistic interpretability，score 6.65 → est 7.2 after post-review simplification (Findings/Weak Accept secured)

**关键 timeline 风险**：Cluster B (Cohere/Edinburgh: Engländer, Althammer, Sherborne, Üstün, Gallé) 预测在 6-9 个月内发表 training-side fix 的 follow-up，AttribRL 必须抢节奏。建议立即启动 W1 pilot (gpt-oss-120b 复现 + Qwen-7B base conversion 验证)。

**两条 paper 互为保险**：
1. AttribRL 训练成功 → Main paper, fallback Findings
2. AttribRL W2 pilot 失败 → 立即 pivot DiscUseBench Findings (instrumentation 在 AttribRL W1 已完成)
3. DiscUseBench H1/H2 失败 → 重定位为 "Negative Result Diagnostic" Findings

总计 4 周可执行 (per anchor design doc)，2×4090 算力 + $30 API 双 paper 共享 instrumentation。

---

## Phase 1: Literature Survey (35 papers, 6 themes, 12 gaps)

**Coverage**:
- D1 Tool-use RL (10 papers): TRM, ToolRL (NeurIPS '25), Search-R1, R1-Searcher/++, ReTool, StepTool, iTool, Tool Zero, AgentPRM
- D2 Agent diagnostics (4): **Anchor P01** "Agents Explore but Agents Ignore" (97.54% / 0.53%), TRACE (WWW '26), DORA Explorer, SciCrafter
- D3 Context utilization & RAG faithfulness (6): Sufficient Context, Lost in Middle, FaithfulRAG, CoRM-RAG, FaithEval, Mindful-RAG
- D4 Counterfactual & attention attribution (6): Causal Rewards, CF Multimodal, Reward Hacking Benchmark, Attention Bias Optimization (NeurIPS '25), LoGra, Reward Shaping
- D5 Benchmarks & MCP (9): BFCL V4, ToolBench, API-Bank, Terminal-Bench (ICLR '26), AppWorld, MCP-Zero, RAG-MCP

**关键 gap (HIGH confidence)**:
- G1 14-model × 5-benchmark D/U/C grid 缺位
- G2 现有 RL 无 utilization reward
- G3 Counterfactual reward 未上 tool agent trajectory training
- G7 Lexical + counterfactual 组合作 verbatim copy defense 未被尝试
- G10 RAG counterfactual perturbation 未上 trajectory
- G12 Plan B diagnostic-only path 开放

**Phase 3 发现的新 priors（lit-survey 漏掉）**:
- **CST (Counterfactual Simulation Training, arXiv:2602.20710)** — AttribRL 必须显式区分 (CoT predictability vs action causal dependency)
- **AgenTracer (OpenReview l05DseqvuD)** — counterfactual replay for failure attribution
- **AgentSHAP (OpenReview zSKpJF2lTU)** — tool importance attribution
- **DFAH (arXiv:2601.15322)** — agent determinism/faithfulness harness

详见 `outputs/LANDSCAPE.md` 与 `outputs/LANDSCAPE.json`。

---

## Phase 2: Idea Generation (14 critique entries → 11 raw → 6 filtered)

**14 critique manifest** (4 维度: 5 unverified assumption + 3 incorrect generalization + 4 experimental flaw + 3 cross-domain misfit) 详见 `outputs/CRITICAL_ANALYSIS.md`。

**11 raw ideas → 6 surviving (He Score ≥12/20)**:

| Rank | Idea | He Score | Risk |
|------|------|---------|------|
| 1 | IDEA-01 AttribRL | 19/20 | HIGH |
| 2 | IDEA-02 DiscUseBench | 17/20 | LOW |
| 3 | IDEA-05 TrajFaith | 15/20 | MEDIUM |
| 4 | IDEA-06 LeakProof-Use | 15/20 | LOW |
| 5 | IDEA-03 Causal AttribRL | 14/20 | HIGH |
| 6 | IDEA-04 Tool Recall→Use | 14/20 | MEDIUM |

**5 个 eliminated 被建议作为 IDEA-01/02 的 subsections**: IDEA-07 judge audit, IDEA-08 mechanism, IDEA-09 obs quality, IDEA-10 PRM audit, IDEA-11 λ schedule。详见 `outputs/IDEAS_FILTERED.md`。

---

## Phase 3: EMNLP Screening (3 ideas screened)

Composite (Novelty 0.25 + Venue 0.35 + Strategic 0.20 + Feasibility 0.20):

| Composite Rank | Idea | Novelty | Venue | Strategic | Feasibility | Composite | Verdict |
|----|------|--------|------|-----------|-------------|-----------|---------|
| 1 | IDEA-02 DiscUseBench | 6.8 | 5.5 | 6.8 | 9.0 | **6.79** | PROCEED WITH CAUTION |
| 2 | IDEA-05 TrajFaith | 7.2 | 6.2 | 6.2 | 7.5 | **6.71** | PROCEED WITH CAUTION |
| 3 | IDEA-01 AttribRL | 7.5 | 5.5 | 7.2 | 6.5 | **6.54** | PROCEED WITH CAUTION |

**战略覆写**：AttribRL 复合分数最低但 anchor design doc 优先级 #1；TrajFaith 建议内嵌入 DiscUseBench (后已采纳，详见 DiscUseBench refinement §4)。

**Reviewer 关键 risks**:
- AttribRL: KL 高 ≠ causal necessity；garbage observation 破坏 support；delayed utilization 未捕捉
- DiscUseBench: incrementality over Anchor P01；closed-model API drift；counterfactual replacement artifact
- TrajFaith: replay nondeterminism；独立 paper 体量；与 AgenTracer/DFAH 边界

详见 `outputs/SCREENING_REPORT.md` (3 reviewer × 3 idea 完整 venue simulation) 与 `outputs/SCREENING_RANKED.md`。

---

## Phase 4: Deep Refinement

### Phase 4a: AttribRL — Socratic 3-turn Dialogue (gpt-5.5 xhigh)

**Score Evolution**: 6.54 (screening) → **7.62** (after 3 Socratic turns + post-revisions)
**Target**: ≥7.5 → **achieved** ✓ (+1.08)
**Verdict**: REVISE (EMNLP Findings/Accept 强潜力，Main 取决实证)

**关键 method upgrades through Socratic dialogue**:
1. **Turn 0 Q5 → gated min mixture**: 替换 plain weighted sum `αR_cf + (1-α)R_lex` → `min(R_cf-ema, R_lex)`（conjunctive gate, 结构性 verbatim copy 防御）
2. **Turn 1 Q1-Q3 → quantitative formalism**:
   - R_cf-ema trajectory-time EMA (carry-forward on non-probe step)
   - Sigmoid normalization + absolute floor κ + null-distractor calibration（防 KL/rouge-L 量纲单边截断）
   - Lemma 1.a refined for trivial-copy only（semantic copy 显式声明为 legitimate utilization）
3. **Turn 2 Q3-Q4 → mechanism precision**:
   - Two-stage first-decision-token + argument-slot KL with $w_{\text{arg}}=0.7$
   - D_t judge prompt v2: "evidence-availability" (decoupled from agent action)
4. **Turn 2 Q5 → 10-variant ablation matrix** (post-revision trimmed to 6 main + 3 appendix + 1 related): V0-V9 包含 CST/AgenTracer-style baselines for explicit object-of-optimization isolation
5. **Post-revisions (after final scoring)**:
   - Drop adaptive α (Simplification)
   - Delayed KL → diagnostic only (Simplification)
   - V9 AgenTracer → related baseline 非 training variant (Simplification)
   - Pearl backdoor framing → soft motivation only (Modernization)
   - 14B 仅子集 (V0/V2/V7 × 2 bench × 3 seed) (Feasibility)

**剩余 weaknesses**:
- KL → ATE 形式 framing 弱（已 downgrade 至 diagnostic counterfactual sensitivity）
- 14B 仅子集
- W1 pilot 必过（gpt-oss-120b 复现 + Qwen-7B base < 30%）

详见 `refine-logs/FINAL_PROPOSAL.md` (canonical clean version) + `refine-logs/REFINEMENT_REPORT.md` + `refine-logs/REVIEW_SUMMARY.md` + `refine-logs/socratic-turn-{0,1,2}-{questions,answers}.md` + `refine-logs/socratic-final-review.md`。

### Phase 4b: DiscUseBench — Standard Review + Post-Review Simplifications

**Score**: 6.65 (REVISE) → estimated 7.2 post-simplification (Findings/Weak Accept-Accept)

**Key post-review revisions**:
1. **Use definition 3-layer** (replace closed-model KL dependency):
   - Layer 1: ActionSchemaDelta on fixed-seed cf rollout
   - Layer 2: BenchmarkScore delta (original correct + cf wrong)
   - Layer 3: LLM-judge adjudication (validator only)
2. **Feasibility trim**: 14 × 5 → **8 × 4** (drop 4 RL ckpts; drop WebArena-Lite to appendix)
3. **NLP 落点 strengthening**: Re-title to "Diagnosing Evidence-to-Action Conversion in Language Agents"; conversion laws explicitly linked to linguistic variables (span_density / evidence_locality / instruction-evidence_conflict)
4. **Pre-registered hypotheses** (not "findings"):
   - H1 (observation salience law): mixed-effects logit `Conversion ~ span_density + model_size + benchmark + (1|task)`
   - H2 (parametric leakage law): use rate inflated ≥15pp for models > 70B without fresh-fact controls
5. **TrajFaith merger**: IDEA-05 内嵌为 secondary "RAG-faithfulness disagreement matrix" 0.5-page section（避免独立 paper 体量问题）
6. **AttribRL 不作 baseline**（保护 main paper 贡献边界）

**Budget verification**: 8 × 4 × 200 instances × 5 distractor types = ~$25 API + ~26 GPU-hr ✓ within $30/2×4090

**剩余 weaknesses**:
- Distractor design W1 pilot validate 是 gating
- Closed-model API drift（EMNLP ARR rolling review 减半此风险）
- H1/H2 是 predictions 而非 facts

详见 `refine-logs/FINAL_PROPOSAL_DiscUseBench.md`。

---

## Strategic Recommendations

### 立即可执行（W1, 5-7 day）

**Pilot 必过 gate**:
1. **gpt-oss-120b on Terminal-Bench + AppWorld**: 复现 anchor 97.54% / 0.53% 数字（±5pp tolerance）
2. **Qwen-7B-Instruct on BFCL**: base conversion rate < 30%（验证 gap 在 7B 也存在）
3. **Distractor pool quality**: 5 distractor types per 4 benchmark, task-validity score > 0.85 (gpt-5.5-mini binary)
4. **Span extractor coverage**: 100-task manual eval, ≥0.80 recall on answer-evidence spans

**两条 paper 共享 W1 instrumentation** — pilot 通过后两条 paper 可并行推进。

### Pivot 决策树

```
W1 pilot 通过?
├── YES → W2 launch AttribRL (Main paper)
│         └── W3 mid-check: V7 vs V2 on BFCL+ToolBench 显著?
│             ├── YES → W4 full + 14B + writing → submit AttribRL Main + DiscUseBench Findings
│             └── NO → freeze AttribRL at current results → submit Findings only (with negative result section)
│                      + accelerate DiscUseBench Main
└── NO (pilot anchor 失败) → fallback to DiscUseBench Findings paper as Plan B
                                + report as "Anchor P01 数据无法规模化复现" diagnostic
```

### Long-term Roadmap (Paper 1 → Paper 4 series)

| # | Paper | Venue | Timeline |
|---|-------|-------|----------|
| 1 | AttribRL (this work) | EMNLP 2026 Main/Findings | W1-W4 2026-05 → submit W4 |
| 2 | DiscUseBench (this work) | EMNLP 2026 Findings | W1-W4 parallel → submit W4 |
| 3 | Causal AttribRL (IDEA-03) | NeurIPS 2026 | Q3-Q4 2026 (deepen Pearl framing) |
| 4 | Tool Recall→Use joint (IDEA-04) | ICLR 2027 | Q1-Q2 2027 (上下游闭环) |

---

## Output Files Manifest

```
outputs/
├── LANDSCAPE.md / LANDSCAPE.json              [Phase 1: 35 papers, 6 themes, 12 gaps]
├── CRITICAL_ANALYSIS.md                       [Phase 2a: 14 critique entries]
├── IDEAS_RAW.md                               [Phase 2b: 11 raw ideas]
├── IDEAS_FILTERED.md                          [Phase 2 filter: 6 surviving]
├── SCREENING_REPORT.md                        [Phase 3: 3 reviewer × 3 idea full simulation]
├── SCREENING_RANKED.md                        [Phase 3: composite ranking]
├── IDEA_DISCOVERY_REPORT.md                   [Phase 5: this file]
├── PIPELINE_LOG.md                            [autonomous decision log]
└── PIPELINE_STATE.json                        [checkpoint state]

refine-logs/
├── skeleton.md                                [Phase 4a: AttribRL skeleton]
├── round-0-initial-proposal.md                [Phase 4a: initial proposal]
├── socratic-turn-{0,1,2}-questions.md         [Phase 4a: Socratic GPT questions]
├── socratic-turn-{0,1,2}-answers.md           [Phase 4a: Claude answers]
├── socratic-final-review.md                   [Phase 4a: final 7-dim score]
├── score-history.md                           [Phase 4a: score evolution table]
├── FINAL_PROPOSAL.md                          [Phase 4a: AttribRL canonical final]
├── FINAL_PROPOSAL_DiscUseBench.md             [Phase 4b: DiscUseBench canonical final]
├── REVIEW_SUMMARY.md                          [Phase 4a: AttribRL round-by-round resolution log]
├── REFINEMENT_REPORT.md                       [Phase 4a: full report with raw reviewer responses]
└── discusebench-skeleton.md                   [Phase 4b: DiscUseBench skeleton]
```

---

## Codex MCP Thread IDs (for follow-up)

- Phase 2 critique + idea generation: `019e17c2-d657-7c51-9a97-a817899d3399`
- Phase 3 screening (3-idea venue simulation): `019e17cb-7b3e-7ea0-97ea-4f94759c0fb6`
- Phase 4a AttribRL Socratic dialogue: `019e17d4-3da8-7633-97d7-624fdcc569cd`
- Phase 4b DiscUseBench review: `019e17e5-82b1-7951-85a8-98770be22530`

---

## 致 user 决策点

1. **是否启动 W1 pilot 立即执行 AttribRL**？(高 longevity + 高 roadmap + Cluster B 6-9 个月时间窗)
2. **是否并行 DiscUseBench 作为 Plan B**？(LOW risk + 共享 W1 instrumentation + Findings 兜底)
3. **AttribRL Round 2 refinement**（目标 9.0/10 → READY） 是否在 W1 pilot 数据回来后启动？

所有 4 weeks 路径都 fit 2×4090 + $30 budget + 共享 instrumentation 基础。
