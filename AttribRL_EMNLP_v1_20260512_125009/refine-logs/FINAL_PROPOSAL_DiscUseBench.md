# Diagnosing Evidence-to-Action Conversion in Language Agents (DiscUseBench)

> **Plan B safety-net paper for AttribRL** — independently publishable EMNLP Findings target.
> **Refined via 1 standard codex review (gpt-5.5 xhigh, thread 019e17e5-...). Score 6.65/10, REVISE.**
> **Post-review simplifications applied below.**

---

## 1. Problem Anchor (immutable)

**Bottom-line problem**: Anchor P01 (Engländer 2026, arXiv:2604.17609) 在 2 benchmark × 4 model 上展示 gpt-oss-120b discovery 97.54% vs use 0.53%。缺规模化 cross-model evidence + 缺 *causal* use definition + 缺 reproducibility infrastructure + 缺 linguistic conversion laws。

**Must-solve bottleneck**: 8 model × 4 benchmark D/U/C 三维表 + causal counterfactual use definition + version frozen reproducibility + ≥2 pre-registered conversion laws **linked to linguistic factors**（不是 leaderboard）。

**Non-goals**: 不训模型；不提新 RL；不解 tool retrieval；不评 multimodal；**不收纳 AttribRL 作为 own baseline**（避免主贡献漂移）。

**Constraints**: 2×4090, $30 API, 4 weeks（trim from 14×5 to 8×4）, EMNLP Findings.

**Success condition**: (i) D/U/C 三维 cross-model stable；(ii) ≥2 pre-registered conversion laws 在 mixed-effects regression 中显著；(iii) RAG-faithfulness vs trajectory-faithfulness disagree on ≥25% successful trajectories；(iv) reproducibility checklist 全填 + dockerized harness + raw trajectories released。

## 2. Skeleton

- **A** → **B** path:
  1. Causal D/U/C definition (decision-level perturbation + task-score delta, no closed-model KL dependency)
  2. 8 × 4 grid with frozen API versions + dockerized open-model harness
  3. Pre-registered hypotheses (2 conversion laws as falsifiable predictions)
  4. Judge calibration κ ≥ 0.75 + bootstrap CI + raw trajectory release
  5. Linguistic analysis: observation salience / evidence locality / instruction-evidence conflict 等可解释 NLP variables driving conversion

## 3. Method

### 3.1 Causal Use definition (post-review revision: 3-layer)

**Layer 1 (primary, reproducible across closed/open models)**:
$$\text{Use}_t^{(1)} = \mathbb{1}[\text{ActionSchemaDelta}(a_t^{\text{orig}}, a_t^{\text{cf}}) > 0 \,\text{ or }\, \text{AnswerCorrectness}(\tau^{\text{orig}}) \ne \text{AnswerCorrectness}(\tau^{\text{cf}})]$$

- 固定 decoding seed + temperature 0
- 替换 $o_t$ → $\tilde{o}_t$ (matched distractor)，重跑 trajectory
- ActionSchemaDelta: function name / argument key / argument value 任一改变即 1
- **不依赖 token-level KL**，闭源模型可用 ✓

**Layer 2 (task-score delta)**:
$$\text{Use}_t^{(2)} = \mathbb{1}[\text{BenchmarkScore}(\tau^{\text{orig}}) > \text{BenchmarkScore}(\tau^{\text{cf}})]$$

- 原轨迹正确 + 扰动轨迹错误 → strong use signal
- 用 benchmark 自带的 scoring (BFCL AST exec match, ToolBench LLM judge w/ ground-truth)

**Layer 3 (adjudication, validator only)**:
$$\text{Use}_t^{(3)} = \text{LLMJudge}_\text{adj}(\tau^{\text{orig}}, \tau^{\text{cf}}, o_t, \tilde{o}_t, \text{task})$$

- gpt-5.5-mini binary：是否 observation difference 解释 trajectory difference
- **仅作 Layer 1+2 的 validator**，不作 primary metric

**Combined**: Use$_t$ = Use$_t^{(1)} \lor$ Use$_t^{(2)}$；Use$_t^{(3)}$ 用于 reliability audit。

### 3.2 Matched distractor per benchmark (5 distractor types)

| Type | Definition |
|------|------------|
| Field replacement | 同 schema 不同 task 的 result value field swap |
| Answer-swap | 把答案数值/字符串替换为同 type 的不同值 |
| Fresh-fact swap | 替换为 synthetic fresh fact (parametric leakage isolation) |
| Irrelevant-but-format-matched | 同 schema length 但 task-irrelevant content |
| Adversarial distractor | 同 task type 但 confusable answer |

**Per benchmark rule table**:
| Bench | Field repl | Answer swap | Fresh-fact | Irrelevant-fmt | Adversarial |
|-------|------------|-------------|------------|----------------|-------------|
| BFCL | function arg field | numeric value | synthetic API | other-task result | wrong-type result |
| ToolBench | API response leaf | string value | synthetic API | different query | wrong-domain API |
| API-Bank | API category | response value | synthetic instance | unrelated call | wrong-arg call |
| SWE-Bench-Lite | diff `+` lines | identifier swap | new repo synthetic | unrelated diff | similar-bug-diff |

### 3.3 Coverage (post-Feasibility simplification: 8 × 4)

**Models** (8, drop 4 RL ckpts including AttribRL):
- Closed (2): GPT-5.5-2026-04, Claude Opus 4.7-2026-03
- Open large (2): Llama 4 Maverick, DeepSeek V3
- Open mid (2): Qwen2.5-7B, Llama 3.1 8B
- RL-trained baselines (2): Search-R1, ToolRL

**Benchmarks** (4, drop WebArena-Lite to appendix):
- BFCL / ToolBench / API-Bank / SWE-Bench-Lite

**Sample budget**: 200 instances × 4 bench × 8 model = 6400 trajectories × ~10 step each = 64K trajectory-steps × 5 distractor types = 320K cf rollouts。Per-step open-model inference ~0.3 sec on 4090; ~26 GPU-hr。Closed-model API: 200×4×2 = 1600 queries per cf type × 5 types ≈ 8K calls × $0.0015 ≈ $12 closed; gpt-5.5-mini judge for Use$^{(3)}$ + Discovery: 200×4×8×~5 obs × 2 (D + adj) = 64K calls × $0.00015 ≈ $10. **Total $25 ≈ within budget**.

### 3.4 Pre-registered conversion laws (post-review: hypotheses not findings)

**Hypothesis H1 (observation salience law)**:
> conversion rate $\propto$ span_density (key evidence tokens / total obs tokens), tested via mixed-effects logistic regression: $\text{logit}(\text{Conversion}) \sim \text{span\_density} + \text{model\_size} + \text{benchmark} + (1|\text{task})$

**Hypothesis H2 (parametric leakage law)**:
> $\text{Use}^{\text{standard}}(\text{model}) - \text{Use}^{\text{leak-controlled}}(\text{model}) \ge 15\text{pp}$ for models with > 70B parameters; ≤ 5pp for models < 13B

Statistical test: paired t-test on model-level use delta；effect size > 0.5 Cohen's d。

### 3.5 Mechanism decomposition (kept appendix, post-Simplification)

3 failure modes clustered via observation features (pre-tool confidence / output entropy / span_density / position / copy_rate / self-correction freq)：
- M1: Prior overconfidence
- M2: Observation unreadability
- M3: Delayed integration failure

Validation: cluster stability AMI > 0.5 cross-model；bootstrap stability.

### 3.6 Reproducibility & Auditing

- **Frozen API endpoints**: model_id + date + temperature + system_prompt + tool_schema_hash (released as JSON config)
- **Dockerized harness**: BFCL/ToolBench/API-Bank/SWE-Bench-Lite Docker images released
- **Raw trajectory release**: full 8×4×200 trajectories with prompt_hash, tool_schema_hash, observation_hash, action_JSON, scores, perturbation_type
- **Bootstrap CI**: 1000 resamples on every D/U/C cell
- **Judge calibration**: 200 human-annotated samples Cohen's κ ≥ 0.75 + confusion matrix released
- **Distractor validity audit**: distractor task-validity score distribution per type

## 4. Frontier Leverage (post-review modernization)

- **TrajFaith merger**: IDEA-05 内嵌为 secondary analysis section "RAG-faithfulness vs trajectory-faithfulness disagreement matrix"，约 0.5 page；不抢主线
- **Mixed-effects regression** for salience law (linguistic interpretability)
- **Stratified sampling**: easy/hard, fresh/stale, short/long observation, high/low salience per benchmark
- **Pre-registration**: H1/H2 在 W1 pilot 后 timestamp 公开（OSF / AspirinGuard），避免 post-hoc 包装

## 5. Anti-Drift Statement

Paper 中心：**Diagnosing Evidence-to-Action Conversion in Language Agents**，**不是 benchmark leaderboard**。每张表都服务 2 个 hypothesis；每个 conversion law 都要有 *linguistic* explanatory variable（observation salience, evidence locality, instruction-evidence conflict）；闭源模型版本必须包含 actual `model_id` + API date metadata，不写未来 model name。AttribRL 不作 baseline（保护 main contribution boundary）。

## 6. Failure Modes & Diagnostics

| Mode | Detection | Mitigation |
|------|-----------|------------|
| Closed-model API drift | reproducibility check via dated re-query | release frozen API call cache (raw response) |
| Use$^{(1)}$ false-negative (action schema unchanged but downstream changes) | Layer 2 task-score delta catches | combined Use criterion |
| Distractor on-support violation | distractor validity score < 0.85 | expand pool / drop distractor type |
| Judge κ < 0.75 | human spot-check on 200 samples | iterate prompt + re-train judge baseline |
| H1/H2 fail | report negative result | EMNLP Findings 接受 negative diagnostic |

## 7. Theory-Experiment Alignment Matrix

| Claim | Type | Protocol | Scale | Feasibility | Flag |
|-------|------|----------|-------|-------------|------|
| H1 salience law | Empirical hypothesis | mixed-effects logit, 1000 trajectories | 8 models × 4 bench × 200 | FEASIBLE | ✓ |
| H2 parametric leakage law | Empirical hypothesis | paired t-test on standard vs leak-controlled use | 8 models × 100 controlled instances | FEASIBLE | ✓ |
| Causal Use$^{(1)+(2)}$ vs LLM-judge Use | Empirical hypothesis | confusion matrix on 200 sample | FEASIBLE | ✓ |
| Judge κ ≥ 0.75 | Inter-rater reliability | Cohen's κ on 200 human-annot | FEASIBLE | ✓ |
| 3-mode cluster stability (AMI > 0.5) | Cluster validity | bootstrap × model family | FEASIBLE WITH CAVEATS | ⚠️ partial (less stable for closed-model features) |

## 8. 4-Week Timeline

| Week | Task | Deliverable |
|------|------|------------|
| W1 | Dockerize harness + distractor construction per bench + judge prompt v2 + pilot 100 instances | gpt-oss-120b 97.54% 复现 + distractor validity > 0.85 |
| W2 | Full 8×4×200 cf rollouts on open models + closed model API calls + Use Layer 1/2 scoring | D/U/C 三维表 raw |
| W3 | Judge Layer 3 + κ calibration + mixed-effects regression for H1/H2 + 3-mode clustering | H1/H2 statistical results + appendix mechanism analysis |
| W4 | Reproducibility checklist + raw trajectory release + paper writing | EMNLP Findings draft v1 |

## 9. Plan B for Plan B (i.e. AttribRL fails + DiscUseBench finds null result)

若 H1 不显著 OR H2 不显著 → 重定位为 "Discovery-to-Use Gap Across Tool Agents: A Negative Result Diagnostic"，EMNLP Findings 仍可接受 (analysis > SOTA 文化)。

---

## Score Snapshot
- Initial (Phase 3 screening): composite 6.79
- After 1 codex review (Phase 4b): 6.65 (REVISE, slight drop due to feasibility/specificity catches)
- **After post-review simplifications (this doc)**: estimated 7.2-7.5 (Findings/Weak Accept-Accept range)

剩余 weakness：
1. Distractor design 仍可能被 challenge（W1 pilot validate is critical）
2. Closed-model API drift 是 venue-specific 风险（EMNLP rolling review 减半）
3. H1/H2 是 prediction 而非已知 fact；需 W1 pilot 强支持
