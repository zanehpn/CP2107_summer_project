# AttribRL: Training Tool-Augmented LLM Agents to Causally Condition Actions on Tool Observations

> **Status**: Refined via 3-turn Socratic dialogue (gpt-5.5 xhigh, thread 019e17d4-...). Final score 7.62/10 (REVISE).
> **Venue**: EMNLP (Main + Findings 双投策略)
> **Plan B fallback**: DiscUseBench Findings paper (see separate refinement)

---

## 1. Problem Anchor (immutable)

**Bottom-line problem**: Tool-augmented LLM agents 在 trajectory 中 *发现* tool output 的能力远超 *使用* tool output 的能力（gpt-oss-120b 97.54% discovery vs 0.53% use, Engländer et al. 2026, arXiv:2604.17609）。Outcome RL（ToolRL/Search-R1/ReTool）与 step-level PRM（TRM/StepTool/AgentPRM）都不把 utilization gap 作为直接 reward 目标。

**Must-solve bottleneck**: 缺一个 *self-supervised、可训练、抗 reward hacking* 的 trajectory-level reward signal，使 RL 优化对象是 "action 真依赖 observation"。

**Non-goals**: 不提新 RL 算法（沿用 GRPO）；不进 multimodal；不解 tool retrieval；不训 frontier-scale。

**Constraints**: 2×4090 (48GB), API ≤ $30, 4 周, EMNLP main+Findings 双投，≥2 model size, ≥3 seeds。

**Success condition**: (i) Qwen2.5-7B 在 BFCL+ToolBench 上 conversion rate +≥30pp 且 verbatim copy < 5%；(ii) Qwen2.5-14B 复现趋势；(iii) AttribRL 在 V7 vs V1 (dense lexical shaping) / V8 (CST-style) 显著优胜 (paired t-test, Bonferroni m=8)；(iv) Plan B DiscUseBench 即使 RL fail 也可独立发表。

## 2. Skeleton (path from State A to State B)

- **State A**: outcome RL 足以教会 utilization；counterfactual reward 只用于静态 RM；attention attribution 不可靠；discovery-utilization gap 是诊断现象。
- **State B**: outcome 与 utilization 在统计上解耦；存在 self-supervised、抗 hacking、可训练的双源 utilization reward；该 reward 通过 GRPO 训练后关闭 gap；与 CST/AgenTracer/TRM 有清晰 axis-level 差异化。
- **Skeleton 5 步**:
  1. DiscUseBench grid 规模化 anchor 现象
  2. Causal D/U/C definition（counterfactual replacement）
  3. AttribRL 双源 reward（gated min mixture）
  4. Diagnostic counterfactual sensitivity guarantees（**不再 frame 为 Pearl backdoor identification**——回应 reviewer "Contribution Quality" critique）
  5. Multi-size × multi-seed × multi-bench 实证 + reward hacking audit + Plan B fallback

## 3. Technical Gap

| Prior | Object of optimization | Delta vs AttribRL |
|-------|------------------------|-------------------|
| ToolRL (NeurIPS '25) | Task outcome correctness | AttribRL: trajectory utilization |
| TRM (OpenReview LnBEASInVr) | Invocation correctness (learned PRM) | AttribRL: observation-action causality, no extra RM |
| StepTool / AgentPRM | Step progress / promise | AttribRL: causal grounding, self-supervised |
| **CST (arXiv:2602.20710)** | CoT predictability under counterfactual input | **AttribRL: policy action sensitivity to observation；不是 CoT 一致性** |
| AgenTracer | Post-hoc failure attribution | AttribRL: train-time reward, 非 diagnosis |
| CoRM-RAG / FaithfulRAG | Document-level RAG faithfulness | AttribRL: trajectory-level sequential dependency |
| Anchor P01 (Engländer 2026) | None (diagnostic) | AttribRL: first training-side response |

**Minimum intervention**: 在 GRPO + outcome reward 上新增 *一个* trajectory-step utilization reward $U_t$；不引入新 reward model parameters，不动 RL 算法。

## 4. Method — AttribRL

### 4.1 Complexity Budget

- **Frozen / reused**: Qwen2.5-7B/14B-Instruct, GRPO (veRL fork from Search-R1), 5 benchmark pipeline (BFCL/ToolBench/API-Bank/WebArena-Lite/SWE-Bench-Lite)
- **New (trajectory function only, no extra params)**: AttribRL reward computer
- **Excluded** (post-Socratic simplification): adaptive α schedule, delayed-KL as training signal (kept only as diagnostic), V9 AgenTracer post-hoc filter as training variant

### 4.2 Core Mechanism (post-revision: gated min mixture)

**Step 1 — Discovery gate** (evidence-availability judge, decoupled from action):
$$D_t = \text{LLMJudge}(o_t, \text{task}) \in \{0, 1\}$$
- Judge prompt v2: "Does this observation contain evidence that would enable solving the task? Ignore agent behavior."
- Two-judge gpt-5.5-mini consensus + 200 human spot-check (κ ≥ 0.75 target)

**Step 2 — Counterfactual KL** (sparse K=5, two-stage token target):
$$R_{cf,t}^{\text{raw}} = w_{\text{fn}} \cdot \text{KL}(p_\theta(a_t^{\text{fn-tok}}|h_t,o_t) \| p_\theta(a_t^{\text{fn-tok}}|h_t,\tilde{o}_t)) + w_{\text{arg}} \cdot \text{KL}(\text{at arg-slot token})$$
- $w_{\text{fn}}=0.3, w_{\text{arg}}=0.7$ (主权重在 argument-slot value position)
- $\tilde{o}_t$ = **matched distractor** (per-benchmark auto rule, see Appendix A; on-support validated via gpt-5.5-mini binary)
- **Null-distractor calibration**: $R_{cf,t}^{\text{net}} = \max(0, R_{cf,t}^{\text{matched}} - R_{cf,t}^{\text{null}})$
- **Absolute floor**: $R_{cf,t}^{\text{net}} := 0$ if $< \kappa = 0.05$ nats
- **Sigmoid normalization**: $\hat{R}_{cf,t} = \sigma((R_{cf,t}^{\text{net}} - \mu_{\text{batch}}) / T_{\text{batch}})$，$\mu_{\text{batch}}$ = median, $T_{\text{batch}}$ = IQR

**Step 3 — Lexical attribution** (every step, rule-based, evidence-role-aware):
$$R_{lex,t} = \text{rouge-L}(\text{EvidenceRoleSpan}(o_t, \text{task}), [a_t \| r_{t:t+w}])$$
- Span extractor 输出 (text, semantic_role) 并仅保留 `answer-evidence` role spans（排除 schema/metadata/boilerplate）
- Per-benchmark rules: BFCL → result/value fields; ToolBench → leaf string values; SWE-Bench → diff `+` lines + identifiers
- Transformed answer recognizer: numeric/unit/code 等 transformation 不要求 exact match，要求 obs 中 *原始数 / 单位 / identifier* 在 reasoning 任一步出现

**Step 4 — EMA propagation** (trajectory time axis only, no cross-trajectory leak):
$$R_{cf-\text{ema}}^{(t+1)} = \begin{cases} \gamma \cdot R_{cf-\text{ema}}^{(t)} + (1-\gamma) \cdot \hat{R}_{cf,t+1} & \text{probe step} \\ R_{cf-\text{ema}}^{(t)} & \text{non-probe (carry forward)} \end{cases}$$
- $\gamma = 0.7$，per-trajectory state，cold-start = 0.1

**Step 5 — Verbatim copy mask** (LCS-based, excludes evidence tokens):
$$\text{Mask}(t) = \mathbb{1}[\text{VerbatimCopyRatio}(a_t) < 0.7]$$
$$\text{VerbatimCopyRatio}(a_t) = \frac{|\text{LCS}(a_t, o_t) \cap \text{NonEvidenceTokens}(o_t)|}{|a_t|}$$
- Legitimate evidence pass-through (BFCL value-into-argument) 不计入 ratio（因 evidence token 不在 NonEvidenceTokens）

**Step 6 — Gated min mixture** (conjunctive gate, post-Socratic upgrade):
$$U_t = D_t \cdot \min(\hat{R}_{cf-\text{ema}}^{(t)}, R_{lex,t}) \cdot \text{Mask}(t)$$

**Step 7 — Outcome integration with λ schedule**:
$$R_{\text{total}}(\tau) = R_{\text{outcome}}(\tau) + \lambda(\text{epoch}) \cdot \sum_t U_t$$
- $\lambda$: linear warmup 0→0.5 (first 30% steps), anneal to 0.1 (no adaptive α — simplification per reviewer)

### 4.3 Why this is the smallest adequate mechanism

- 不动 GRPO，不动 backbone，不引入 reward model parameters
- 唯一新引入的可计算量 = $U_t$ 一个 trajectory function
- 双源 min-gate 是 *结构性* verbatim copy 防御（lemma 1.a），非 hyperparameter trick

### 4.4 Verbatim Copy Defense (Lemma 1.a, paper-formalizable)

设 $\pi_{\text{trivial-copy}}$ 为"复制 *任意* obs 的 *固定 schema position* token"的 policy（trivial copy mode）。则:
- $R_{lex,t}(\pi_{\text{trivial-copy}})$ 可能偶然 > 0 if span 命中 evidence
- $R_{cf,t}(\pi_{\text{trivial-copy}}) \to 0$ 因 action 不依赖 obs 内容差异
- $\text{Mask}(t) = 0$ 当 LCS-ratio ≥ 0.7
- → $U_t(\pi_{\text{trivial-copy}}) = 0$ ✓

**Caveat written into paper**: *semantic copy* (复制 current obs evidence span + transformation) **不是 hacking**——是 legitimate utilization，应获 reward。AttribRL 防御的是 trivial copy / schema boilerplate copy，不是 honest evidence pass-through。

### 4.5 Diagnostic Counterfactual Sensitivity (not Pearl backdoor)

**Reviewer "Contribution Quality" 建议**：降调 Pearl backdoor identifiability framing。

修订 framing：AttribRL 测量的是 *policy-level counterfactual sensitivity*，**不是** environment-level ATE。这是 *diagnostic* signal 而非 *causal identification*。

- Pearl/backdoor 仅在 Appendix B 作 motivation hint，不作 main paper claim
- Main claim 改为："$U_t$ 是 *self-supervised proxy* for utilization；其与 *human-annotated* utilization 的 correlation 在 200 sample 上 ≥ 0.7 (Spearman)；其与 conversion rate trend 协同上升"——empirical claim 而非 causal claim

## 5. Failure Modes & Diagnostics

| Failure mode | Detection | Mitigation |
|--------------|-----------|------------|
| Verbatim copy hacking | held-out 上 verbatim copy ratio > 10% | Mask threshold 0.7→0.5；提高 R_cf weight |
| Counterfactual breaks support | distractor task-validity score < 0.85 | 扩 distractor pool; K=5→K=10 |
| Delayed utilization missed (diagnostic only) | window-1 vs window-3 forward-KL trend flat | window W=5 diagnostic ablation |
| λ misset | accuracy drop ≥ 3pp | revert to λ=0.1 fixed (no curriculum) |
| Training diverge | GRPO KL > 0.05 | standard clipping + reward normalization |
| **Plan B fallback** | W2 pilot 不收敛 / V7 vs V2 无显著差异 | 直接发 DiscUseBench Findings paper |

## 6. Experiments — Trimmed Matrix (post-Feasibility simplification)

**主实验 (6 variants × 7B × 3 seeds × 5 benchmark)**:

| ID | Variant | R_cf | R_lex | Mask | D_t | λ schedule | Purpose |
|----|---------|------|-------|------|-----|------------|---------|
| V0 | Outcome only (Search-R1 reproducer) | × | × | × | × | n/a | strict baseline |
| V1 | Outcome + dense lexical shaping | × | ✓ | × | × | fixed | 排除 "any dense reward" |
| V2 | R_lex + Mask + D_t | × | ✓ | ✓ | ✓ | fixed | 排除 "lexical attribution alone" |
| V3 | R_cf + Mask + D_t | ✓ | × | ✓ | ✓ | fixed | 排除 "cf alone" |
| V7 | **AttribRL full (proposed)** | ✓ | ✓ | ✓ | ✓ | ✓ | this paper |
| V8 | CST-style predictability reward | predict CoT-cf | × | × | × | fixed | direct CST contrast |

**Ablation appendix (1-seed diagnostic on BFCL+ToolBench)**:
- V4 AttribRL no schedule
- V5 AttribRL no Mask
- V6 AttribRL no D_t

**Cross-size (BFCL+ToolBench only, V0/V2/V7 × 14B × 3 seeds)**: 验证 trend 复现

**Related baseline (no training)**:
- V9 AgenTracer-style post-hoc filter as related work comparison

**Statistical test**: V7 vs V1, V2, V3, V8 paired t-test on conversion rate (per benchmark × seed) with Bonferroni m=8.

**Stopping rule**: V7 在 BFCL+ToolBench 上不超过 V2 → 停止扩展至全部 benchmark，pivot 至 Plan B DiscUseBench paper。

## 7. Resource Budget

- **Compute**: 主实验 ~12 GPU-day (7B 6 variant × 2 epoch × 3 seed avg)；14B 子集 ~3 GPU-day；ablation pilot ~3 GPU-day = **总 18 GPU-day = 9 day on 2×4090** ✓
- **API**: gpt-5.5-mini judge × 14 model × 5 bench DiscUseBench ≈ 50K calls; ~$30 ✓
- **Span extractor**: rule-based per-benchmark Python；3 person-day implementation

## 8. 4-Week Timeline

| Week | Task | Deliverable |
|------|------|------------|
| W1 | DiscUseBench pipeline + LLM judge prompt v2 + counterfactual operator + span extractor + W1 pilot | gpt-oss-120b discovery > 90%, use < 5% 复现；Qwen-7B base conversion < 30% 验证 |
| W2 | AttribRL veRL fork + V0/V7 训练；6 variant 主表前 50% | 6 variants × 3 seeds BFCL+ToolBench |
| W3 | 全 5 benchmark × 6 variants 完成 + 14B cross-size + ablation appendix | 完整主表 + 14B trend |
| W4 | Reward hacking audit + judge calibration + paper writing + Plan B contingency | Paper draft v1 |

## 9. Anti-Drift Statement

Paper 主线坚持讲 **one core object: evidence-specific action sensitivity reward**。所有 simplification (drop adaptive α, delayed KL → appendix, V9 → related baseline) 是为收束主贡献叙事。如 reviewer 强加扩展 → push back, evidence anchor。

---

## Appendix A — Per-benchmark matched distractor rules

| Bench | Source pool | Reject filter | Pool size |
|-------|-------------|---------------|-----------|
| BFCL | same task_type 其他 function-call results | oracle answer extractor 失败 + token overlap < 0.3 | ~5000 |
| ToolBench | same API schema 不同 query 的 response | leaf string token overlap < 0.3 | ~3000 per API |
| API-Bank | same API category 其他 instance | answer slot 缺失 | ~1500 |
| WebArena-Lite | same domain 其他页面 HTML | required answer slot 不出现 | ~800 |
| SWE-Bench-Lite | 同 repo 其他 diff (length ∈ [0.7×, 1.4× orig]) | bug-fix identifier 不出现 | ~600 |

## Appendix B — Soft causal motivation (downgraded from main)

Pearl backdoor 提供 *motivation*：若 $H_t$ 是 sufficient adjustment set，then $\mathbb{E}[A_{t+1}|do(O_t=\tilde{o})] - \mathbb{E}[A_{t+1}|O_t=o] \approx \text{KL}(\pi(\cdot|h_t,o) \| \pi(\cdot|h_t,\tilde{o}))$。**但 AttribRL 不 claim identifiability**——只 claim sensitivity proxy。

## Appendix C — Object-of-optimization comparison

| Method | Optimizes |
|--------|-----------|
| Outcome RL (Search-R1) | $\mathbb{E}[r(\text{task success})]$ |
| TRM (PRM) | $\mathbb{E}[r(\text{invocation correct})]$ |
| CST (counterfactual sim) | $\mathbb{E}[p(\text{CoT}|cf-\text{input}) = p(\text{CoT}|\text{input})]$ |
| AgenTracer | post-hoc reweighting (no training reward) |
| **AttribRL** | $\mathbb{E}[\text{KL}(\pi(a|h,o)\|\pi(a|h,\tilde{o}))]$ at decision tokens, gated by $D_t \cdot \text{Mask}$ |

## Appendix D — Theory-Experiment Alignment Matrix (TE)

| Claim | Type | Protocol | Required Scale | Feasibility | Flag |
|---|---|---|---|---|---|
| AttribRL conversion +30pp on BFCL | Empirical hypothesis | V7 vs V1/V2/V3 paired t-test | 5 bench × 7B × 3 seeds | FEASIBLE | ✓ |
| Verbatim copy < 5% | Empirical hypothesis | held-out copy audit | held-out 1000 samples | FEASIBLE | ✓ |
| 14B trend reproducibility | Empirical hypothesis | V0/V2/V7 × 14B × 3 seeds on BFCL+ToolBench | 2 sizes × 3 seeds | FEASIBLE | ✓ |
| Sensitivity-utilization correlation ≥ 0.7 | Empirical hypothesis | $U_t$ vs human-annotated utilization on 200 sample, Spearman | 200 human-annot | FEASIBLE | ✓ |
| GRPO mixed-reward convergence | Convergence bound | training curves × 3 seeds × 3 LR | 3 seeds × 3 LR | FEASIBLE WITH CAVEATS (仅 Qwen-7B 完整) | ⚠️ |
| Pearl backdoor identifiability | Causal | (downgrade to soft motivation only in Appendix B) | — | NOT FEASIBLE → downgraded | ⚠️ resolved by claim weakening |

## Appendix E — Plan B contingency

若 W2 pilot 显示 AttribRL 不收敛 OR V7 vs V2 在 BFCL+ToolBench 上无显著差异，立即 pivot 到 DiscUseBench Findings paper（已有 instrumentation；见 /idea-refine for DiscUseBench, separate run）。Plan B paper 仍 fully publishable (composite 6.79, Findings/Weak Accept secured)。
