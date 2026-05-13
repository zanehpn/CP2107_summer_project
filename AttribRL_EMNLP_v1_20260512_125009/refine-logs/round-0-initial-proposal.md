# Research Proposal: AttribRL — Counterfactual-Lexical Attribution Rewards for Tool-Augmented Agents

## Problem Anchor

- **Bottom-line problem**: 当前 tool-augmented LLM agents 在 trajectory 中 *发现* tool output 的能力远超 *使用* tool output 的能力——anchor 事实：gpt-oss-120b 97.54% discovery vs 0.53% use（Engländer et al. arXiv:2604.17609）。outcome RL（ToolRL/Search-R1/ReTool）和 step-level PRM（TRM/StepTool/AgentPRM）都没有把这个 utilization gap 当作直接 reward 目标。
- **Must-solve bottleneck**: 缺一个 *self-supervised、可训练、抗 reward hacking* 的 trajectory-level reward signal，使 RL 优化的对象是 "action 真的依赖 observation"，而非 "task 成功" 或 "tool call 正确"。
- **Non-goals**: 不试图 (a) 提出新 RL 算法（沿用 GRPO）；(b) 进入 multimodal / web agent（先解决 text-only tool agents）；(c) 涉及 tool retrieval（上游问题）；(d) 训练 frontier-scale models（限 ≤14B）。
- **Constraints**: 算力 2×4090 (48GB)；API 预算 ≈ $30；4 周可执行；EMNLP main + Findings 双投策略；至少 ≥2 model size 与 ≥3 seeds（来自 R2 反馈）。
- **Success condition**: (i) Qwen2.5-7B 在 BFCL/ToolBench/API-Bank/WebArena-Lite/SWE-Bench-Lite 5 个 benchmark 上 conversion rate 从 ~18% → ≥55%；同时 final task success +≥5pp；(ii) Qwen2.5-14B 上稳定复现 trend；(iii) verbatim copy ratio < 5%（reward hacking 不发生）；(iv) Anchor P01 现象在 14-model × 5-benchmark grid 上规模化复现（Plan B 独立可发 DiscUseBench Findings paper）。

## Skeleton

- **State A**: outcome RL 足以教会 utilization；counterfactual reward 只在静态 RM 场景；attention attribution 不可靠；discovery-utilization gap 是诊断现象。
- **State B**: outcome 与 utilization 在统计上解耦；存在 self-supervised、抗 hacking、可训练的双源 utilization reward；该 reward 通过 GRPO 训练后关闭 gap；与 CST/AgenTracer/TRM 有清晰 axis-level 差异化。
- **Skeleton Path**:
  - Step 1 — DiscUseBench: 14 model × 5 bench 规模化 anchor 现象
  - Step 2 — Causal Discovery/Use/Conversion definition: 用 counterfactual observation replacement 严格化"use"
  - Step 3 — AttribRL 双源 reward: counterfactual KL + lexical span + λ schedule
  - Step 4 — Theoretical guarantees: Pearl backdoor identifiability + verbatim-copy negative control + Pareto schedule analysis
  - Step 5 — Multi-size × multi-seed × multi-benchmark 实证 + reward hacking audit + fallback Plan B

## Technical Gap

**当前方法在哪里失败**：
1. Outcome RL（ToolRL/Search-R1）：模型可能通过 prior / shortcut / format imitation 答对，与 observation 无因果依赖。Reward 信号无法区分"用了 obs"和"shortcut"。
2. Step-level PRM（TRM/StepTool/AgentPRM）：奖励 invocation correctness 或 step progress，不奖励"action 因 observation 改变"。**TRM 的 3B reward model 可学会一个 invocation-format-correct 但 observation-irrelevant 的 step pattern**。
3. CST（arXiv:2602.20710）：是最 close 的 counterfactual-reward prior，但其优化目标是 **CoT predictability**——让 simulator 在 counterfactual 输入下也能预测原 CoT。AttribRL 优化目标是 **policy action sensitivity**——让 action 在 observation 替换后改变。Object of optimization 完全不同。
4. AgenTracer：用 counterfactual replay 做 *post-hoc failure attribution*，不是 training reward。
5. RAG faithfulness (CoRM-RAG/Mindful-RAG)：document-level counterfactual，不是 trajectory-level sequential dependency。

**为什么 naive fix 不够**：
- "更多 outcome rollouts"：只放大 shortcut signal；不会教会 utilization
- "更强 PRM"：仍然是 progress proxy，不是 causal grounding
- "Prompt engineering"：可短期改善 conversion，但训练目标若不变，gap 会回弹

**最小可行 intervention**：在已有 GRPO + outcome reward 上，新增一个 **trajectory-step level utilization reward**，由两路构成：
- 路 1（counterfactual, 慢）：sparse K=5 sampling，把 $o_t$ 替换为 *matched distractor*（不是 random garbage），测 $\text{KL}(\pi_\theta(\cdot|h_t, o_t) \| \pi_\theta(\cdot|h_t, \tilde{o}_t))$
- 路 2（lexical, 快）：rouge-L between key evidence spans of $o_t$ and downstream reasoning + answer

**核心 technical claim**：双源结构（slow causal + fast lexical）+ λ curriculum 是关闭 discovery-utilization gap 的 *smallest adequate* mechanism；双源是抗 verbatim-copy hacking 的 *结构性*（非 hyperparameter）防御。

**Required minimum evidence**：
- E1（必需）: Qwen2.5-7B + Qwen2.5-14B × 3 seeds × 5 benchmark 上 conversion +≥30 pp，verbatim copy ratio < 5%
- E2（必需）: AttribRL 与 TRM/ToolRL/StepTool/CST-adapted-to-tool 的 head-to-head 对比，差异显著
- E3（必需）: irrelevant-observation negative control（matched distractor）下 pure-lexical agent 表现退化 ≥ ; AttribRL 维持
- E4（强加分）: 14-model × 5-benchmark DiscUseBench grid 揭示至少 2 个新的机制规律

## Method Thesis

- **One-sentence thesis**: We train tool-augmented LLM agents to causally condition their actions on tool observations by optimizing a self-supervised dual-source attribution reward — sparse counterfactual replacement KL plus dense lexical span overlap — mixed with outcome reward under a λ curriculum within GRPO, closing the discovery-to-utilization gap with structural defense against verbatim-copy reward hacking.
- **Why smallest adequate**: 不动 RL 算法（沿用 GRPO）、不动模型 backbone（沿用 Qwen2.5-7B/14B）、不引入额外标注（self-supervised）；只新增 *一个* trajectory-level reward 项，但用结构性双源防御 hacking。
- **Why timely in foundation-model era**: tool-augmented LM 是 LM post-training 的核心；MCP 生态 4400+ servers 让 utilization 而非 retrieval 成为下游瓶颈；EMNLP 2025 已接 ≥4 篇 tool-use RL paper。

## Contribution Focus

- **Dominant contribution**: AttribRL — 首个把 trajectory-level utilization 写进 RL training reward 的方法；双源 reward 设计本身是新的 *formulation*，不是已知技术拼装。
- **Supporting contribution（≤1）**: DiscUseBench — 14 model × 5 benchmark 的 Discovery/Use/Conversion 三元评测协议（同时是 AttribRL 的 evaluation 基础设施 + 独立 Findings fallback）。
- **Explicit non-contributions**:
  - 不提出新的 RL 算法（GRPO 是 black box reuse）
  - 不做 multimodal observation
  - 不解决 tool retrieval / MCP selection
  - 不训练 frontier-scale models
  - 不与 CST 整合成 unified counterfactual training framework（保持 scope）

## Proposed Method

### Complexity Budget

- **Frozen/reused**: Qwen2.5-7B-Instruct backbone, GRPO algorithm (veRL fork from Search-R1), BFCL/ToolBench/API-Bank/WebArena-Lite/SWE-Bench-Lite 数据 pipeline。
- **New trainable**: AttribRL reward 计算 module（rouge-L extractor + counterfactual rollout dispatcher + KL-on-action-distribution scorer + λ scheduler）。**不**新增任何可训练 reward model parameter——所有 reward 都是 trajectory function。
- **Tempting additions intentionally excluded**: 学一个 reward model（如 TRM 路线）、加 attention attribution（如 AgenTracer 路线）、做 causal SCM full formalization（IDEA-03 路线，scope creep）。

### System Overview

```
[Tool-augmented trajectory τ = (s_0, a_0, o_0, ..., s_T)]
                ↓
   ┌────────────────────────────────────────────┐
   │ AttribRL Reward Computer (per τ)            │
   │                                             │
   │ For each step t with observation o_t:        │
   │   ─────────── Path 1 (cheap, every step) ───│
   │   span_t = key_span_extract(o_t, task)        │
   │   R_lex_t = rouge_L(span_t, action_t + reasoning_{t:t+w}) │
   │                                             │
   │   ─────────── Path 2 (expensive, every K=5) ─│
   │   õ_t = matched_distractor(o_t, task)        │
   │   π1 = π_θ(·|h_t, o_t)                       │
   │   π2 = π_θ(·|h_t, õ_t)                       │
   │   R_cf_t = KL(π1 || π2)                      │
   │                                             │
   │   ─────────── Mixture ───────────────────── │
   │   D_t = LLM_judge_discovered(o_t, task)      │
   │   U_t = (α R_cf_t + (1-α) R_lex_t) × D_t      │
   │                                             │
   │ R_total(τ) = R_outcome(τ) + λ(epoch) Σ_t U_t  │
   └────────────────────────────────────────────┘
                ↓
        [GRPO group-normalized advantage Â_i]
                ↓
        [Policy update on π_θ]
```

### Core Mechanism — 双源 Attribution Reward

**Counterfactual replacement (slow, K=5 sparse)**:
- 关键 design：$\tilde{o}_t$ **不是 random garbage**——是 **matched distractor**：从 task-relevant 但 *不含 answer evidence* 的 tool output 池中随机抽样（例如 BFCL 上是同 schema 的其他 API 文档）。这关键回应 R3 weakness "garbage breaks support → OOD artifact"。
- $R_{\text{cf}}_t = \text{KL}(\pi_\theta(\cdot|h_t,o_t) \| \pi_\theta(\cdot|h_t,\tilde{o}_t))$——衡量 policy 在 step t 对 obs 的 sensitivity；不需要 ground-truth action label。
- **Delayed-utilization extension**（回应 R3 weakness "delayed utilization missed"）：除单步 KL，再计算 *forward roll-out* KL: $R_{\text{cf-fwd}}_t = \mathbb{E}_{a_{t+1:t+W}}[\text{KL}(\pi_\theta^{(W)}(\cdot|h_t,o_t) \| \pi_\theta^{(W)}(\cdot|h_t,\tilde{o}_t))]$，window W=3。

**Lexical attribution (fast, every step)**:
- $\text{span}_t = \text{KeySpanExtract}(o_t, \text{task})$——关键 design：span 抽取是 **rule-based**（task-specific extractor: BFCL 取 function-call result fields, ToolBench 取 API response value fields, SWE-Bench 取 diff/file content），**不依赖 LLM 调用**，因此 R_lex 是真正 self-supervised（回应 R1 weakness "R_lex key span source unclear"）。
- $R_{\text{lex}}_t = \text{rouge-L}(\text{span}_t, [a_t \| \text{reasoning}_{t:t+w}])$，window w=2 steps。

**Discovery gating**:
- $D_t = \text{LLM\_judge\_discovered}(o_t, \text{task})$ — 0/1 标签，只对已 discovered 的 step 计 utilization reward；避免对 noise observation 惩罚。
- 用 gpt-5.5-mini（低成本）作 judge；与 DiscUseBench 共享 judge。

**Mixture**:
- $U_t = (\alpha R_{\text{cf}}_t + (1-\alpha) R_{\text{lex}}_t) \cdot D_t$，default α=0.6（lexical 多但有 hack 风险，counterfactual 是结构性防御）。
- $R_{\text{total}}(\tau) = R_{\text{outcome}}(\tau) + \lambda(\text{epoch}) \cdot \sum_t U_t$
- **λ schedule**: linear warmup from 0.0 → 0.5 over first 30% of training; then anneal to 0.1 (回应 R2 weakness "λ schedule 决定结论"； 同时是 IDEA-11 的 minimal ablation)。

**为什么这是核心 novelty**:
- 双源不是 A+B 拼装——而是 **针对 verbatim-copy hacking 的结构性防御**: pure-R_lex 可被 verbatim copy 攻击（agent 把 obs 复制到 reasoning 即 rouge-L=1）；但 verbatim copy 不会改变 π(a|h,õ)，所以 R_cf 不会奖励 verbatim copy。两者必须同时高，强迫 agent 真正 condition action on obs。

### Optional Supporting Component — DiscUseBench

**为什么必要而非 sprawl**: DiscUseBench 是 AttribRL evaluation 的硬数据基础设施（5 benchmark instrumented for D/U/C measurement），且独立作为 Plan B Findings paper（user 明确 Plan B 路径）。

- **Coverage**: 5 benchmark（BFCL/ToolBench/API-Bank/WebArena-Lite/SWE-Bench-Lite）× 14 model（4 frontier closed + 3 open large + 3 open mid + 4 RL-trained 包括 AttribRL ours）
- **Metrics**: Discovery rate (LLM judge), Use rate (counterfactual replacement-based), Conversion rate
- **Closed-model 冻结**: GPT-5.5-2026-04, Claude Opus 4.7-2026-03, Gemini 3 Pro-2026-04（回应 R2 Benchmark Specialist weakness）

### Modern Primitive Usage

- **GRPO**: 政策更新 black box，沿用 veRL fork (Search-R1 codebase)
- **gpt-5.5-mini as LLM judge**: 仅作 binary discovery gate，避免 LLM judge over-confidence (R1 weakness "judge can be fooled by lexical overlap")。Judge 只判 *binary discovery* 而不判 use；use 由 counterfactual oracle 客观决定。
- **QLoRA rank 64**: 适配 2×4090 算力
- **Matched distractor sampling**: 从 task-relevant pool 抽样而非 random garbage（关键 design）

### Integration

```
veRL fork (Search-R1) trainer
  ↓
Custom RewardComputer plugin:
  - on_rollout(τ) -> R_outcome (task verifier)
  - on_rollout(τ) -> R_attr (双源 utilization)
  - aggregate -> R_total -> GRPO advantage
```

### Failure Modes and Diagnostics

| Failure mode | Detection | Mitigation |
|--------------|-----------|------------|
| Verbatim copy hacking (R_lex 被 hack) | verbatim copy ratio > 10% on held-out | 提高 α (counterfactual weight)；R_cf 结构性防御保底 |
| Counterfactual breaks support (R_cf OOD) | matched distractor RM 检测：distractor 与 o_t 的 task validity 差异显著 | distractor 池增大；fallback 至 pure R_lex with verbatim audit |
| Delayed utilization missed | window W=3 forward KL 仍 → 0 | 增大 W；或 multi-step rollout counterfactual |
| λ misset (overfit utilization, lose accuracy) | accuracy 下降 ≥ 3pp | adaptive λ：$\lambda_{t+1} = \lambda_t - \eta \cdot \max(0, \text{acc}_t - \text{acc}_{\text{base}} - 5\text{pp})$ |
| Training diverge | GRPO KL 爆炸 | 标准 GRPO clipping + reward normalization |
| Fallback to Plan B | 若 W2 pilot 显示训练不收敛 | DiscUseBench 独立 Findings paper（已有 instrumentation）|

### Novelty and Elegance Argument

**Closest work table**:

| Work | Object of optimization | Reward signal | Delta |
|------|------------------------|---------------|-------|
| ToolRL (NeurIPS '25) | task outcome | refined outcome | AttribRL: trajectory utilization |
| TRM (OpenReview '26) | invocation correctness | learned PRM | AttribRL: observation-action causality, no extra reward model |
| StepTool / AgentPRM | step progress | rule shaping / MC progress | AttribRL: causal grounding, self-supervised |
| **CST (arXiv:2602.20710)** | CoT predictability | counterfactual input → simulator can predict CoT | **AttribRL: policy action 对 obs 的 causal dependency；不是 CoT 一致性** |
| AgenTracer | failure attribution | counterfactual replay → decisive error | AttribRL: training reward, 非 post-hoc diagnosis |
| CoRM-RAG / FaithfulRAG | document-level faithfulness | passage perturbation | AttribRL: trajectory-level sequential dependency |
| Anchor P01 | none (diagnostic) | none | AttribRL: first training-side response |

**Why focused not module pile**: 唯一新引入的可计算量是 $U_t$ 一个 reward function；GRPO unchanged, backbone unchanged, no new neural network。

## Theoretical Grounding

### T1 — Formalizability Scan

| Component | Formal object | Draft |
|-----------|--------------|-------|
| Counterfactual KL | Information-theoretic | $R_{\text{cf}}_t = \text{KL}(\pi_\theta(\cdot|h_t,o_t) \,\|\, \pi_\theta(\cdot|h_t,\tilde{o}_t))$ |
| Lexical attribution | Sequence similarity | $R_{\text{lex}}_t = \text{rouge-L}(\text{span}_t, [a_t \| r_{t:t+w}])$ |
| Mixture reward | Convex combination | $R_{\text{total}} = R_{\text{out}} + \lambda \sum_t [\alpha R_{\text{cf}}_t + (1-\alpha) R_{\text{lex}}_t] D_t$ |
| λ schedule convergence | Curriculum optimization | Conjecture: linear-warmup + annealing 使 $\|\pi_T - \pi^*\|_\text{TV} \le O(\sqrt{T^{-1}})$ under GRPO clipping (assumes 标准 PPO 收敛) |
| Counterfactual identifiability | Causal inference | Under Pearl backdoor with $H_t$ as adjustment set: $\mathbb{E}[\Delta A_t | do(O_t = \tilde{o}_t)] \approx \text{KL}(\pi(\cdot|h_t, o_t) \| \pi(\cdot|h_t, \tilde{o}_t))$ when distractor $\tilde{o}_t$ is on-support |
| Verbatim copy defense | Two-source structural inequality | Conjecture: 对 pure-copy agent $\pi_{\text{copy}}$, $R_{\text{lex}}(\pi_{\text{copy}}) \to 1$ 但 $R_{\text{cf}}(\pi_{\text{copy}}) \to 0$ 因 copy 不依赖 obs；联合 reward $U$ 严格小于 honest utilization agent |

### T2 — Assumption Inventory

| Claim | Assumptions | Classification |
|-------|-------------|----------------|
| Counterfactual KL ≈ ATE | (i) $H_t$ d-separates $O_t$ from confounders；(ii) $\tilde{o}_t$ on-support | (i) RESTRICTIVE (history 是 sufficient adjustment set，可能漏 latent confounders)；(ii) UNVERIFIED (matched distractor 是否真 on-support 需 empirical validation) |
| GRPO convergence with mixed reward | clipping ε 足够小；reward normalization on | STANDARD |
| Verbatim copy 结构性防御 | counterfactual reward 在 train 时 active；R_cf 颗粒度 ≥ R_lex | RESTRICTIVE（若 K=5 sparse 太稀疏，verbatim hacking 可能在非 K-步 occur）|
| Discovery gate D_t 准确 | LLM judge agreement ≥ 0.85 with oracle | UNVERIFIED — 需 human spot-check |
| Lexical span extractor 与 task answer 高 recall | rule-based extractor 在 5 benchmark 上覆盖 ≥ 80% answer span | UNVERIFIED — 需 W1 pilot 验证 |

**Empirical evidence for RESTRICTIVE/UNVERIFIED**：
- Counterfactual on-support: 训练时记录 distractor task validity score（用 gpt-5.5-mini binary 判定），report 分布；正常应 > 0.85
- Discovery gate accuracy: 200 human-annotated samples 计算 κ
- Span extractor coverage: pilot 在 100 task 上手算 coverage

### TE — Theory-Experiment Alignment Matrix

| Theoretical Claim | Claim Type | Standard Validation Protocol | Required Scale | Feasibility | Flag |
|---|---|---|---|---|---|
| GRPO λ-mixed reward 收敛 | Convergence bound | Training curve (reward, KL, accuracy) vs steps × 3 seeds × 3 LR | 3 seeds × 3 LR | FEASIBLE WITH CAVEATS: 仅在 Qwen-7B 上完整跑 3×3；Qwen-14B 减至 3 seeds × 1 LR | ⚠️ partial |
| Conversion rate 提升 (+30pp) | Empirical hypothesis | Multi-benchmark conversion × 3 seeds, paired t-test vs TRM/ToolRL | 5 benchmark × 2 size × 3 seeds | FEASIBLE | ✓ |
| Verbatim copy defense (double-source > single-source) | Empirical hypothesis | 4-variant ablation: outcome-only / +R_lex / +R_cf / +both | 4 variants × 2 size × 3 seeds | FEASIBLE | ✓ |
| Counterfactual on-support assumption | Distributional check | Distractor validity score 分布；report mean ± std；compare to random garbage baseline | 1000 distractor samples | FEASIBLE | ✓ |
| Pearl backdoor identifiability | Sample complexity / causal | 不可严格 validate; weakened claim: "matched distractor 比 random garbage 显示更稳定的 R_cf vs ATE proxy" | controlled synthetic task (50 instances with known causal answer) | FEASIBLE WITH CAVEATS: 仅 synthetic substrate 上做 | ⚠️ partial |
| Delayed-utilization detection (W=3 fwd KL) | Empirical hypothesis | Conversion gap between W=1, W=3, W=5；should be monotonic non-decreasing | 3 W values × 1 size × 3 seeds | FEASIBLE | ✓ |
| 14B size transfer | Generalization bound | Cross-size: train 7B, evaluate trend at 14B（少量 steps） | 2 sizes × 3 seeds | FEASIBLE | ✓ |

**⚠️ Theory-Experiment Gap** (1 flag):
- Pearl backdoor identifiability 严格 validate 不可能（无 ground-truth ATE）。
  - (a) **Weaken**：claim 改为 "matched distractor 比 random garbage 在 controlled synthetic task 上更稳定逼近 known answer change"（已采用）
  - (b) **Theory-only**: 提供形式化 lemma（matched-distractor backdoor adjustment 的 identifiability sketch），但不 formal proof（EMNLP 不要求严格证明）
  - (c) **Redesign**: 在 BFCL synthetic-API-doc 子集上构造 ground-truth observation→action graph，用 50 instances 比较 R_cf 与 oracle ATE

## Evaluation Sketch

- **How validated**: (1) Main table: 14 model × 5 benchmark D/U/C 三维（DiscUseBench）; AttribRL row 是 Qwen-7B + Qwen-14B；(2) Method ablation: outcome only / +R_lex only / +R_cf only / +both / +both + λ schedule（5 variants）；(3) Reward hacking audit: verbatim copy ratio、distractor validity score、delayed window scan。
- **Key metric**: Conversion rate primary；final task success secondary；verbatim copy ratio < 5% 为硬底线。
- **Success**: Qwen-7B BFCL conversion 18% → ≥55%；Qwen-14B trend 复现；double-source > single-source 显著 (paired t-test p<0.05)；DiscUseBench 揭示 ≥2 个新机制结论。
- **Failure**: AttribRL 与 TRM/ToolRL 在 conversion 上无显著差异，或 verbatim copy ratio > 10% → fallback Plan B (DiscUseBench Findings)。

## Resource Estimate

- **Scale**: MEDIUM (4 person-weeks per design doc §8)
- **Compute**: 2×4090 (48GB)。AttribRL training estimate: 7B × 5 variant × 2 epoch + 14B × 2 variant × 1 epoch ≈ 18 GPU-day（含 counterfactual K=5 sparse 1.5× overhead）。Buffer 3 GPU-day for ablation。
- **Data**: 5 benchmark 全开源；BFCL train + ToolBench train (~30K trajectories) 作为 RL data。
- **API budget**: gpt-5.5-mini judge for ~50K observation-trajectory pairs × 14 model × 5 bench DiscUseBench eval。预算 $30 内。
