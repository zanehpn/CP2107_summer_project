# Filtered Research Ideas — AgentRead

**Direction**: AgentRead: Diagnosing and Closing the Discovery-to-Utilization Gap in Tool-Augmented LLMs
**Date**: 2026-05-12
**Pipeline**: 11 raw ideas → Feasibility filter → Novelty quick-check → Impact filter → Prof. He 4-dim filter (≥12/20) → Anti-pattern check → **6 surviving**
**Landscape source**: outputs/LANDSCAPE.md (35 papers, 12 gaps)
**Codex thread ID**: 019e17c2-d657-7c51-9a97-a817899d3399
**Venue (forwarded to next phase)**: EMNLP

---

## Surviving Ideas (ranked by Prof. He composite score, descending)

### Rank 1: AttribRL — Counterfactual-Lexical Attribution Rewards for Tool-Augmented Agents (IDEA-01)

- **Anchored critique**: CRITIQUE-02 + CRITIQUE-03 + CRITIQUE-05 + CRITIQUE-07 + CRITIQUE-14 — 一次同时消化 5 个最高优先级 critique。
- **Thesis**: We show that tool-augmented agents close the Discovery-to-Utilization gap by optimizing a self-supervised counterfactual+lexical attribution reward under GRPO.
- **Gap addressed**: G2 (no RL with utilization reward) + G3 (counterfactual not on tool agent RL) + G7 (lexical+counterfactual defensive combo)。
- **Core mechanism**: utilization reward = α·counterfactual (observation replacement → action distribution shift) + (1-α)·lexical (rouge-L of action vs observation key spans)，与 outcome reward 通过 λ schedule 混合，GRPO 训练。
- **Non-obvious because**: outcome reward 与 utilization 解耦——97.54% discovery vs 0.53% use 是经验铁证；AttribRL 直接优化此缺失变量；lexical+counterfactual 组合是 verbatim-copy defense（单 lexical 可被 hack，单 counterfactual 太贵）。
- **Contribution type**: new method + new formulation + empirical
- **Theorem scaffold**: $R = R_{\text{out}} + \lambda_t R_{\text{attr}}$，$R_{\text{attr}} = \alpha R_{\text{cf}} + (1-\alpha) R_{\text{lex}}$，$R_{\text{cf}} = \mathrm{KL}(\pi_\theta(\cdot|h_t,o_t) \| \pi_\theta(\cdot|h_t,\tilde{o}_t))$
- **ML 子领域客观规律**: GRPO/PPO clipping (policy drift)、Pearl backdoor (do-intervention 近似)、reward learning consistency (Pareto curve)
- **Risk**: HIGH — counterfactual reward 在 actual training 中是否稳定无先例
- **Effort**: 6-9 person-weeks
- **Closest work**: ToolRL (Qian et al. NeurIPS 2025) — delta：ToolRL 奖励 final outcome；AttribRL 奖励 observation→action attribution 并显式测 Discovery/Use/Conversion。
- **He Score**: Longevity 5 + Passion 4 + Application 5 + Uniqueness 5 = **19/20**
- **Anti-pattern flags**: 无 (lexical+counterfactual 不是 "A+B stitching" 因为有明确机制学 rationale：counterfactual 防 lexical hacking)
- **Quick novelty**: **LIKELY NOVEL** — web search 确认无 prior work 把 trajectory-level counterfactual + lexical 组合作为 tool agent 的 RL training reward
- **Why this ranks #1**: Anchor 设计文档充分论证、5 个 critique 直击、Plan B (DiscUseBench) 防御稳健、与 Cohere/Edinburgh Cluster B 的 timeline race 必须抢节奏

---

### Rank 2: DiscUseBench — A Diagnostic Benchmark for Discovery, Use, and Conversion in Tool Agents (IDEA-02) [Plan B]

- **Anchored critique**: CRITIQUE-01 + CRITIQUE-04 + CRITIQUE-08 + CRITIQUE-09
- **Thesis**: We show that tool-agent success hides distinct failure modes by measuring Discovery / Use / Conversion across 14 models and 5 benchmarks.
- **Gap addressed**: G1, G5, G6, G12
- **Core mechanism**: 14 model × 5 benchmark 的 D/U/C 三维表 + mechanism analysis；不训练，纯诊断
- **Non-obvious because**: P01 anchor 只覆盖 2 benchmark × 4 model；缺规模化 cross-family evidence；EMNLP precedent (API-Bank, P31, EMNLP 2023) 接受 strong diagnostic
- **Contribution type**: diagnostic + empirical benchmark
- **Theorem scaffold**: Success ≠ Discovery × Use × Reasoning（条件意义下），模型 final accuracy 相近但 conversion 显著分布不同
- **ML 子领域客观规律**: measurement reliability (inter-judge agreement, bootstrap CI)、causal decomposition、error taxonomy stability (AMI/silhouette)
- **Risk**: LOW — instrumentation + evaluation 即可，负结果也强
- **Effort**: 4-6 person-weeks
- **Closest work**: "Agents Explore but Agents Ignore" (P01) — delta：从 2 bench × 4 model 扩到 5 bench × 14 model，加 mechanism analysis 与更细 use definition。
- **He Score**: Longevity 5 + Passion 4 + Application 4 + Uniqueness 4 = **17/20**
- **Anti-pattern flags**: 无
- **Quick novelty**: **LIKELY NOVEL** — MCP-Bench 等只评 single-dimension；无 cross-model 三维表
- **Why this ranks #2**: 既是 IDEA-01 的 Plan B safety net，又是独立可发表 strong diagnostic paper；effort 与 budget 都 fit EMNLP

---

### Rank 3: TrajFaith — Agent-Specific Faithfulness Evaluation for Tool Trajectories (IDEA-05)

- **Anchored critique**: CRITIQUE-06 + CRITIQUE-13
- **Thesis**: RAG-style faithfulness metrics misjudge tool agents; we introduce trajectory-aware sufficiency and necessity tests.
- **Gap addressed**: G10, G5
- **Core mechanism**: 三类 trajectory faithfulness — local necessity / delayed necessity / sequential sufficiency；替换 observation 并 replay subsequent policy 而非简单删除 passage
- **Non-obvious because**: tool output 改变 future action distribution 而非 final answer text；RAG metric 漏掉 delayed use
- **Contribution type**: diagnostic + new evaluation formulation
- **Theorem scaffold**: RAG faithfulness 与 trajectory faithfulness 在 25-40% successful trajectories 上 disagree；$N_t = \mathbb{1}[Y(O_t) \ne Y(\tilde{O}_t) \lor A_{t+1:T}(O_t) \ne A_{t+1:T}(\tilde{O}_t)]$
- **ML 子领域客观规律**: causal mediation (Pearl)、sequential decision theory、RAG perturbation as baseline
- **Risk**: MEDIUM — trajectory replay infra
- **Effort**: 5-8 person-weeks
- **Closest work**: CoRM-RAG (P20) — delta：从 retrieved-document counterfactual 转向 sequential trajectory counterfactual
- **He Score**: Longevity 4 + Passion 3 + Application 4 + Uniqueness 4 = **15/20**
- **Anti-pattern flags**: 无
- **Quick novelty**: **LIKELY NOVEL** — RAG counterfactual 已成熟但从未上 tool trajectory
- **Why this ranks #3**: 直接桥接 Cluster D (RAG faithfulness) 与 agent 社区；可与 IDEA-02 合并为 strong evaluation paper

---

### Rank 4: LeakProof-Use — Controlled Benchmarks for Parametric-Knowledge Isolation in Tool Agents (IDEA-06)

- **Anchored critique**: CRITIQUE-10
- **Thesis**: Apparent tool use is inflated by parametric knowledge leakage; we provide answer-swapped & freshness-controlled benchmarks.
- **Gap addressed**: G5, G6
- **Core mechanism**: fresh synthetic facts / answer-swapped known facts / hidden-state tool outputs；observation 替换后仍输出原答案则判 parametric shortcut
- **Non-obvious because**: 大模型答对 ≠ 用 tool 答对；这决定 RL reward 是否真训练 tool use
- **Contribution type**: diagnostic benchmark
- **Theorem scaffold**: $\text{Use}_{\text{standard}} > \text{Use}_{\text{leak-controlled}}$；gap 随 model strength 单调上升
- **ML 子领域客观规律**: dataset contamination theory、counterfactual evaluation、calibration analysis
- **Risk**: LOW — 小规模可完成
- **Effort**: 3-5 person-weeks
- **Closest work**: Sufficient Context (P17) — delta：从 RAG answerability 转为 tool-agent parametric leakage isolation
- **He Score**: Longevity 4 + Passion 3 + Application 4 + Uniqueness 4 = **15/20**
- **Anti-pattern flags**: 无
- **Quick novelty**: **LIKELY NOVEL**
- **Why this ranks #4**: 低成本 + 高 leverage — IDEA-01/02 的"硬控制"实验子集，可独立 publish 或并入 DiscUseBench appendix

---

### Rank 5: Causal AttribRL — Explicit Trajectory Causal Models Replace Lexical Attribution (IDEA-03)

- **Anchored critique**: CRITIQUE-05 + CRITIQUE-12
- **Thesis**: Explicit causal trajectory attribution is a more faithful reward signal than lexical/attention.
- **Gap addressed**: G3, G5, G6, G7
- **Core mechanism**: trajectory SCM；reward = $do(O_t = \tilde{O}_t)$ 对 $A_{t+1:T}, Y$ 的 average treatment effect
- **Non-obvious because**: attention 高亮无关、lexical 被 copy hack；causal effect 虽贵但可作 high-precision teacher
- **Contribution type**: new method + theoretical formulation
- **Theorem scaffold**: $U_t = \mathbb{E}_{\tilde{o}}[D(p_\theta(A_{t+1:T},Y|H_t,O_t), p_\theta(A_{t+1:T},Y|H_t,\tilde{o}))]$；conjecture：$U_t$ 升而 $R_{\text{out}}$ 不变时 conversion 更稳健
- **ML 子领域客观规律**: Pearl intervention、Jain&Wallace attention critique、variance-cost tradeoff
- **Risk**: HIGH — causal score 成本极高，可能不实用作 online RL
- **Effort**: 7-10 person-weeks
- **Closest work**: Causal Rewards (P23) — delta：从 static RM 转向 trajectory SCM
- **He Score**: Longevity 4 + Passion 3 + Application 3 + Uniqueness 4 = **14/20**
- **Anti-pattern flags**: 1 flag — **"Scale-dependent"**：causal effect 估计变量较大，可能需更多 rollouts 才能稳定；user 算力 2×4090 有 risk
- **Quick novelty**: **LIKELY NOVEL**
- **Why this ranks #5**: 是 IDEA-01 的 theoretical 强化版；若 IDEA-01 受 lexical hack 困扰可退化至此

---

### Rank 6: Tool Recall→Utilization — Joint Optimization for MCP Retrieval Agents (IDEA-04)

- **Anchored critique**: CRITIQUE-11
- **Thesis**: High tool Recall@k does not imply downstream utilization; we optimize retrieval with a conversion-aware objective.
- **Gap addressed**: G11, G2
- **Core mechanism**: retrieval reward 从 "gold tool in top-k" 改为 "retrieved tool caused useful downstream action"；rerank with utilization-conditioned label
- **Non-obvious because**: agent setting 中 Recall@k 仅是入口；真正瓶颈是 tool 是否进入 causal trajectory
- **Contribution type**: new formulation + empirical
- **Theorem scaffold**: $\rho(\text{Recall@k}, \text{Success}) < \rho(\text{ToolConversion@k}, \text{Success})$；$s(q,t) = s_{\text{ret}}(q,t) + \beta\hat{P}(\text{Use}=1|q,t,h)$
- **ML 子领域客观规律**: learning-to-rank consistency、off-policy evaluation (IPS)、IR metric mismatch
- **Risk**: MEDIUM
- **Effort**: 5-7 person-weeks
- **Closest work**: MCP-Zero (P34) / RAG-MCP (P35) — delta：从 Recall@k 转为 downstream utilization
- **He Score**: Longevity 4 + Passion 3 + Application 4 + Uniqueness 3 = **14/20**
- **Anti-pattern flags**: 无
- **Quick novelty**: **LIKELY NOVEL**
- **Why this ranks #6**: 桥接 Cluster A (tool RL) 与 MCP retrieval 社区；与 IDEA-01 正交且互补

---

## Eliminated Ideas

| # | Idea | Stage | Reason |
|---|------|-------|--------|
| IDEA-07 | Can LLM Judges Detect Tool Utilization? | He Filter | 12/20 (Longevity 3 + Passion 3 + Application 3 + Uniqueness 3) — judge audit 议题已在 NLP eval 通用化；niche 不足以独立 EMNLP Main；建议作为 IDEA-02 (DiscUseBench) 的 reliability subsection 出现 |
| IDEA-08 | Failure Modes Decomposition | He Filter | 12/20 (Longevity 3 + Passion 3 + Application 3 + Uniqueness 3) — mechanism analysis 过于探索性；建议作为 IDEA-02 的 mechanism subsection 出现 |
| IDEA-09 | Observation Quality as Hidden Bottleneck | He Filter | 13/20 (Longevity 3 + Passion 3 + Application 4 + Uniqueness 3) — 与 Lost in the Middle 同质化高；anti-pattern: 部分 overlap with existing context-position literature；建议作为 IDEA-01 ablation 或 IDEA-02 appendix |
| IDEA-10 | What Do PRMs Actually Reward? | He Filter | 13/20 (Longevity 4 + Passion 3 + Application 3 + Uniqueness 3) — 价值高但操作复杂、依赖现有 PRMs；建议作为 IDEA-01 与 IDEA-03 的对比 baseline 出现 |
| IDEA-11 | λ Is the Method | Strategic Merge | 12/20 (Longevity 3 + Passion 3 + Application 3 + Uniqueness 3) — 是 IDEA-01 的训练动力学子组件；保留为 IDEA-01 主表 5.2 ablation subsection（不独立发表）|

---

## Risk Distribution of Survivors

| Risk Level | Count | Ideas |
|------------|-------|-------|
| HIGH | 2 | IDEA-01 (AttribRL), IDEA-03 (Causal AttribRL) |
| MEDIUM | 2 | IDEA-04 (Recall→Use), IDEA-05 (TrajFaith) |
| LOW | 2 | IDEA-02 (DiscUseBench), IDEA-06 (LeakProof-Use) |

均衡分布（2/2/2），符合 instruction 要求（≥2 HIGH + ≥2 LOW）。

---

## Strategic Composition Map

```
[Main Stream — 主流核心]
    IDEA-01 AttribRL（HIGH risk, anchor design doc）
        ├── 依赖：IDEA-02 DiscUseBench（提供训练数据 + 评测协议）
        ├── 升级路径：IDEA-03 Causal AttribRL（若 lexical 被 hack）
        ├── 训练动力学：IDEA-11（合并入 ablation）
        └── 评测可信度：IDEA-07 + IDEA-08（合并入 appendix）

[Plan B — 独立诊断 paper]
    IDEA-02 DiscUseBench（LOW risk）
        ├── 控制实验：IDEA-06 LeakProof-Use
        ├── 评测细化：IDEA-05 TrajFaith
        └── 失败诊断：IDEA-08 + IDEA-09（合并）

[正交扩展]
    IDEA-04 Recall→Use（与主流形成上下游闭环）
```

---

## Suggested Next Steps

1. **Phase 3 (idea-screen)**: 对 top 2 (IDEA-01 AttribRL + IDEA-02 DiscUseBench) 跑 EMNLP venue 的 multi-dimensional screening — novelty cross-verification + reviewer simulation + strategic fit
2. **Phase 4 (idea-refine)**: 对 top 2 跑 iterative refinement，AttribRL 优先使用 Socratic-auto 模式（method paper 需要 deep mechanism dialogue）
3. **Future**: 如果 AttribRL screen 通过且 refine 评分 ≥ 9，立即进入 W1 pilot；否则 fallback 至 IDEA-02 DiscUseBench 独立路径

---

## Methodology Notes

- Brainstorming model: gpt-5.5 with xhigh reasoning effort (via Codex MCP, threadId 019e17c2-d657-7c51-9a97-a817899d3399)
- Filtering pipeline: Feasibility → Novelty quick-check → Impact → Prof. He 4-dim (threshold 12/20) → Anti-pattern
- Anchor design doc 已锁定 IDEA-01 为 AttribRL；其他 ideas 是其防御/扩展
- Risk distribution intentionally balanced: 2H/2M/2L 同时 cover method、diagnostic、bench、eval 4 类贡献
