# Screening Report — W-DiffPolicy

**Direction**: W-DiffPolicy: Wasserstein-Regularized Diffusion Policies for Multi-Modal Offline Reinforcement Learning
**Venue**: NeurIPS 2026
**Date**: 2026-05-09
**Ideas screened**: 4 (IDEA-01, IDEA-06, IDEA-09, IDEA-10)
**Composite weights**: novelty=0.25, venue=0.35, strategic=0.20, feasibility=0.20
**Codex thread**: `019e0d5d-3c03-79c0-9ec5-dfe562c2f248`

---

## Executive Summary

经过 Module A (novelty cross-verification w/ web search 与 cross-model verify)、Module B (NeurIPS 3-reviewer + meta simulation)、Module C (5-dim strategic fit) 三层评审，4 个 idea 出现明显分化：

**Top 2 (PROCEED)**: IDEA-10 KL-Mode Collapse in Continuous Control (composite **7.64**) 与 IDEA-01 Mode-Conditional W₂ Diffusion Policy (composite **7.23**)。两者形成完美互补——**IDEA-10 给出理论动机**（何时 KL collapse 必然/不必然发生），**IDEA-01 给出方法实现**（mode-conditional W₂ 保 mode）。可视为同一论文的理论+方法双柱，或两篇独立 sibling paper（第二篇 short paper 走 theory track）。

**Bottom 2 (CAUTION)**: IDEA-06 Value-Conditioned Wasserstein 因 OTPR (2502.12631, 已用 Q-function as transport cost on diffusion policy + online RL) 与 P16 Rethinking OT (Maximin Q-cost) 重叠被严重压缩 novelty (5.5/10)，三 reviewer 均给负面 verdict (Meta Weak Reject)。IDEA-09 NeoRL-2 OT Stress Test 因 benchmark-only 风险偏 + 实现工作量大被降到 CAUTION (Meta Weak Reject)，但若加入小理论 counterexample 可重启。

**关键警示**：本次 screening 揭示 **OTPR (2502.12631, Feb 2025)** 是上一阶段 lit-survey 漏掉的关键威胁论文——Q-cost OT for diffusion policy 已存在（虽是 online fine-tuning 不是 offline RL）。这一发现使 v1 W-DiffPolicy 必须把 "Q-cost OT" 视为 known，把差异化重心放回 **mode-preservation + offline behavior regularization + theorem** 三个维度。

---

## Per-Idea Reports

### IDEA-01: Mode-Conditional W₂ Diffusion Policy — **PROCEED**

#### Module A: Novelty Assessment
- **Score**: 8/10
- **Recommendation**: PROCEED
- **Core Claims**:
  1. 全局 W₂ 不能保 per-mode mass — Novelty: HIGH — Closest: P14 Q-DOT (IQL global W₂)
  2. Latent mode variable + per-mode W₂ 是 mode-preservation 的合理 formalization — Novelty: HIGH — Closest: LOM (单 mode 选择，方向相反)
  3. Mode-recall theorem `Recall ≥ 1 - O(ε/Δ²) - O(mode-est-error)` — Novelty: HIGH — Closest: 无直接对应理论

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| LOM | 2025 | ICLR | mod | LOM 选最佳单 mode；本提案保留多 mode |
| Latent Diffusion ORL (Venkatraman) | 2024 | ICLR | mod | 用 latent diffusion 压缩 skill；无 OT 正则、无 mode preservation |
| DSRL | 2025 | arXiv | sup/mod | RL over latent-noise space；无 OT、无 offline mode-preservation |
| Q-DOT | 2025 | RLC | mod | ICNN W₂ on IQL，非 diffusion，非 mode-conditional |

- **Key differentiator**: 把 OT 从 global behavior distance 改成 mode-conditional safety constraint——不是"学一个 latent"，而是证明/测量每个 value-relevant mode 是否被保住。
- **Suggested NeurIPS positioning**: "现有 diffusion offline RL 把 multimodality 当作 density-modeling 副产品；我们把 mode preservation 变成显式约束、指标和理论对象。" 不要写成"latent diffusion + W₂"。

#### Module B: NeurIPS Reviewer Simulation

**Reviewer 1 (Empiricist)**:
- Tier: 2
- Strengths: (a) 补 return-only 评估盲点，mode recall/mass error/per-mode return 都是 reviewer 能理解的指标；(b) 相比 Q-DOT，实验 delta 清晰：diffusion + per-mode W₂
- Weaknesses: (a) latent mode z 学习可能成为 hidden confound；(b) 必须证明不是 auxiliary latent module 起作用，而是 per-mode W₂；(c) D4RL Kitchen/AntMaze mode ground truth 不够干净，需 synthetic-controlled split
- Verdict: **Weak Accept** (6/10)
- 怎样 Accept: 三层实验链 (synthetic + Kitchen + AntMaze)，每层做 global W₂ / KL / no mode-mass / oracle mode 的 ablation，公开 mode extraction code

**Reviewer 2 (Innovator)**:
- Tier: 1/2 边界
- Strengths: (a) 不是简单 KL→W₂，而是指出 global distance ≠ mode preservation；(b) 把 diffusion ORL、OT、multi-modal diagnostics 三线合并成新问题定义
- Weaknesses: (a) 若只 D4RL+几个 mode 指标会回到 Tier 2；(b) "value-relevant mode" 若太工程化像 metric hacking
- Verdict: **Accept** (8/10)
- 怎样 Accept: 开篇强力反例展示 global W₂ 也吞 rare mode，再证明 conditional W₂ 修复

**Reviewer 3 (Rigorist)**:
- Tier: 2
- Strengths: (a) 有可证明对象 (mode separation, conditional W₂ error, mode-est error)；(b) 比 "W₂ 更平滑所以更好" 严谨
- Weaknesses: (a) mode 是 observable partition / latent variable / learned cluster 三种条件不同；(b) theorem 若只 Gaussian mixture toy case 主文应避免过度声称；(c) ICNN estimation error 必须进入 bound
- Verdict: **Weak Accept** (6/10)
- 怎样 Accept: 给出 finite-sample theorem，把 mode-est error 与 ICNN-W₂ error 显式写进 guarantee

**Meta Review**:
- 争议: Innovator 认为是 mode-preservation 问题的重定义；Empiricist/Rigorist 担心 mode variable 与 theorem assumptions 太脆
- 最终裁决: **Weak Accept**
- 若接收冲 Spotlight 需要的: 干净反例—global KL/W₂ 都高分但丢 mode；Mode-Conditional W₂ 保 mode 不牺牲 return
- Top 3 风险:
  1. Mode extraction 不稳定
  2. Theorem 与 ICNN 实现脱节
  3. Return 提升不明显，只剩 diagnostic 贡献

**Venue Score**: avg(6, 8, 6) = **6.7/10**

#### Module C: Strategic Fit

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Longevity | 8/10 | Multi-modal preservation 是 RL/imitation 的根本问题；不会被一两年的趋势消解 |
| Roadmap Viability | 8/10 | Paper 1 = method+theorem；Paper 2 = mode-aware distillation (RACTD-W)；Paper 3 = real-robot transfer。3+ paper arc 自然 |
| Application Grounding | 7/10 | Kitchen/AntMaze + NeoRL-2 实践相关；机器人 manipulation 工业价值高 |
| Execution Uniqueness | 7/10 | 作者 v1 设计、ICNN 经验、双卡硬件已就绪；U-Tokyo Q-DOT 组是直接竞争但作者 framing 更清晰 |
| Iteration Readiness | 7/10 | Synthetic mixture skeleton experiment <1 周；Kitchen 单卡数小时；快速反馈 |

**Strategic Score**: avg = **7.4/10**

#### Composite Score for IDEA-01

| Component | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Novelty | 8.0/10 | 0.25 | 2.000 |
| Venue | 6.7/10 | 0.35 | 2.345 |
| Strategic | 7.4/10 | 0.20 | 1.480 |
| Feasibility | 7.0/10 | 0.20 | 1.400 |
| **Composite** | | | **7.225 → PROCEED** |

---

### IDEA-06: Value-Conditioned Wasserstein — **PROCEED WITH CAUTION**

#### Module A: Novelty Assessment
- **Score**: 5.5/10
- **Recommendation**: PROCEED WITH CAUTION
- **关键 finding**: OTPR (2502.12631, Feb 2025) 已用 Q-function as transport cost + diffusion policy + RL fine-tuning。Q-cost OT 不再是 novel angle。

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| **OTPR** | 2025 | arXiv 2502.12631 | **fund** | Online RL fine-tuning of diffusion-policy IL；Q-function as transport cost；policy as OT map |
| **Rethinking OT (P16)** | 2024 | NeurIPS | **fund** | Maximin OT formulation with Q as transport cost |
| Q-DOT | 2025 | RLC | mod | ICNN W₂ on IQL，Euclidean cost |

- **Key differentiator**: 唯一可守的新颖点：value-conditioned W₂ 作为 **offline diffusion policy** 的 behavior regularizer（OTPR 是 online fine-tuning），用以选择性保护高价值 modes 而非把整个 policy update 重写成 OT map。
- **Positioning**: 必须克制——"Q-cost OT is known; what is unknown is whether value-aware transport can be a conservative, mode-sensitive behavior regularizer for OFFLINE diffusion policies."

#### Module B: NeurIPS Reviewer Simulation

**Reviewer 1 (Empiricist)**: Tier 2/3 边界。Strengths: 解决 naive mode preservation 问题；可清晰 ablation。Weaknesses: Q bias 污染 transport cost；offline 下尤其危险；OTPR/P16 的差异不清；若只 +2-3 分 D4RL return 不够。**Verdict: Weak Reject (4/10)**

**Reviewer 2 (Innovator)**: Tier 3。Strengths: "保高价值 mode" narrative 对；可作为 IDEA-01 的 ablation。Weaknesses: OTPR 已是 diffusion+Q-cost；P16 已是 offline+Q-cost；"offline behavior regularizer" delta 不足以让 NeurIPS 兴奋。**Verdict: Reject (3/10)**

**Reviewer 3 (Rigorist)**: Tier 2/3 边界。Strengths: 明确指出 pure geometry ≠ RL value；可建 regret bound。Weaknesses: max(0, ΔQ) 方向/尺度/calibration 需严格解释；Q 不准时正则可能破坏 support；与 Maximin OT 关系必须形式化。**Verdict: Weak Reject (4/10)**

**Meta Review**: Empiricist 仍看到价值 (展示保坏 mode 的失败)；Innovator 认为 novelty ceiling 锁死；Rigorist 要求重新建立 objective 的理论正当性。**最终裁决: Weak Reject**。若拒可投 RLC、CoRL、ICLR workshop，或合并入 IDEA-01 作 ablation。Top 3 风险: (1) 被判 OTPR 的 obvious variant；(2) Q-error 让 cost 不稳；(3) 正结果只是小 return gain。

**Venue Score**: avg(4, 3, 4) = **3.7/10**

#### Module C: Strategic Fit

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Longevity | 6/10 | Q-cost OT 趋势能持续 2-3 年但不是基础问题 |
| Roadmap Viability | 5/10 | 自然延伸有限，OTPR/P16 已锁住主路径 |
| Application Grounding | 6/10 | 同 D4RL/NeoRL-2 testbed |
| Execution Uniqueness | 5/10 | 作者有 expertise 但 OTPR/P16 已在路径上 |
| Iteration Readiness | 7/10 | 改 ground cost 快；Euclidean vs value-cost ablation 直接 |

**Strategic Score**: avg = **5.8/10**

#### Composite Score for IDEA-06

| Component | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Novelty | 5.5/10 | 0.25 | 1.375 |
| Venue | 3.7/10 | 0.35 | 1.295 |
| Strategic | 5.8/10 | 0.20 | 1.160 |
| Feasibility | 7.0/10 | 0.20 | 1.400 |
| **Composite** | | | **5.23 → PROCEED WITH CAUTION** |

**强烈建议**: 不作为独立论文，而是合并入 IDEA-01 作 ablation 章节（"Q-aware ground cost vs Euclidean ground cost"），保留新颖部分（softmax(V/τ) mode mass weighting）。

---

### IDEA-09: NeoRL-2 Transfer Stress Test for OT-Regularized Offline RL — **PROCEED WITH CAUTION**

#### Module A: Novelty Assessment
- **Score**: 6.5/10
- **Recommendation**: PROCEED WITH CAUTION

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| OTDF | 2025 | ICLR | fund/mod | Transition-level OT alignment + selective sharing；本提案在 NeoRL-2 上压测并加 Bellman-relevance diagnostic |
| NeoRL-2 | 2025 | arXiv | mod | 提供 benchmark；本提案专门检验 OT transfer 假设 |
| Q-DOT | 2025 | RLC | mod | OT 正则 offline RL 但无 NeoRL-2 transfer test |

- **Key differentiator**: 不是 "OTDF 跑到 NeoRL-2 上"，而是检验 OT alignment 是否会删除 Bellman-critical rare transitions，并提出 transition-critical OT correction
- **Positioning**: experimental-flaw / stress-test paper；现有 cross-domain OT 证明 distribution alignment 而非 offline-control usefulness。纯 benchmark 风险偏高，最好附带小理论模型解释 OT failure。

#### Module B: NeurIPS Reviewer Simulation

**Reviewer 1 (Empiricist)**: Tier 2。Strengths: NeoRL-2 conservative data + delay + 安全约束 正好测 OT alignment 盲点；若证明 lower OT distance 反降 Bellman coverage 是强 finding。Weaknesses: NeoRL-2 较新，baseline tuning 受审；需覆盖 CQL/IQL/TD3+BC 等非-OT baselines；stress test 若无 correction 易被认为是 negative benchmark。**Verdict: Weak Accept (6/10)**

**Reviewer 2 (Innovator)**: Tier 2/3 边界。Strengths: 有机会推翻 cross-domain OT 直觉；负结果若强会影响评价范式。Weaknesses: 若主贡献是 benchmark study，main track novelty 不稳；transition-critical correction 若只是 TD-error weighting 显得 incremental。**Verdict: Weak Reject (4/10)**

**Reviewer 3 (Rigorist)**: Tier 2。Strengths: Bellman-relevant coverage 比 marginal OT distance 更原则化；可建 counterexample。Weaknesses: "Bellman-critical" 必须形式化，否则只是重命名 TD-weighting；需区分 dynamics mismatch / reward sparsity / support scarcity 三种 failure source；NeoRL-2 复杂性可能掩盖机制。**Verdict: Weak Reject (4/10)**

**Meta Review**: Empiricist 喜欢系统压测；Innovator/Rigorist 认为若无理论反例或新 metric 主会 novelty 偏弱。**最终裁决: Weak Reject**。若拒可投 NeurIPS Datasets&Benchmarks / RLC / ICLR workshop。Top 3 风险: (1) 变成 "我们在 NeoRL-2 上跑了很多 baseline" 的工程报告；(2) OTDF 复现+NeoRL-2 tuning 超预算；(3) negative result 不稳无法形成清晰结论。

**Venue Score**: avg(6, 4, 4) = **4.7/10**

#### Module C: Strategic Fit

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Longevity | 7/10 | NeoRL-2 likely 持续 benchmark；real-world stress test 价值长期 |
| Roadmap Viability | 6/10 | 可以扩展为"real-world OT toolkit"系列但范围有限 |
| Application Grounding | 9/10 | 直接关联工业管道、核聚变、健康医疗——最强 application |
| Execution Uniqueness | 6/10 | 作者有 ICNN 经验；NeoRL-2 复现 overhead 是劣势 |
| Iteration Readiness | 4/10 | NeoRL-2 simulator 较慢；cross-domain 实验长 |

**Strategic Score**: avg = **6.4/10**

#### Composite Score for IDEA-09

| Component | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Novelty | 6.5/10 | 0.25 | 1.625 |
| Venue | 4.7/10 | 0.35 | 1.645 |
| Strategic | 6.4/10 | 0.20 | 1.280 |
| Feasibility | 5.0/10 | 0.20 | 1.000 |
| **Composite** | | | **5.55 → PROCEED WITH CAUTION** |

**强烈建议**: 不作为本轮（6-8 周）主线；可作为 IDEA-01 主论文中的 cross-domain transfer 实验段（NeoRL-2 上 1-2 个 task 的迁移性 figure），或作为 follow-up paper（v2）。

---

### IDEA-10: KL-Mode Collapse in Continuous Control — **PROCEED**

#### Module A: Novelty Assessment
- **Score**: 8/10
- **Recommendation**: PROCEED

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| **P19 KL-Mode-Collapse** | 2025 | arXiv 2510.20817 | **fund/mod** | LM/化学 LM only；本提案推到 continuous control + state-conditioned action modes + Q-guidance |
| 2602.02250 Well-Posed KL-Reg Control via W | 2026 | arXiv | sup | LQG / 线性 Gaussian 控制，无多模态、无 RL、无 diffusion |
| Diffusion-QL / BDPO | 2023/2025 | ICLR/ICML | mod | KL-like 正则；本提案审计 objective-level mode-collapse 条件 |

- **Key differentiator**: 把 P19 从 sequence-generation target distribution 推到 state-conditioned continuous-control MDP，并回答 "KL collapse 在 offline diffusion control 中何时真发生、何时只是错误类比"
- **Positioning**: "Translation is not automatic"——P19 给 LM RL 的 collapse theory；IDEA-10 给 control-specific theorem/counterexample with Q-guidance, behavior support, action-mode geometry

#### Module B: NeurIPS Reviewer Simulation

**Reviewer 1 (Empiricist)**: Tier 2。Strengths: two-mode toy MDP 快速 (<1 周)；Kitchen/AntMaze mode-controlled splits 连接实践。Weaknesses: 若只 toy MDP NeurIPS 会觉得影响有限；需比较 forward KL/reverse KL/score-BC/Q-guidance/W₂；mode collapse 指标与 return 必须分开报告。**Verdict: Weak Accept (6/10)**

**Reviewer 2 (Innovator)**: Tier 1。Strengths: 高概念清晰——纠正快速传播的错误类比；正负结果都可发表。Weaknesses: 若只 "depends on hyperparameters" 会显得弱；theorem 需写成 reviewer 一眼能记的 threshold law。**Verdict: Accept (8/10)**

**Reviewer 3 (Rigorist)**: Tier 1/2 边界。Strengths: `m₂* → 0 iff ΔQ/τ > θ(behavior mass ratio)` 是好形式：可证明、可解释、可实验验证；KL/score-BC/Q-guidance/W₂ 同框分析。Weaknesses: state-conditioning 让 derivation 比 P19 复杂；score-matching BC vs KL 等价关系在 diffusion 下要小心限定；W₂ comparison 若附带可能分散主线。**Verdict: Accept (8/10)**

**Meta Review**: Innovator/Rigorist 认为高价值理论校准；Empiricist 担心实验不够真实。**最终裁决: Accept**。冲 Spotlight/Best Paper 需简洁 threshold theorem + 让人改口的实验 (P19-style intuition 既有成立边界也有失败边界)。Top 3 风险: (1) theorem 太窄被认为 toy；(2) score-BC 与 KL 关系处理不严谨；(3) 实验无法从 toy 平滑过渡到 Kitchen/AntMaze。

**Venue Score**: avg(6, 8, 8) = **7.3/10**

#### Module C: Strategic Fit

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Longevity | 9/10 | 理论论文老化慢；KL/W₂ 正则在 RL 是基础问题 |
| Roadmap Viability | 8/10 | Paper 1 = threshold theorem；Paper 2 = empirical validation in offline RL；Paper 3 = 推广到其他 policy class |
| Application Grounding | 5/10 | 理论性，应用 grounding 间接 |
| Execution Uniqueness | 7/10 | 作者深读 P19 + W-DiffPolicy v1；理论工作依赖少 |
| Iteration Readiness | 8/10 | Toy MDP 解析 + 小实验；快速反馈 |

**Strategic Score**: avg = **7.4/10**

#### Composite Score for IDEA-10

| Component | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Novelty | 8.0/10 | 0.25 | 2.000 |
| Venue | 7.3/10 | 0.35 | 2.555 |
| Strategic | 7.4/10 | 0.20 | 1.480 |
| Feasibility | 8.0/10 | 0.20 | 1.600 |
| **Composite** | | | **7.635 → PROCEED** |

---

## Recommendation Summary

| Rank | IDEA | Composite | Verdict | 行动 |
|------|------|-----------|---------|------|
| 1 | IDEA-10 KL-Mode Collapse Continuous Control | **7.64** | PROCEED | 进入 /idea-refine 深度精炼 |
| 2 | IDEA-01 Mode-Conditional W₂ Diffusion Policy | **7.23** | PROCEED | 进入 /idea-refine 深度精炼 (主论文方向) |
| 3 | IDEA-09 NeoRL-2 OT Stress Test | **5.55** | CAUTION | 作为 IDEA-01 的 cross-domain 实验段或 v2 follow-up |
| 4 | IDEA-06 Value-Conditioned Wasserstein | **5.23** | CAUTION | 作为 IDEA-01 的 ablation；不独立成文（OTPR 重叠） |

**关键策略建议**:
1. **IDEA-01 是主论文**——直接对应作者 v1 W-DiffPolicy 的 mode-preservation theorem 设计目标，且通过 screening。
2. **IDEA-10 是 sibling theory paper**——可独立成文 (short paper / theoretical track) 或作为 IDEA-01 的 motivation theorem section。建议先做 standalone short paper：理论性强、风险低、可冲 NeurIPS Spotlight。
3. **IDEA-06 / IDEA-09 整合进 IDEA-01**——分别作为 ablation 与 cross-domain 实验段，不浪费已发现的 idea，但避免 OTPR/benchmark-only 风险。
