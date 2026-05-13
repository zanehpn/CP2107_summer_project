# Filtered Research Ideas

**Direction**: W-DiffPolicy: Wasserstein-Regularized Diffusion Policies for Multi-Modal Offline Reinforcement Learning
**Date**: 2026-05-09
**Pipeline**: 11 ideas → Feasibility (11 通过) → Novelty quick-check (11 通过) → Impact (10 通过) → Prof. He 4-dim filter (10 通过) → Anti-pattern check
**Landscape source**: outputs/LANDSCAPE.md
**Critique source**: outputs/CRITICAL_ANALYSIS.md
**Codex thread ID**: `019e0d5d-3c03-79c0-9ec5-dfe562c2f248`

---

## 评分一览（按 He 综合分降序）

| Rank | IDEA | Title | Risk | Effort | He Score | Novelty | Anti-Pattern Flags |
|------|------|-------|------|--------|----------|---------|--------------------|
| 1 | IDEA-01 | Mode-Conditional W₂ Diffusion Policy | MEDIUM | 5-7w | **17/20** | LIKELY NOVEL | none |
| 2 | IDEA-06 | Value-Conditioned Wasserstein | MEDIUM | 5-7w | **17/20** | LIKELY NOVEL | mild "structure-modified" 但有新机制 |
| 3 | IDEA-09 | NeoRL-2 Transfer Stress Test | HIGH | 7-8w | **17/20** | LIKELY NOVEL | none (但工作量风险) |
| 4 | IDEA-10 | KL-Mode Collapse in Continuous Control | MEDIUM | 4-6w | **17/20** | LIKELY NOVEL | none |
| 5 | IDEA-02 | Reverse-Path != Terminal | LOW | 3-4w | **16/20** | LIKELY NOVEL | none (高 leverage critique) |
| 6 | IDEA-03 | Occupancy-Aware Wasserstein | HIGH | 7-8w | **16/20** | LIKELY NOVEL | none |
| 7 | IDEA-07 | ICNN-W₂ Reliability Pressure Test | MEDIUM | 4-6w | **15/20** | LIKELY NOVEL | none |
| 8 | IDEA-11 | Distill Without Losing Modes | MEDIUM | 6-8w | **14/20** | LIKELY NOVEL | mild "A+B" (RACTD+W₂)，但 mode-loss 角度新 |
| 9 | IDEA-04 | ModeBench-ORL Diagnostic | LOW | 3-5w | **13/20** | LIKELY NOVEL | mild "overly niche" 风险 |
| 10 | IDEA-08 | Sinkhorn-Diffusion Policy | MEDIUM | 5-6w | **12/20** | LIKELY NOVEL | mild "A+B" (Sinkhorn+diffusion) |
| ✗ | IDEA-05 | Budget-Normalized Diffusion ORL | LOW | 4-6w | 11/20 ✗ | LIKELY NOVEL | "audit paper"，passion 低 |

**Threshold**: 12/20 (10 ideas 通过, 1 淘汰)

---

## Surviving Ideas (Top 6 Recommended for Phase 3 Screening)

### Rank 1: Mode-Conditional W₂ Diffusion Policy (IDEA-01)
- **Anchored critique**: CRITIQUE-01, CRITIQUE-07, CRITIQUE-13 — 直接回应 "return-only improvement 可能只是保留单一高 return mode" 的根本评估盲点
- **Thesis**: 我们证明 conditional W₂ regularization 比全局 KL/W₂ 更能保留 value-relevant behavior modes（通过引入 latent mode variable）
- **Gap addressed**: G1 + G2 + G3
- **Core mechanism**: Latent mode variable z + per-mode W₂ + mode-mass regularizer，避免 W₂ 通过 mass 平移而非 per-mode preservation
- **Non-obvious because**: 全局 W₂ 可平均化 mode 而非逐 mode 保真——这是 W-DiffPolicy v1 的隐含假设漏洞
- **Contribution type**: new method + theoretical result + diagnostic
- **Theorem scaffold**: `Recall_mode(πθ) ≥ 1 - O(ε/Δ²) - O(mode-est-error)`，且全局 W₂ 给不出此 bound
- **Risk**: MEDIUM — mode 分解质量是核心风险，但 synthetic mixture skeleton experiment ≤1 周
- **Effort**: 5-7 person-weeks
- **Closest work**: P14 Q-DOT — delta: IQL→diffusion + global→mode-conditional
- **He Score**: Longevity 4 + Passion 5 + Application 4 + Uniqueness 4 = **17/20**
- **Anti-pattern flags**: none
- **Quick novelty**: LIKELY NOVEL (web search 确认 mode-conditional W₂ + diffusion + offline RL 未被做过)
- **Why this ranks #1**: 直接 sharpen v1 W-DiffPolicy 的核心假设（多模态保留），同时给出可证明的 mode-recall bound——是新方法+理论+诊断三合一，passion 完美对齐 v1 设计。

---

### Rank 2: Value-Conditioned Wasserstein (IDEA-06)
- **Anchored critique**: CRITIQUE-13, CRITIQUE-01 — 修正 "mode preservation 总是好" 的错误目标
- **Thesis**: naive mode preservation 会保留低价值/有害 mode；value-conditioned W₂ 只保护 return-relevant modes
- **Gap addressed**: 避免 v1 W-DiffPolicy 被批评为 "保多模态但不区分好坏"
- **Core mechanism**: Ground cost 改写为 `c((s,a),(s,a')) = ||a-a'||² + γ max(0, Qβ(s,a') - Qβ(s,a))`，让 OT mass 优先对齐高价值 support
- **Non-obvious because**: 引入 Q 看似破坏 OT 几何纯洁性，但 offline RL 目标本就是 constrained improvement 而非纯 density matching
- **Contribution type**: new method + new formulation
- **Theorem scaffold**: `min_θ L_diff - α E[Q] + λ W²_{c_Q}(πθ,β)`；conjecture: Q error bounded 时 value-cost W₂ 比 uniform W₂ 有更小 regret upper bound
- **Risk**: MEDIUM — Q bias 可能污染 regularizer，可由 ensemble uncertainty gate 缓解
- **Effort**: 5-7 person-weeks
- **Closest work**: P14 Q-DOT — delta: Q-DOT 用 W₂ regularize IQL；本提案把 value 信息塞进 transport cost 本身（不仅仅是 outer loop）
- **He Score**: Longevity 4 + Passion 5 + Application 4 + Uniqueness 4 = **17/20**
- **Anti-pattern flags**: 弱 "structure-modification" 但 ground-cost 改造是新机制
- **Quick novelty**: LIKELY NOVEL (BWD-IQL 用 Bellman-W 做数据质量；本提案把 Q 嵌入 transport cost 用于 diffusion 训练——未做)
- **Why this ranks #2**: 与 IDEA-01 形成完美 sibling，分别从 "保哪些 mode" 与 "保的方向" 两个角度修补 v1。给 W-DiffPolicy 提供 "选择性 mode 保留" 的差异化卖点。

---

### Rank 3: NeoRL-2 Transfer Stress Test for OT-Regularized Offline RL (IDEA-09)
- **Anchored critique**: CRITIQUE-10, CRITIQUE-12 — 把 NeoRL-2 从 "更难 benchmark" 变成检验 OT transfer 假设的因果压力测试
- **Thesis**: OT alignment 在 near-real-world 下可能提高 marginal similarity 却降低 Bellman-relevant coverage；提出 transition-critical OT filtering
- **Gap addressed**: G9 (NeoRL-2 上 OT-reg 方法未被系统测量)
- **Core mechanism**: NeoRL-2 source-target mixtures + 三种 OT 比较 + TD-error/advantage-weighted cost
- **Non-obvious because**: OTDF 做 transition-level OT 但 transition similarity ≠ Bellman usefulness
- **Contribution type**: cross-domain paper + empirical finding + new formulation
- **Theorem scaffold**: Empirical hypothesis—lower transition OT distance 与 final return 负相关 (除非 cost 含 value relevance)
- **Risk**: HIGH — NeoRL-2 复现工作量大；6-8 周窗口紧
- **Effort**: 7-8 person-weeks
- **Closest work**: P24 OTDF — delta: dataset filtering distribution alignment → value-relevant transfer validation
- **He Score**: Longevity 4 + Passion 4 + Application 5 + Uniqueness 4 = **17/20**
- **Anti-pattern flags**: 工作量风险高，但无 anti-pattern
- **Quick novelty**: LIKELY NOVEL
- **Why this ranks #3**: 真实场景影响最大；NeurIPS 偏好 cross-domain stress test。但作为 v1 的扩展实验段，不是 main paper。

---

### Rank 4: KL-Mode Collapse in Continuous Control (IDEA-10)
- **Anchored critique**: CRITIQUE-05, CRITIQUE-01 — 检验 P19 LM 定理是否真迁移到 control
- **Thesis**: KL-regularized offline RL 的 mode collapse 取决于 mode value gap、support overlap、Q-guidance strength；KL 不一定必然 collapse
- **Gap addressed**: G10。为 W-DiffPolicy 提供更严谨的理论动机
- **Core mechanism**: 构造 two-mode continuous MDP，解析比较 forward/reverse KL、score-matching BC、W₂ 的 optimal mass allocation；扩展到 Kitchen/AntMaze mode-controlled splits
- **Non-obvious because**: control 的 state-conditioned action modes ≠ sequence modes；Q term 改变最优 mass 分配
- **Contribution type**: theoretical result + empirical finding
- **Theorem scaffold**: KL-reg 次优 mode mass `m₂* → 0` iff `ΔQ/τ > θ(behavior mass ratio)`；W₂ 的阈值依赖 mode separation
- **Risk**: MEDIUM — 理论范围较窄但适合作为 W-DiffPolicy 引言性 theorem 或 short paper
- **Effort**: 4-6 person-weeks
- **Closest work**: P19 KL-Mode-Collapse — delta: LM → continuous control + Q-aware analysis
- **He Score**: Longevity 5 + Passion 5 + Application 3 + Uniqueness 4 = **17/20**
- **Anti-pattern flags**: none
- **Quick novelty**: LIKELY NOVEL
- **Why this ranks #4**: 提供 W-DiffPolicy 的理论 backbone——证明何时 KL 真的 collapse、何时不会。可作为 v1 论文的 motivation theorem 或独立小论文。

---

### Rank 5: Reverse-Path Regularization Is Not Terminal Policy Regularization (IDEA-02)
- **Anchored critique**: CRITIQUE-02
- **Thesis**: Reverse diffusion path divergence 与 terminal action divergence 可解耦；path-regularized diffusion policy 仍发生 terminal mode loss
- **Gap addressed**: 挑战 BDPO 的根基假设
- **Core mechanism**: toy mixture MDP + D4RL conditional action slices + path/terminal divergence correlation 实验
- **Non-obvious because**: 直觉上 Markov kernel 接近应推 terminal 接近，但 reverse process 的 small local errors 在 low-density regions 累积成 mode-level mass shift
- **Contribution type**: theoretical result + empirical finding + diagnostic
- **Theorem scaffold**: 存在两 K-step reverse processes，`Σ_t KL ≤ ε` 但 terminal mixture weight error `||wθ-wβ||₁ ≥ c`，c 不随 ε 线性消失
- **Risk**: LOW — 即使只是 strong empirical counterexample 也能直击 BDPO 解释漏洞
- **Effort**: 3-4 person-weeks
- **Closest work**: P04 BDPO — delta: 不提出更快 policy，而验证 reverse-kernel KL 是否真测到了它声称的东西
- **He Score**: Longevity 4 + Passion 4 + Application 3 + Uniqueness 5 = **16/20**
- **Anti-pattern flags**: none
- **Quick novelty**: LIKELY NOVEL
- **Why this ranks #5**: 最高 uniqueness（无人做过），低风险/低成本/负结果可发表。是 W-DiffPolicy 的 sibling 反思之作——可作为附录性 theoretical result 增强主线。

---

### Rank 6: Occupancy-Aware Wasserstein Diffusion Policy (IDEA-03)
- **Anchored critique**: CRITIQUE-04, CRITIQUE-11
- **Thesis**: action-space W₂ 不足以保证 offline safety；提升到 discounted state-action occupancy 的 OT
- **Gap addressed**: v1 W-DiffPolicy 仍可能在动态系统中诱导 OOD state
- **Core mechanism**: short-horizon rollout / successor features 估计 ρπ；`min_π E[-Q] + λ W₂²(ρπ, ρβ)`
- **Non-obvious because**: 不需要 full model，short-horizon proxy 即可捕捉 compounding shift
- **Contribution type**: new formulation + new method + theoretical result
- **Theorem scaffold**: `J(β)-J(π) ≤ O(W₂(ρπ,ρβ)) + O(value error)`，一般不能由 `E_s W₂(π(.|s),β(.|s))` 单独控制
- **Risk**: HIGH — occupancy estimation 噪声大；7-8 周窗口紧
- **Effort**: 7-8 person-weeks
- **Closest work**: P16 Rethinking OT in Offline RL — delta: 抽象 OT formulation → diffusion policy occupancy-regularized 训练
- **He Score**: Longevity 5 + Passion 4 + Application 3 + Uniqueness 4 = **16/20**
- **Anti-pattern flags**: none
- **Quick novelty**: LIKELY NOVEL
- **Why this ranks #6**: 最高 longevity（occupancy 视角是 fundamental），但风险高且时间紧。可作为 v2 或 follow-up 论文，本轮不优先。

---

## 其他保留 ideas（非 top-6 但仍可作 sibling/backup）

### Rank 7: ICNN-W₂ Reliability Pressure Test (IDEA-07) — 15/20
诊断性论文，挑战 v1 的技术底座；正负结果都可发表。**可作为 v1 的 ablation appendix 或独立 short paper。**

### Rank 8: Distill Without Losing Modes (IDEA-11) — 14/20
针对 RACTD 的 mode-aware 反击，需要 teacher 质量足够。**与 IDEA-01 自然连接：先做 IDEA-01 W-teacher，再做 IDEA-11 distillation。**

### Rank 9: ModeBench-ORL Diagnostic (IDEA-04) — 13/20
benchmark 论文，低风险但 venue 适配性中等（NeurIPS Datasets&Benchmarks track）。**可作为 IDEA-01 的 evaluation 附录或独立 short paper。**

### Rank 10: Sinkhorn-Diffusion Policy (IDEA-08) — 12/20
sibling alternative，理论化 entropic bias 是亮点。**可作为 v1 的 W₂-vs-Sinkhorn ablation。**

---

## Eliminated Ideas

| # | Idea | Stage | Reason |
|---|------|-------|--------|
| IDEA-05 | Budget-Normalized Diffusion ORL | He Filter | 综合 11/20 — Longevity 3 (audit 论文老化快)、Passion 2 (与作者方法-building 目标冲突)；audit 论文虽合理但与 W-DiffPolicy 主线无直接耦合 |

---

## Risk Distribution of Survivors

| Risk Level | Count | Ideas |
|------------|-------|-------|
| HIGH | 2 | IDEA-09 (NeoRL-2 stress test), IDEA-03 (Occupancy-aware) |
| MEDIUM | 6 | IDEA-01, IDEA-06, IDEA-07, IDEA-08, IDEA-10, IDEA-11 |
| LOW | 2 | IDEA-02 (Reverse-path), IDEA-04 (ModeBench) |

Diversity check：
- ≥ 2 HIGH risk/high reward: ✓ (IDEA-03, IDEA-09)
- ≥ 2 LOW risk/solid: ✓ (IDEA-02, IDEA-04)
- 至少 5 个不同 critique anchor 被覆盖: ✓ (CRITIQUE-01/02/04/05/07/08/10/11/13 共 9 个被覆盖)
- 论文类型覆盖: ✓
  - 诊断: IDEA-04, IDEA-07
  - 理论: IDEA-02, IDEA-10
  - 方法（v1 扩展）: IDEA-01, IDEA-06, IDEA-03
  - 实验缺陷修正: IDEA-05 (淘汰)
  - 跨域: IDEA-09
  - 蒸馏: IDEA-11

---

## Suggested Next Steps

1. **`/idea-screen` 在 Top 4 (IDEA-01, IDEA-06, IDEA-09, IDEA-10)** 上做多维度筛选（novelty 深度查新 + venue 模拟 reviewer + strategic fit）
2. **REFINE_TOP_N=2**（pipeline 配置）将选取 screening 后的 Top 2 做深度精炼
3. **作者实际 v1 设计与 IDEA-01 + IDEA-06 高度对齐**——这两个 idea 实际上是 v1 W-DiffPolicy 的两条强化路线，建议优先精炼这两条

---

## Methodology Notes

- Brainstorming model: gpt-5.5 with xhigh reasoning effort
- Filtering pipeline: Feasibility → Novelty quick-check → Impact → Prof. He 4-dim (threshold 12/20) → Anti-pattern check
- Novelty checks: 3 targeted WebSearch queries 验证 top idea 的核心 angle 未被做过；deep novelty 留给 `/idea-screen`
- Prof. He scores 反映了作者的明确兴趣（W-DiffPolicy v1 设计），passion 维度对 W₂+diffusion+多模态对齐的 idea 给高分
- 所有 idea 都 anchor 到 CRITICAL_ANALYSIS.md 中的具体 critique，非凭空想象
