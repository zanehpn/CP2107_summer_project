# When Does KL-Regularization Cause Mode Collapse in Continuous Control? A Threshold-Law Analysis with Wasserstein Comparison

> **Sibling Theory Paper to W-DiffPolicy** — IDEA-10 of pipeline
> **Initial Score**: 7.05 (REVISE)
> **Status after surgical theory bugs fixed**: ready for paper writing
> **Venue Target**: NeurIPS 2026 short paper / theory track
> **Codex thread**: `019e0d5d-3c03-79c0-9ec5-dfe562c2f248`
> **硬件**: 2× RTX 4090, 4-6 weeks (理论为主，实验为 sanity)

---

## Problem Anchor

- **Bottom-line**: P19 (KL-Regularized RL is Designed to Mode Collapse, arxiv 2510.20817, Oct 2025) 在 LM/化学 LM 上证 KL collapse；许多 follow-up offline RL 工作 (W-DiffPolicy, Q-DOT, BWD-IQL) 隐式将该结论推广到 continuous control 来 motivate W₂-OT；但**这一推广未独立验证**。
- **Must-solve**: 在 continuous-action state-conditioned MDP 中给出 (a) KL-regularized policy 的 mode mass allocation 解析公式；(b) 何时稀有 mode mass m_rare* < ε (collapse) 何时不会的 threshold；(c) W₂ regularization 的对应 threshold；(d) 在 D4RL controlled mixture 上实证验证 phase boundary。
- **Non-goals**: 不提出新算法；不重写 W-DiffPolicy 框架；不延伸到 LM
- **Constraints**: 2× 4090；4-6 周；D4RL Kitchen/AntMaze controlled split
- **Success**: (a) two-point/Gaussian-mixture MDP 解析定理 — KL 与 W₂ 各自 threshold；(b) Corollary 给出 KL collapse vs W₂ preserve 的 regime；(c) D4RL controlled split (2 tasks × 3 mixtures × 3 regs × 3 seeds = 54 runs) 实证 phase boundary

## Skeleton

- **State A**: P19 → control 的推广被默认假设
- **State B**: 推广有边界；本论文给出形式化 threshold + 实证对照；W-DiffPolicy 等方法的 motivation 何时合理何时不合理
- **Path**: setup (two-point + Gaussian mixture MDP) → analytical (Theorem 1 KL + Theorem 2 W₂) → corollaries (regime map) → D4RL controlled empirical (sanity check) → implications

## Setup (Section 2)

**Two-point MDP** (主理论 simplification):
- Single state s; action a ∈ {μ_1, μ_2} ⊂ R^d, |μ_2 - μ_1| = Δ_a
- Behavior policy: β = m_1 δ_{μ_1} + m_2 δ_{μ_2}, m_1 + m_2 = 1, WLOG m_2 ≤ m_1 (mode 2 = rare)
- Q-function: Q(μ_1) = q_1, Q(μ_2) = q_2
- ΔQ := q_2 - q_1 (signed; rare mode 可以高/低 value)
- Policy class: π = p_1 δ_{μ_1} + p_2 δ_{μ_2}, p_1 + p_2 = 1

**Gaussian mixture extension** (corollary):
- β(a) = m_1 N(a; μ_1, σ²) + m_2 N(a; μ_2, σ²)；σ → 0 退化到 two-point

**Regularization objective** (统一 reverse KL convention):
$$\max_\pi \mathbb{E}_{a \sim \pi}[Q(a)] - \tau \cdot D(\pi || \beta)$$

with D ∈ {reverse KL `D_KL(π || β)`, W₂², chunk-conditional W₂² (W-DiffPolicy main)}.

## Theorem 1 — KL Closed-Form (Section 3)

**Theorem 1 (KL mass allocation)**. Under the reverse-KL-regularized objective:
$$\pi_{KL}^*(a) \propto \beta(a) \exp(Q(a) / \tau)$$

For two-point setup:
$$p_2^* = \frac{m_2 \exp(q_2 / \tau)}{m_1 \exp(q_1 / \tau) + m_2 \exp(q_2 / \tau)} = \frac{m_2 \exp(\Delta Q / \tau)}{m_1 + m_2 \exp(\Delta Q / \tau)}$$

**Corollary (KL collapse threshold)**: 
$$p_2^* < \varepsilon \iff \Delta Q < \tau \cdot \log\frac{m_2(1-\varepsilon)}{\varepsilon m_1}$$

注意：当 mode 2 是 high-value (ΔQ > 0)，KL 不会 collapse mode 2；当 mode 2 是 low-value (ΔQ << 0)，KL 会 collapse。**P19 在 LM 上证的 collapse 来自 reward/reference probability 的不平衡 + 低正则强度，对应 control 中是 ΔQ << 0 + small τ 的 regime。**

**关键 finding**: KL 不会 collapse rare *high-value* mode；它 collapse 的是 rare *low-value* mode 与 Q-guidance 误排序的 rare mode。

## Theorem 2 — W₂ Two-Point Bang-Bang Threshold (Section 4)

**Theorem 2 (W₂ two-point threshold)**. For two-point setup, W₂²-regularized optimum:
$$\pi_W^* = \arg\max_\pi E_\pi[Q] - \alpha \cdot W_2^2(\pi, \beta)$$

W₂² between p = (p_1, p_2) δ-policy and β = (m_1, m_2) δ-policy:
$$W_2^2(\pi, \beta) = |p_1 - m_1| \cdot \Delta_a^2 = |p_2 - m_2| \cdot \Delta_a^2$$

(因为 transport 一单位 mass 从 μ_1 到 μ_2 cost = Δ_a²)

Objective:
$$\max_{p_2 \in [0, 1]} (p_1 q_1 + p_2 q_2) - \alpha \Delta_a^2 |p_2 - m_2|$$

**Bang-bang structure**: 
- 若 ΔQ > α Δ_a²：optimum = (0, 1)（全部转 mass 到高 value mode）
- 若 ΔQ < -α Δ_a²：optimum = (1, 0)（全部 mass 留在主 mode）
- 若 |ΔQ| ≤ α Δ_a²：optimum = (m_1, m_2)（保持 behavior，无 incentive 移动 mass）

**Corollary (W₂ preserve regime)**:
$$|\Delta Q| \leq \alpha \Delta_a^2 \implies \pi_W^* = \beta \implies \text{rare mode preserved at full } m_2$$

**关键对比 vs KL**:
- KL: 软 threshold (log scale)；任何 ΔQ < 0 都会 partially collapse rare mode
- W₂: 硬 threshold (bang-bang)；存在一个 dead zone `|ΔQ| ≤ α Δ_a²` 内 rare mode 被完全保住

## Theorem 3 — Gaussian Mixture Smoothing (附录)

In the σ > 0 mixture extension, the bang-bang structure smooths to a logistic-like S-curve. Threshold scales as:
$$\Delta Q \approx \alpha (\Delta_a^2 - O(\sigma^2)) \implies \text{collapse onset}$$

主线只引用此结果；详证留 Appendix。

## Corollary — Regime Map (Section 5)

| Regime | KL behavior | W₂ behavior | Implication |
|---|---|---|---|
| Δ_a 大, |ΔQ| 小 | 部分 collapse | preserve | **W-DiffPolicy 类方法显著优于 KL 类**——支持 Diffusion-QL → W₂ replacement 的 motivation |
| Δ_a 小, |ΔQ| 小 | 部分 collapse | 部分 collapse | 两者都不解决；mode preservation 不可达 |
| Δ_a 大, |ΔQ| 大 | collapse to high-Q mode | bang-bang 翻转到高-Q mode | Q-guidance 主导；rare mode 被 sacrifice 是合理的 (only if Q-estimation 准) |
| Δ_a 小, |ΔQ| 大 | collapse | bang-bang collapse | 两者都不能保 rare mode；需 mode-conditional W₂ (W-DiffPolicy 主) |

**Crucial implication for W-DiffPolicy main paper**: per-state Δ_a 在 D4RL Kitchen/AntMaze 多模态任务上通常大；ΔQ 在 multi-modal benchmark 上中等；因此本理论支持 W-DiffPolicy 在那些 task 上的 motivation 是 valid 的——但前提是 ΔQ_estimated 与真实 ΔQ 没有大偏差（Q-error 可让 KL 错杀 rare 高-Q mode）。

## P19 Differentiation (Discussion)

P19 在 LM 上证 reverse-KL collapse；本文展示该 collapse 在 control 中**有 threshold 与 regime 限制**：

> P19's collapse mechanism has a control-specific threshold law; direct transfer to offline continuous-control RL is valid only under specific value-gap, behavior-mass, and action-geometry regimes. We characterize these regimes analytically and validate them on D4RL controlled splits.

**不**主张 "P19 is wrong"；主张 "control 推广有结构化边界"。

## Sibling Positioning vs W-DiffPolicy

| Paper | Type | Contribution |
|---|---|---|
| W-DiffPolicy main (IDEA-01) | Method | Mode-Conditional W₂ + finite-sample recall theorem + D4RL Pareto |
| 本论文 (IDEA-10) | Theory | KL/W₂ collapse threshold + regime map + minimum D4RL validation |

不引入新方法。本论文为 W-DiffPolicy 等方法提供更严谨的 theoretical motivation：何时 W-OT 替换 KL 是合理的，何时不是。

## D4RL Empirical Sanity Check (Section 6)

**Reduced scope** (per reviewer feedback): synthetic phase diagram 主 + D4RL controlled split sanity:

- **Synthetic phase diagram (主 figure)**: two-point MDP，扫 (Δ_a, ΔQ, τ_KL, α_W) ∈ {3 × 3 × 3 × 3} × 3 seeds = 243 cells，每 cell 一次 closed-form 计算 + 1 numeric simulation；耗时 < 1 day on CPU
- **D4RL controlled split (sanity)**: Kitchen-mixed + AntMaze-medium-play 2 tasks × 3 mixture splits × 3 regs (KL Diffusion-QL / W₂ global / W-DiffPolicy main) × 3 seeds = 54 runs ≈ 4-5 days dual-4090
- **预期**: phase diagram 上预测的 collapse / preserve / bang-bang regime 与 D4RL 实测 m_rare recall 高度一致

## Theory-Experiment Alignment Matrix

| Claim | Type | Validation | Scale | Feasibility |
|---|---|---|---|---|
| Theorem 1 KL closed-form | Construction | closed-form derivation + 1d numeric verification | trivial | FEASIBLE |
| Theorem 2 W₂ bang-bang threshold | Construction + bang-bang analysis | closed-form + 1d numeric | trivial | FEASIBLE |
| Theorem 3 Gaussian smoothing | Construction (corollary) | closed-form + numeric | trivial | FEASIBLE |
| Regime map (corollary) | Phase-diagram analysis | synthetic phase diagram (243 cells) | 243 (CPU) | FEASIBLE |
| D4RL controlled split sanity | Empirical hypothesis | 2 tasks × 3 mix × 3 reg × 3 seed = 54 runs | 54 dual-4090 ≈ 4-5 days | FEASIBLE |
| ΔQ_estimated vs ΔQ_true 误差影响 | Empirical hypothesis (附录) | Q-bias controlled experiment | 12 runs | FEASIBLE |

无 NOT FEASIBLE。Reduced scope 让 theory paper 实验体量合理。

## Failure Modes & Mitigation

| Failure | Detection | Mitigation |
|---|---|---|
| 解析公式与数值仿真不一致 | numeric vs closed-form | 修订 derivation；honest report |
| D4RL 实测 phase boundary 与预测显著偏差 | per-task phase diagram | 报告 misalignment as gap；归因 Q-error / state heterogeneity |
| Δ_a 在某些 task 太小 | 实测 Δ_a | 该 task 标 "low-multimodality regime"，不主张 |

## Resource Estimate

- **Scale**: SMALL-MEDIUM (3-5 person-weeks)
- **Compute**: LOW-MEDIUM (大部分 CPU 解析；D4RL sanity 54 runs ≈ 4-5 days dual-4090)
- **Data**: 全 public

## Timeline (4-6 weeks)

| Week | Task |
|---|---|
| W1 | Theorem 1 + 2 derivation + 1d numeric verification |
| W2 | Theorem 3 + Corollary regime map + synthetic phase diagram (243 cells) |
| W3 | D4RL controlled split setup (2 tasks × 3 mixtures × 3 regs × 3 seeds) |
| W4 | D4RL runs complete + analysis |
| W5 | 写作 + 与 W-DiffPolicy main paper 对齐 narrative |
| W6 | 投 NeurIPS 2026 short paper / theory track |

## Differentiation Table (Novelty Argument)

| Work | Mechanism | Delta vs IDEA-10 |
|---|---|---|
| P19 (Oct 2025, 2510.20817) | KL-collapse 定理 in LM RL | 本: control-MDP threshold + W₂ 比较 + regime map + D4RL 实证 |
| 2602.02250 (Feb 2026) | Wasserstein-KL analogue in LQG control | 本: continuous-action multi-modal MDP + bang-bang W₂ + reverse KL + offline RL |
| Q-DOT / W-DiffPolicy / BWD-IQL | use W₂ as alternative to KL | 本: 不提方法；给出 W-OT 替换 KL 的 theoretical motivation 何时合理 |
| W-DiffPolicy main (IDEA-01) | Mode-conditional W₂ method | 本: theoretical sibling; 提供 motivation theorem |

**关键差异化**:
- 仅 P19 LM 推到 control = **未验证**；本论文显式给 threshold + regime
- 仅 LQG W-KL = **不涉及 multi-modal action distribution**；本论文核心研究多模态
- 不提新方法 = 与 W-DiffPolicy main paper 互补而非竞争
