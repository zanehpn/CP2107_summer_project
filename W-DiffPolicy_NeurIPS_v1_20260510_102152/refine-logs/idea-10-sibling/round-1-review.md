# Round 1 Review — IDEA-10 (Sibling Theory)

| Dim | Weight | Score | Weighted |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 8.7 | 1.31 |
| Method Specificity | 25% | 6.2 | 1.55 |
| Contribution Quality | 25% | 6.8 | 1.70 |
| Frontier Leverage | 15% | 7.3 | 1.10 |
| Feasibility | 10% | 7.5 | 0.75 |
| Validation Focus | 5% | 7.0 | 0.35 |
| Venue Readiness | 5% | 6.8 | 0.34 |
| **Overall** | | | **7.05** |

**Verdict**: REVISE
**Drift Warning**: NONE
**Anchor**: preserved

## Core Issues (3 surgical theory bugs)

### Bug #1: ΔQ sign vs collapse direction (CRITICAL)
- 当前: ΔQ = q_2 - q_1，mode 2 = rare high-value；KL threshold "m₂* < ε iff ΔQ/τ large" → 反逻辑（高 value mode 不应 collapse）
- Collapse 应发生于：mode 2 lower value 或 Q-guidance/estimation 误排 rare mode 顺序
- Fix: 重新定义——collapse of rare mode = mass(rare) < ε；rare mode 可以 high-value 也可以 low-value，关键看 Q-guidance 的估计偏差

### Bug #2: KL convention 必须统一 (CRITICAL)
- 当前混用 forward `D_KL(π || β)` 与 reverse `D_KL(β || π)`
- π* ∝ β · exp(Q/τ) 来自 max E_π[Q] - τ D_KL(π || β)（reverse KL of π relative to β）
- Fix: 整本论文用 reverse KL `D_KL(π || β)`（这是 RL 标准），明确写出 max objective

### Bug #3: W₂ threshold 形式 (CRITICAL)
- 当前: `ΔQ > C τ / Δ_a² · log(1/ε)`
- 问题: 在 two-point 抽象下 W₂² shifting cost ≈ δ Δ_a²；optimum 通常 bang-bang (preserve 全部 mass 或 transport 全部)，不应有 log(1/ε) 平滑
- Fix: 改为 `ΔQ ≷ α Δ_a²` bang-bang threshold；Gaussian correction 留作 proposition

## Simplification

1. Drop score-matching BC 从主定理；放 empirical discussion
2. 用 two-point discrete (actions ∈ {μ_1, μ_2}) 作主理论；Gaussian 作 corollary
3. D4RL 缩为 2 tasks × 3 mixtures × 3 regs × 3 seeds = 54 runs；让 toy phase diagram 作 lead figure，D4RL 作 sanity check

## Theorem Hierarchy (建议)

- Theorem 1 (KL): closed-form mass allocation under reverse KL；rare mode mass formula
- Theorem 2 (W₂): two-point bang-bang threshold `ΔQ ≷ α Δ_a²`
- Corollary 1: regimes where KL collapses but W₂ preserves (Δ_a 大、ΔQ moderate)
- Corollary 2: regimes where both collapse (Δ_a 小或 ΔQ 极大)
- Corollary 3: regimes where neither collapses (ΔQ 小)

## P19 Differentiation (建议)

不要 claim "P19 is wrong"。改为：

> **P19's collapse mechanism has a control-specific threshold law; direct transfer to offline continuous-control RL is valid only under specific value-gap, behavior-mass, and action-geometry regimes. We characterize these regimes analytically and validate them on D4RL controlled splits.**

## Sibling Positioning vs W-DiffPolicy

清晰。这是 theory sibling—不引入新方法，仅给出 W-DiffPolicy 等方法的 motivation 何时合理何时不合理。

## Empirical Bridge

减小 scope：1 个 synthetic phase diagram (主) + 1 个 D4RL controlled split (sanity)。更多 task 不会救一个错误的定理。
