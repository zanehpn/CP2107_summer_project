# Skeleton — IDEA-10 KL-Mode Collapse in Continuous Control (Sibling Theory Paper)

## State A
- Reviewer 接受 P19 (KL-Regularized RL is Designed to Mode Collapse, Oct 2025) 在 LM/化学 LM 上的 mode-collapse 定理；许多 W₂-OT 论文（W-DiffPolicy v1, Q-DOT, BWD-IQL）将 P19 结论隐式推广为 "control 中 KL 也必然 collapse 因此 W₂ 必然更好"
- 但这个推广未被验证；control 的 state-conditioned action mode 与 sequence-level token mode 结构不同；Q-guidance 改变最优 mass 分配

## State B
- Reviewer 接受 control 中 KL collapse 不必然发生——存在 threshold 决定 collapse vs non-collapse；该 threshold 依赖 (a) mode value gap ΔQ, (b) 正则强度 τ, (c) behavior mass 比，(d) Q-guidance scale；W₂ 的 threshold 依赖 mode separation Δ_a
- 在 D4RL Kitchen/AntMaze mode-controlled splits 上实证验证 threshold law 与 P19 风格预测的成立/失败边界

## Skeleton Path
1. **Step 1 — Setup**: two-mode continuous-action MDP（state-conditioned action 分布带两个 separable mode；Q 值 ΔQ；behavior mass m_β = (m_1, m_2)）；这是 minimal control analog of P19 LM setting
2. **Step 2 — Analytical comparison**: 解析推导 KL-regularized optimum、forward KL / reverse KL、score-matching BC、W₂-regularized optimum 的次优 mode mass m₂*
3. **Step 3 — Threshold law (主定理)**: m₂* → 0 iff ΔQ/τ > θ_KL(m_β, ε)；W₂ regularization 的 threshold θ_W₂(Δ_a) 通常更宽松
4. **Step 4 — D4RL controlled split**: Kitchen/AntMaze 中人工 controlled mixture (mode 1 = 主导, mode 2 = rare value-relevant)；扫 ΔQ/τ 验证 threshold 预测
5. **Step 5 — Implications**: P19 在 control 中适用边界；何时 KL 真够用、何时必须 W₂；为 W-DiffPolicy 等方法提供更严谨理论 motivation

## 非可跳性
- 跳 Step 1 → 没有 control-specific 形式化对象；定理无法 apply
- 跳 Step 2 → 没有 analytical 对照；reviewer 认为是 heuristic
- 跳 Step 3 → 论文核心 theorem 缺失
- 跳 Step 4 → empirical bridge 断；reviewer 认为 toy-only
- 跳 Step 5 → reviewer 不知道这篇论文相对 P19 + W-DiffPolicy 的定位
