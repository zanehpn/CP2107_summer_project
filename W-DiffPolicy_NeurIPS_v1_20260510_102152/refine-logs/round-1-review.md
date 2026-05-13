# Round 1 Review

**Reviewer**: gpt-5.5 via Codex MCP (xhigh reasoning)
**Codex thread**: `019e0d5d-3c03-79c0-9ec5-dfe562c2f248`

## 评分

| Dimension | Weight | Score | Weighted |
|---|---:|---:|---:|
| 1. Problem Fidelity | 15% | 8.5 | 1.275 |
| 2. Method Specificity | 25% | 6.5 | 1.625 |
| 3. Contribution Quality | 25% | 7.0 | 1.750 |
| 4. Frontier Leverage | 15% | 8.0 | 1.200 |
| 5. Feasibility | 10% | 6.5 | 0.650 |
| 6. Validation Focus | 5% | 6.0 | 0.300 |
| 7. Venue Readiness | 5% | 8.0 | 0.400 |
| **OVERALL** | | | **7.45** |

**Verdict**: REVISE

## Critical Weaknesses

### CRITICAL #1: z 的内部不一致（Method Specificity）
方法说 `π_θ(a|s,z)` + per-mode W₂，但 inference 是 "vanilla diffusion" + zero overhead。两者不能同时为真，除非 z 仅是训练期 partition 而非 policy condition。Fix:
- **Preferred**: 让 z 是 inference-time 轻量条件，rollout 时 sample `z ~ m_β(z|s)` 后 sample `a ~ π_θ(a|s,z)`；overhead 称 "negligible" 而非 "zero"
- Alternative: 保持 vanilla inference，但移除 `π_θ(a|s,z)` 表示，把 per-mode 正则定义为 marginal policy 上的 stratified batch constraints

### CRITICAL #2: m_θ(z|s) 未定义（Method Specificity）
对生成的 policy action，没有 trajectory snippet τ，`q_ξ(z|s,τ)` 无法在训练或 rollout 时赋 policy-mode mass。Fix:
- 加 frozen mode classifier `c_ω(z|s, a_{0:H})`，或用相同 fixed trajectory encoder 处理 generated action chunks

### IMPORTANT #1: Contribution Sprawl（Contribution Quality）
主贡献开始累积副线：ModeBench、finite-sample theorem、ICNN bias study、NeoRL-2 transfer、P19 control 推广、BDPO path critique。Fix:
- 主论文 = diagnosis → method → theorem → focused validation
- Move ModeBench to "evaluation protocol" (not contribution)
- Drop P19/control-extension language 除非直接支持定理

### IMPORTANT #2: Feasibility Overload（Feasibility）
180 runs + synthetic sweep + ICNN bias test + NeoRL-2 + Adroit + theorem，6 周双 4090 上太乐观。Fix:
- Core: synthetic closed-form + Kitchen + AntMaze；baseline = Diffusion-QL / global W₂-DiffPolicy / KL/BDPO
- NeoRL-2 + Adroit → appendix stretch goals only

### IMPORTANT #3: Validation Focus 过于宽泛（Validation Focus）
不需要每个 benchmark family 来证明 per-mode W₂ work；reviewer 只需要 clean demonstration: global KL/W₂ 丢 rare mode 而本方法不丢。
- 把硬 threshold "rare-mode recall ≥ 90%" 换成 Pareto claim："dominates global W₂/KL on rare-mode recall at matched return"

## Simplification Opportunities

1. Delete q_ξ 的 joint EM fine-tune，**冻结 mode assignment** after offline clustering；mode 定义动而 mode preservation 同时证明会破坏 story
2. ModeBench 仅作 evaluation 不作 contribution；只用三个指标（rare-mode recall / mode-mass error / return）
3. 从核心实验中移除 Adroit + NeoRL-2

## Modernization Opportunities

1. 用 frozen contrastive trajectory encoder（小 SimCLR-style sequence autoencoder）做 mode discovery；比 raw GMM 稳定但仍轻量
2. 若 z 是 inference-time，用 learned mode token 或 FiLM conditioning 嵌入 diffusion denoiser——这是 conditional generation 最干净的现代 primitive
3. 不要 LLM classifier / Q-cost OT（drift + novelty 冲突）

## Drift Warning

**MINOR DRIFT**: 偶尔从 "mode-preserving W₂ diffusion policy" 漂移到 "new benchmark/protocol + P19 control theory + NeoRL-2 transfer"。
最严重的非-drift 问题是 z 的内部一致性。

## 单句论文中心

**"Global behavior distances can preserve average distributional closeness while losing rare valuable modes; mode-conditional W₂ prevents that with a measurable recall guarantee."**

<details>
<summary>Raw Reviewer Response (full)</summary>

详见 codex thread 19e0d5d-3c03-79c0-9ec5-dfe562c2f248；本文已 distill 为以上结构化分析。

</details>
