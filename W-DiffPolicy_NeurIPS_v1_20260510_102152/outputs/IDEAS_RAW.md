# Generated Research Ideas (Raw)

**Direction**: W-DiffPolicy: Wasserstein-Regularized Diffusion Policies for Multi-Modal Offline Reinforcement Learning
**Date**: 2026-05-09
**Model**: gpt-5.5 (xhigh reasoning)
**Landscape source**: outputs/LANDSCAPE.md（29 篇论文，6 主题，12 gap）
**Critique source**: outputs/CRITICAL_ANALYSIS.md（13 critiques across 4 dimensions）
**Ideas generated**: 11
**Codex thread ID**: `019e0d5d-3c03-79c0-9ec5-dfe562c2f248`

---

## IDEA-01: Mode-Conditional W₂ Diffusion Policy — 面向多模态保真的 Wasserstein 行为正则
- **Anchored critique**: CRITIQUE-01, CRITIQUE-07, CRITIQUE-13。直接回应 "return-only improvement 可能只是保留单一高 return mode" 的问题。
- **Thesis**: 我们证明并实证展示，conditional W₂ regularization 比全局 KL/W₂ 更能保留 value-relevant behavior modes，方法是在 state-action policy 之外显式引入 latent mode variable。
- **Gap addressed**: G1 (W₂×diffusion empty), G2 (mode-preservation theorem), G3 (mode-coverage diagnostic)。修正 v1 W-DiffPolicy 中 "全局 W₂ 自动保多模态" 的潜在弱点。
- **Core mechanism**: 学习轨迹级 latent mode `z`，优化 `L = L_diff + λ E_z[W₂²(πθ(a|s,z), β(a|s,z))] + η D(mπ(z|s), mβ(z|s)) - α Q(s,a)`，其中 mode mass regularizer 防止稀有高价值 mode 消失。
- **Non-obvious because**: 反对意见 "W₂ 本身已经能处理多峰分布"；问题是全局 W₂ 可通过 mass 移动**平均化** mode，而非逐 mode 保真。
- **Contribution type**: new method / theoretical result / diagnostic
- **Theorem scaffold**: 若 mode 间隔 ≥ Δ 且 conditional W₂ error ≤ ε << Δ²，则 `Recall_mode(πθ) ≥ 1 - O(ε/Δ²) - O(mode-est-error)`。
- **Risk**: MEDIUM — mode 分解质量是核心风险，但 skeleton experiment 可用 synthetic mixture + Kitchen 快速验证。
- **Effort**: 5-7 person-weeks
- **Closest work**: P14 Q-DOT — delta：从 IQL 的全局 ICNN-W₂ 转为 diffusion policy 的 mode-conditional OT。

---

## IDEA-02: Reverse-Path Regularization Is Not Terminal Policy Regularization
- **Anchored critique**: CRITIQUE-02。检验 BDPO-style reverse-kernel regularization 是否真约束最终 action distribution。
- **Thesis**: 我们证明 reverse diffusion path divergence 与 terminal action divergence 可以解耦，并通过构造性实验展示 path-regularized diffusion policy 仍会发生 terminal mode loss。
- **Gap addressed**: 挑战 P04 BDPO 的根基——"沿扩散反向过程累计 KL/W₂ 等价于行为策略 closeness" 的基础假设。
- **Core mechanism**: 构造 toy mixture MDP 与 D4RL conditional action slices，比较 `Σ_t D(Pθ(x_{t-1}|x_t,s), Pβ(x_{t-1}|x_t,s))` 与 `D(πθ(a|s),β(a|s))`、mode recall、return 之间的相关性。
- **Non-obvious because**: 直觉上每一步 Markov kernel 接近应推出 terminal 接近；但 reverse process 的 small local errors 可在 low-density regions 累积成 mode-level mass shift。
- **Contribution type**: theoretical result / empirical finding / diagnostic
- **Theorem scaffold**: 存在两个 K-step reverse processes，使得 `Σ_t KL(Kθ,t || Kβ,t) ≤ ε` 但 terminal mixture weight error `||wθ-wβ||₁ ≥ c`，其中 c 不随 ε 线性消失。
- **Risk**: LOW — 即使只得到强 empirical counterexample 也能直击 BDPO 解释漏洞。
- **Effort**: 3-4 person-weeks
- **Closest work**: P04 BDPO — delta：不提出更快 policy，而验证 reverse-kernel KL 是否真测到了它声称测的东西。

---

## IDEA-03: Occupancy-Aware Wasserstein Diffusion Policy
- **Anchored critique**: CRITIQUE-04, CRITIQUE-11。回应 per-state action OT 无法约束 induced MDP occupancy 的问题。
- **Thesis**: 我们展示 action-space W₂ regularization 不足以保证 offline safety，并提出 occupancy-aware OT，把 policy distance 从 `a|s` 提升到 discounted state-action occupancy。
- **Gap addressed**: v1 W-DiffPolicy 若只做 `W₂(π(a|s),β(a|s))`，仍可能在动态系统中诱导 OOD state distribution。
- **Core mechanism**: 用 learned dynamics 或 conservative successor embedding 近似 occupancy transport cost：`min_π E[-Qπ(s,a)] + λ W₂²(ρπ(s,a), ρβ(s,a))`，`ρπ` 用 short-horizon rollout / successor features 估计。
- **Non-obvious because**: 反对意见 "offline RL 无法可靠估计 ρπ"；但不需要完整 model，只需 short-horizon occupancy drift proxy 即可捕捉 action-OT 看不到的 compounding shift。
- **Contribution type**: new formulation / new method / theoretical result
- **Theorem scaffold**: 若 Bellman error 对 occupancy shift Lipschitz，则 `J(β)-J(π) ≤ O(W₂(ρπ,ρβ)) + O(value error)`，且一般不能由 `E_s W₂(π(.|s),β(.|s))` 单独控制。
- **Risk**: HIGH — occupancy estimation 噪声大，但若成立则贡献明显强于简单 KL→W₂ replacement。
- **Effort**: 7-8 person-weeks
- **Closest work**: P16 Rethinking OT in Offline RL — delta：从 abstract OT formulation 落到 diffusion policy 的 occupancy-regularized 训练目标。

---

## IDEA-04: ModeBench-ORL — Offline RL 的 Mode Preservation Diagnostic Benchmark
- **Anchored critique**: CRITIQUE-07, CRITIQUE-10, CRITIQUE-13。把 "多模态保留" 从口号变成可复现实验协议。
- **Thesis**: 我们表明现有 D4RL return 无法判断 multimodal preservation，并提出 value-conditioned mode recall、mode precision、occupancy drift 三个指标重新评估 diffusion/OT 方法。
- **Gap addressed**: G3 (no standard mode-coverage metric)，导致 KL/W₂/LOM 谁更 "保 mode" 无法比较。
- **Core mechanism**: trajectory embedding 聚类得到 behavior modes，再报告 `mode recall@τ`、`mode mass error`、`per-mode return`、`state-occupancy W₂/MMD`，并在 Kitchen/AntMaze 合成可控 mixture splits。
- **Non-obvious because**: 反对意见 "mode label 是人为的"；benchmark 同时提供 synthetic-labeled 与 unsupervised-labeled 两套 protocol 检验稳定性。
- **Contribution type**: diagnostic / experimental finding
- **Theorem scaffold**: Empirical hypothesis—相同 average return 下，Diffusion-QL/BDPO/Q-DOT-derived methods 的 mode recall 差异显著，且 return-mode-recall rank correlation `ρ < 0.5`。
- **Risk**: LOW — 实现成本低；负结果 "return ≈ mode recall" 同样关闭一个 fake gap。
- **Effort**: 3-5 person-weeks
- **Closest work**: P20 LOM — delta：LOM 选单 mode，本提案评估和量化 mode preservation。

---

## IDEA-05: Budget-Normalized Diffusion Offline RL — 重新打开 SOTA 比较
- **Anchored critique**: CRITIQUE-08, CRITIQUE-09。针对 diffusion policy 领域 compute/inference/tuning 不公平比较。
- **Thesis**: 我们证明许多 diffusion offline RL 的 reported gains 在 normalized NFE、update count、model size 与 offline model selection 下会显著改变。
- **Gap addressed**: 纠正 SORL/RACTD/BDPO/EDP 间不同采样步数、调参预算和 online validation 泄漏造成的结论偏差。
- **Core mechanism**: 构建统一 budget table（training FLOPs、NFE、wall-clock、参数量、Q ensemble 数、offline-only selection），在每个 budget slice 重画 score-efficiency Pareto frontier。
- **Non-obvious because**: 反对 "engineering audit"；但 diffusion offline RL 主要 claims 就是 speed/SOTA tradeoff，budget normalization 直接决定结论是否成立。
- **Contribution type**: experimental flaw correction / empirical finding
- **Theorem scaffold**: Empirical hypothesis—在 fixed FLOPs+NFE+offline-only-selection 下，至少一个 published SOTA ordering 反转。
- **Risk**: LOW — 复现工作量是主要风险，但实验协议清晰。
- **Effort**: 4-6 person-weeks
- **Closest work**: P06 RACTD — delta：不追求新 speedup，而检验 speedup 与 return 是否在公平预算下仍成立。

---

## IDEA-06: Value-Conditioned Wasserstein — 不要保留所有 Behavior Modes
- **Anchored critique**: CRITIQUE-13, CRITIQUE-01。修正 "mode preservation 总是好" 的错误目标。
- **Thesis**: 我们展示 naive mode preservation 会保留低价值/有害行为，并提出 value-conditioned W₂ 只保护 return-relevant modes。
- **Gap addressed**: 避免 v1 W-DiffPolicy 被批评为 "保多模态但不区分好坏 mode"。
- **Core mechanism**: 将 ground cost 改成 value-aware：`c((s,a),(s,a')) = ||a-a'||² + γ max(0, Qβ(s,a') - Qβ(s,a))`，或对 mode 权重加 `softmax(V_z/τ)`，让 OT mass 优先对齐高价值 behavior support。
- **Non-obvious because**: 反对 "引入 Q 会破坏 OT 的纯几何解释"；但 offline RL 的目标本来不是 density matching，而是 constrained improvement。
- **Contribution type**: new method / new formulation
- **Theorem scaffold**: `min_θ L_diff(θ) - α E[Q(s,aθ)] + λ W_{2,c_Q}²(πθ,β)`；conjecture：在 Q error bounded 时，value-conditioned W₂ 比 uniform W₂ 有更小 regret upper bound。
- **Risk**: MEDIUM — Q bias 可能污染 regularizer，可由 ensemble uncertainty gate 缓解。
- **Effort**: 5-7 person-weeks
- **Closest work**: P14 Q-DOT — delta：Q-DOT 用 W₂ regularize IQL，本提案把 value 信息放进 transport cost 本身。

---

## IDEA-07: ICNN-W₂ 何时失效 — Diffusion Policy 中 Neural OT Estimation 的压力测试
- **Anchored critique**: CRITIQUE-03, CRITIQUE-12。检验 ICNN-W₂ 是真实行为距离还是 estimator artifact。
- **Thesis**: 我们展示 ICNN-estimated W₂ 在 high-dim、multi-modal、finite offline data 下系统性低估 support mismatch，并提出 reliability diagnostics。
- **Gap addressed**: 直接打击 W-DiffPolicy 与 Q-DOT 共同依赖的技术底座——ICNN 能否稳定估计 offline RL 所需的条件 W₂。
- **Core mechanism**: 在 synthetic Gaussian mixtures、Kitchen action slices、Adroit/NeoRL subsets 上比较 ICNN-W₂、Sinkhorn、sliced W₂、MMD、held-out mode recall；报告 estimator bias 随 dim/sample/separation 的 scaling。
- **Non-obvious because**: 反对 "Makkuva 2020 已证明 ICNN 学 Brenier map"；但该证明不等于 finite-sample conditional offline RL 场景下的 estimator 可靠性。
- **Contribution type**: diagnostic / empirical finding
- **Theorem scaffold**: Empirical hypothesis—当 d 或 mode count m 增大时，ICNN-W₂ 与 true mixture W₂ 的 rank correlation 下降；rare-mode mass `<5%` 时低估 mode loss。
- **Risk**: MEDIUM — 若 ICNN 很稳，反过来增强 W-DiffPolicy 可信度。
- **Effort**: 4-6 person-weeks
- **Closest work**: P22 Makkuva 2020 — delta：从 ideal OT learning 转向 offline RL 条件分布压力测试。

---

## IDEA-08: Sinkhorn-Diffusion Policy — Entropic OT 不是 W₂ 的廉价替代
- **Anchored critique**: CRITIQUE-03, CRITIQUE-11。探索 G8，重点是解释 entropic bias 而非简单套 Sinkhorn。
- **Thesis**: 我们证明 entropic OT regularization 在 diffusion offline RL 中产生可控的 mode smoothing bias，并给出何时优于 ICNN-W₂ 的判据。
- **Gap addressed**: 针对 ICNN-W₂ 训练不稳和高维估计成本，提供 sibling alternative，同时不把 Sinkhorn 当黑箱距离。
- **Core mechanism**: minibatch Sinkhorn divergence 替代 ICNN potential：`L = L_diff - αQ + λ S_ε(πθ(.|s), β(.|s))`，分析 ε 对 mode merging 的影响；annealed ε 或 mode-separated batches 控制 bias。
- **Non-obvious because**: 反对 "Sinkhorn 已是成熟 distributional RL 工具"；但 policy behavior regularization 的关键不是分布距离本身，而是 entropic smoothing 是否吞掉稀有 action modes。
- **Contribution type**: new method / theoretical result
- **Theorem scaffold**: Conjecture—存在 critical entropy `ε* ≈ Δ²/log(1/w_min)`，当 ε > ε* 时 Sinkhorn regularizer 合并 separation 为 Δ、mass 为 w_min 的 rare mode。
- **Risk**: MEDIUM — 方法可能不赢 ICNN-W₂，但理论化 entropic bias 仍有独立贡献。
- **Effort**: 5-6 person-weeks
- **Closest work**: P28 Sinkhorn Distributional RL — delta：从 value distribution learning 转到 diffusion policy behavior regularization。

---

## IDEA-09: NeoRL-2 Transfer Stress Test for OT-Regularized Offline RL
- **Anchored critique**: CRITIQUE-10, CRITIQUE-12。把 NeoRL-2 从 "更难 benchmark" 变成检验 OT transfer 假设的因果压力测试。
- **Thesis**: 我们展示 OT alignment 在 near-real-world offline RL 中可能提高 marginal similarity 却降低 Bellman-relevant coverage，并提出 transition-critical OT filtering。
- **Gap addressed**: G9 (NeoRL-2 上 OT-regularized methods 未被系统测量；cross-domain OT 可能保错样本)。
- **Core mechanism**: 对 NeoRL-2 构造 source-target dataset mixtures，比较 transition-level OT、action-level W₂、occupancy-aware OT；加入 TD-error/advantage 加权 cost，避免 rare high-impact transitions 被过滤。
- **Non-obvious because**: 反对 "OTDF 已做 transition-level OT"；但 transition similarity 不等于 Bellman usefulness，尤其在 sparse reward / long horizon。
- **Contribution type**: cross-domain paper / empirical finding / new formulation
- **Theorem scaffold**: Empirical hypothesis—存在 tasks where lower transition OT distance 与 final offline RL return 负相关，除非 cost 包含 value/TD relevance。
- **Risk**: HIGH — NeoRL-2 复现实验和稳定性风险高，但成功后比 D4RL-only 更有 NeurIPS 分量。
- **Effort**: 7-8 person-weeks
- **Closest work**: P24 OTDF — delta：从 dataset filtering 的 distribution alignment 转向 value-relevant transfer validation。

---

## IDEA-10: KL-Mode Collapse in Continuous Control — 从语言模型定理到 Offline RL 反例/定理
- **Anchored critique**: CRITIQUE-05, CRITIQUE-01。检验 P19 的理论是否真迁移到 control。
- **Thesis**: 我们证明 KL-regularized offline RL 中 mode collapse 是否发生取决于 mode value gap、support overlap 与 Q-guidance strength，而非 KL 必然导致 collapse。
- **Gap addressed**: G10。修正 "KL 一定 mode collapse，因此 W₂ 一定更好" 的过度论证，为 W-DiffPolicy 提供更严谨动机。
- **Core mechanism**: 构造 two-mode continuous-action MDP，解析比较 forward KL、reverse KL、score-matching BC、W₂ regularization 下的 optimal policy mass allocation；扩展到 D4RL Kitchen/AntMaze 的 mode-controlled splits。
- **Non-obvious because**: 反对 "P19 已证明 KL collapse"；但 control 的 state-conditioned action modes 与 sequence-model modes 不同，Q term 改变最优 mass 分配。
- **Contribution type**: theoretical result / empirical finding
- **Theorem scaffold**: 在 two-mode MDP 中，KL-regularized optimum 的次优 mode mass `m₂*` 满足 `m₂* → 0` iff `ΔQ/τ > θ(behavior mass ratio)`；W₂ 的阈值依赖 mode separation `Δa`。
- **Risk**: MEDIUM — 理论范围可能较窄，但足以作为 W-DiffPolicy 强理论引言或独立 short paper。
- **Effort**: 4-6 person-weeks
- **Closest work**: P19 KL-Regularized RL is Designed to Mode Collapse — delta：从 LM/chemical-LM 转到 continuous-control offline RL。

---

## IDEA-11: Distill Without Losing Modes — OT-Regularized Consistency Distillation for Diffusion Policies
- **Anchored critique**: CRITIQUE-01, CRITIQUE-08。利用 RACTD threat window，重点放在 distillation-induced secondary mode loss。
- **Thesis**: 我们展示 one-step consistency distillation 会在 teacher 已保留 modes 的情况下二次丢失稀有 mode，并用 mode-aware OT distillation 修复。
- **Gap addressed**: G4/G11 (RACTD-style speedup × training-time OT regularization 正交性未验证)。
- **Core mechanism**: Student loss 不仅匹配 teacher action mean，而匹配 teacher conditional distribution：`L_student = E||fφ(x_t,t,s)-x_0^teacher||² + λ W₂²(πφ(.|s,z), πteacher(.|s,z)) - αQ`，报告 NFE-return-mode Pareto。
- **Non-obvious because**: 反对 "distillation 只是压缩 teacher"；但 one-step student 的函数类和采样温度天然偏向 dominant modes。
- **Contribution type**: new method / empirical finding
- **Theorem scaffold**: Empirical hypothesis—fixed teacher return 下，standard consistency distillation 的 rare-mode recall 显著低于 teacher，而 OT/mode-aware distillation 可在 1-NFE 下恢复 ≥ ?% recall。
- **Risk**: MEDIUM — 需要 teacher 质量足够好，可用现成 Diffusion-QL/BDPO teacher 降低风险。
- **Effort**: 6-8 person-weeks
- **Closest work**: P06 RACTD — delta：从 reward-aware speedup 转向 mode-preserving distillation objective。
