# Research Proposal: Mode-Conditional W-DiffPolicy — Wasserstein-Regularized Diffusion Policies with Per-Mode Mass Preservation for Multi-Modal Offline Reinforcement Learning

## Problem Anchor

- **Bottom-line problem**: 离线 RL 中，behavior policy 经常是真实多模态分布（Kitchen 多种烹饪策略、AntMaze 多种导航路径、Adroit 多种抓取手势），但**当前 diffusion policy 的 KL/score-BC 行为正则在多模态上系统失效**（mode 漂移、稀有但高价值 mode 被吞），导致 SOTA 在 multi-modal benchmark 上离 oracle 还有显著差距。
- **Must-solve bottleneck**: **multi-modal 任务上稀有 value-relevant mode 的保留**——不是简单 "整体 distribution 距离更小"，而是 "**每个有价值的 mode 都被独立保住、且其 mass 不被平均化或漂移到主导 mode**"。
- **Non-goals**:
  - 不解决 inference 加速（让 RACTD/SORL/EDP 各自做）
  - 不重写 offline RL 的 Q-learning 框架（仍以 IQL/TD3+BC 风格 critic 为基础）
  - 不做 online fine-tuning（OTPR 的领域，避开）
  - 不主张 "首次 Q-cost OT"（OTPR/P16 已占）
- **Constraints**:
  - 硬件：2× RTX 4090（48GB 总显存）
  - 时间：6-8 周到 NeurIPS 2026 abstract / paper deadline
  - 主 benchmark：D4RL Kitchen / AntMaze / Adroit + 1-2 NeoRL-2 task
  - 已有：v1 W-DiffPolicy ICNN-OT 实现 + Diffusion-QL / BDPO baseline 复现
- **Success condition**: 论文同时满足：(a) 在 D4RL Kitchen/AntMaze 上 rare-mode recall ≥ 90%（v1 KL/global-W₂ baseline ≤ 50%）；(b) 平均 return 不弱于 v1 baseline；(c) 一个干净 finite-sample mode-recall theorem，error 项含 ICNN-W₂ estimator error；(d) toy synthetic mixture 上有可视化的 "global W₂ 也吞 rare mode" 反例 figure；(e) OTPR/Q-DOT/BDPO 在 mode-preservation 维度上劣于本方法。

## Skeleton

- **State A**: Reviewer 相信 diffusion policy 已能自然保多模态，行为正则只是更好的 distribution-distance 选择问题（KL → W₂）
- **State B**: Reviewer 接受 "global W₂ 也会吞稀有 mode"，必须以 per-mode 形式约束 mass，且本方法有形式化保证 + 实证证据
- **Skeleton Path**:
  1. Step 1 (Diagnosis): 即使 W₂ 替换 KL，全局 W₂ 仍可吞 rare value-relevant mode
  2. Step 2 (Formalization): latent mode variable + per-mode W₂ + mode mass regularizer
  3. Step 3 (Theory): finite-sample mode-recall bound 含 ICNN-W₂ + GMM 估计 error
  4. Step 4 (Empirical): synthetic + D4RL + 1-2 NeoRL-2，每层 oracle/global-W₂/KL ablation
  5. Step 5 (Differentiation): vs OTPR/Q-DOT/BDPO/P19/LOM 差异表

## Technical Gap

### 当前 diffusion offline RL 的失效点

- **Diffusion-QL (P01)** 用 score-matching BC 近似 KL：在 multi-modal behavior 上，BC loss 在密集主导 mode 处 dominate，稀有 mode 的 score 梯度被淹。
- **BDPO (P04, ICML 2025)** 把 KL 严格化为反向 transition kernel KL 累加：reverse-path 累积 KL ≤ ε 与 terminal 多模态 mass error 可以解耦（CRITIQUE-02）。
- **Q-DOT (P14, RLC 2025)** 用 ICNN-W₂ 替 KL 但只在 IQL（point-estimate policy）；其全局 W₂ 仍可通过 mass 平移代价换取距离最小，从而坍多模态。
- **OTPR (2502.12631)** 把 Q 当作 transport cost、policy 当作 OT map——但是 **online RL fine-tuning** 而非 offline 行为正则；且 Q-cost 不解决 "rare mode 是否被保住" 问题，只解决 "哪些 action 朝高 Q 走"。
- **LOM (P20, ICLR 2025)** 直接放弃多模态，用 GMM 选最佳单 mode——丧失 multi-modal robustness 与稀有但安全的 backup mode。
- **P19 (KL-Mode-Collapse, Oct 2025)** 在 LM 上证 KL 必 collapse；其类比未推广到 control，且没给 W₂ 替代的 mode-preservation 定理。

### 为何 naive 修复不够

- "更大模型 + 更多步数 diffusion"：Diffusion-QL 已用 K=5..50 step，问题是损失函数本身 mode-blind，加步数无法定向保 rare mode。
- "更小 KL 系数"：放小正则破坏 OOD-safety，引入 unsupported action。
- "score-matching BC + entropy bonus" (P12)：entropy 是无方向的；不知道哪个 mode 该保。
- "直接 W₂ 替代 KL"（v1 W-DiffPolicy）：仍是 *global* W₂；mass 可以平移而非逐 mode 保真。

### 最小充分干预

**Mode-Conditional W₂ 行为正则**：以 latent mode variable z 为锚，把全局 W₂ 拆成 per-mode conditional W₂ + 显式 mode-mass 项；用 ICNN OT map 估计每个 mode 的 conditional 距离（discriminator-free）。这是「最小机制」因为：
- 不引入新的 generative model（仍用 Diffusion-QL 框架）
- 不引入 online interaction（仍是 offline）
- 不依赖 Q-cost OT（避开 OTPR 重叠）
- 仅在 loss 中替换一项（KL/global-W₂ → per-mode W₂ + mode-mass）

### Frontier 替代路线

**Route A（采用，本提案）**：传统 Diffusion-QL 框架 + ICNN-OT + GMM 模式估计 + 新的 per-mode 损失。技术成熟、可证明、可在 6-8 周内复现并扩展。

**Route B（不采用）**：用 LLM 当 mode classifier（"这个 trajectory 属于哪种 cooking strategy"）。理由：(a) 引入 LLM 推理成本与 latent-mode 标注噪声；(b) 偏离 Problem Anchor（mode 应是 trajectory 几何/价值结构，不是语义标签）；(c) 对 D4RL action 这种低维连续空间杀鸡用牛刀。

### 核心技术 claim

> **"Mode preservation 不是 distribution-distance 的副产品，而是必须以 mode-conditional 形式独立约束的对象。Per-mode W₂ + mode-mass regularizer 在 ICNN-OT 实现下既给出 finite-sample recall 保证，又在 D4RL 多模态任务上保 ≥ 90% rare-mode recall。"**

### 所需最小证据

- **Toy theorem**：两 Gaussian mode + W₂ ε 接近的反例（global W₂ 可漂移；conditional W₂ 不可）
- **Finite-sample bound**：`Recall_mode ≥ 1 - O((ε_W + ε_ICNN)/Δ²) - O(ε_mode-est)`
- **Synthetic mixture MDP 实验**：4-mode 控制任务，oracle vs learned mode estimation 对照
- **D4RL Kitchen/AntMaze**：rare-mode recall + return 双维度对比 KL / global W₂ / Q-DOT-derived
- **NeoRL-2 1-2 task**（如 Pipeline 或 RocketLanding）：cross-domain transfer 的 zero-shot 评测

## Method Thesis

- **One-sentence thesis**: 我们证明并展示——multi-modal offline RL 中 global Wasserstein 行为正则会通过 mass 平移系统性丢失稀有 value-relevant mode，并提出 Mode-Conditional W₂ Diffusion Policy（per-mode conditional W₂ + 显式 mode-mass regularizer + 形式化 finite-sample mode-recall guarantee）从 distribution-matching 升级为 mode-preserving。
- **Why smallest adequate**: 仅替换损失函数中的一个项，无新生成模型、无在线交互、无 Q-cost 引入。
- **Why timely**: 2025-2026 三股力量汇聚——P19 给出 KL 在 LM 上 collapse 定理但缺 control 推广；Q-DOT/BWD-IQL/SWFP 把 W₂ 引入 RL 但仅止于 global 形式；diffusion offline RL 已成为标配 (Diffusion-QL/BDPO/SORL)。Mode-Conditional W₂ 是把这三条线汇总到 *per-mode formalization* 的自然下一步。

## Contribution Focus

- **Dominant contribution**: **Mode-Conditional W₂ 行为正则 + 形式化 finite-sample mode-recall 定理**——以 per-mode conditional W₂ 替代 global W₂/KL，证明 mode-preservation 在 ICNN-W₂ + GMM mode 估计 error 下的 recall 下界。
- **Optional supporting contribution**: **ModeBench-ORL 评测协议（轻量版）**：trajectory embedding clustering + per-mode return + rare-mode recall 三个指标，作为 Section 4 的核心评估，附带 oracle-mode 和 unsupervised-mode 两套 protocol 供 reviewer 验证稳定性。
- **Explicit non-contributions**:
  - 不主张 "首个 W₂ + diffusion policy"（v1 框架已有；本提案突出 per-mode 化）
  - 不主张 "Q-cost OT"（OTPR 已占）
  - 不引入 inference 加速（与 RACTD/SORL 正交）
  - 不做 online RL（与 OTPR 划清界限）
  - 不做完整 NeoRL-2 cross-domain 系统压测（IDEA-09 的工作；本文只取 1-2 task 作 transfer 验证）

## Proposed Method

### Complexity Budget

- **Frozen / reused**:
  - Diffusion-QL backbone: K-step DDPM action denoiser ε_θ(a_k, k, s)
  - IQL-style critic Q_φ(s,a) for policy improvement term
  - ICNN OT potential η_ψ (Makkuva 2020) for W₂ estimation, discriminator-free
- **New trainable components** (≤ 2):
  1. **Mode encoder** q_ξ(z | s, τ)：把 trajectory τ embedding 投到 GMM latent mode（K=4..8 个 mode），用 unsupervised clustering 初始化、与 policy 联合精调
  2. **Per-mode ICNN potential** η_ψ_z：每个 mode 一个轻量 ICNN（共享前几层，最后一层 mode-conditional），实现 conditional W₂ 估计
- **Tempting additions intentionally NOT used**:
  - LLM mode classifier
  - online RL fine-tuning
  - Q-aware ground cost (避开 OTPR)
  - consistency distillation (RACTD 路线)
  - 多 critic ensemble (P12 路线)

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Offline Dataset D = {(s, a, r, s', τ)}                            │
│   τ = recent k-step state history (or full trajectory id)         │
│         │                                                          │
│         ▼                                                          │
│ ┌──────────────────┐         ┌──────────────────────┐            │
│ │ Mode Encoder     │         │ Q-Network Q_φ(s,a)   │            │
│ │ q_ξ(z|s,τ)       │         │ (IQL-style critic)   │            │
│ │ K-mode GMM       │         └──────────────────────┘            │
│ └──────────────────┘                                              │
│         │                                                          │
│         ▼                                                          │
│ ┌──────────────────────────────────────────────────────────┐     │
│ │ Diffusion Policy π_θ(a | s)                              │     │
│ │   K-step denoising: ε̂ = ε_θ(a_k, k, s)                   │     │
│ │   sample a_diff ~ π_θ                                     │     │
│ └──────────────────────────────────────────────────────────┘     │
│         │                                                          │
│         ▼                                                          │
│ ┌──────────────────────────────────────────────────────────┐     │
│ │ Mode-Conditional ICNN OT                                  │     │
│ │   for z = 1..K:                                           │     │
│ │     T_z = ∇η_ψ_z (gradient of convex potential)           │     │
│ │     W₂²(π_θ(a|s,z), β(a|s,z)) = E[||a_diff - T_z(a_β)||²]│     │
│ └──────────────────────────────────────────────────────────┘     │
│         │                                                          │
│         ▼                                                          │
│ Loss = -E[Q(s, a_diff)]                                           │
│        + α · Σ_z m_β(z|s) · W₂²(π_θ(a|s,z), β(a|s,z))             │
│        + γ · D_TV(m_θ(z|s), m_β(z|s))   ← mode-mass 正则           │
│        + β · L_DSM (diffusion score-matching, IQL-style)          │
└─────────────────────────────────────────────────────────────────┘
```

### Core Mechanism

#### Module 1: Mode Encoder q_ξ(z | s, τ)

- **Input**: state s 与 trajectory snippet τ（最近 k=10 步 state-action pairs）
- **Output**: 离散 mode 概率 m_θ(z|s,τ) 在 K=4..8 mode 上
- **训练**: 两阶段：(a) 先用 trajectory clustering（最简单 GMM on trajectory embedding 由小 LSTM/transformer 给出）初始化；(b) 与 policy 联合 fine-tune，用 EM-like 迭代避免 collapse
- **关键设计**: K 是超参，用 BIC/elbow on synthetic + Kitchen 选；oracle protocol 直接用 dataset 中的 task-id（Kitchen 有 task labels）作 sanity check

#### Module 2: Per-mode ICNN Potential η_ψ_z

- **Input**: action a (or noisy action a_k), conditioned on s 和 z
- **Output**: convex potential 标量；其梯度 T_z = ∇η_ψ_z 是 per-mode OT map
- **训练**: 借 Q-DOT 的 dual formulation（discriminator-free, Brenier 定理），按 mode 分批：每个 batch 内对每个 z 单独估计 W₂
- **实现细节**: K 个 ICNN 共享前 2 层（state encoder），最后 1 层 mode-conditional；显存 < 1GB extra

#### Loss & Training Signal

```
L_total(θ, ξ, ψ) = 
    L_RL:          - E_{s,a_diff~π_θ} [ Q_φ(s, a_diff) ]
    + L_modeW2:    + α · Σ_{z=1..K} m_β(z|s) · W₂²_ψ_z(π_θ(a|s,z), β(a|s,z))
    + L_modeMass:  + γ · D_TV(m_θ(z|s), m_β(z|s))
    + L_DSM:       + β · E_{a_β,t} [ || ε - ε_θ(a_β + σ_t·ε, t, s) ||² ]
```

其中 m_β(z|s) 来自 q_ξ 在 dataset 上的 marginal；m_θ(z|s) = Σ_a π_θ(a|s) · q_ξ(z|a,s)。

#### Inference Path

测试时：(a) 给 s，diffusion policy K-step 采样 a；(b) Mode encoder 仅用于训练，inference 时不需要——这是关键工程优势：**training-time mode-aware regularization, inference-time vanilla diffusion sampling**。

### Optional Supporting Component: ModeBench-ORL（轻量评估协议）

- **Input**: 任一 offline RL agent 的 rollout
- **Output**: 三个指标 + figure
  - **Rare-mode Recall@τ**: dataset 中 mass 5-20% 的 mode 在 rollout 中的 recall（需 trajectory clustering 用同 K 与 q_ξ）
  - **Mode-mass Error**: ||m_θ - m_β||₁
  - **Per-mode Return**: 不只是 average return，每个 mode 独立的 return 分布
- **训练 signal**: 无需训练，纯评估
- **Why no contribution sprawl**: 这是 Section 4 的评估底层而非独立贡献；论文标题不涉及 benchmark；附录提供 oracle-mode 和 unsupervised-mode 两套 protocol 用于 reproducibility

### Modern Primitive Usage

- **Diffusion 的角色**: 多模态 expressive policy class（已是 SOTA 标配，不算 frontier-decoration）
- **ICNN-OT 的角色**: discriminator-free conditional W₂ estimator（核心新颖性）
- **Per-mode latent z 的角色**: 把 "mode preservation" 这个抽象目标转为可优化、可度量、可证明的对象——这是 frontier 转向 *structured generative model regularization* 的具体落点

### 集成到 base generator

- **Frozen**: Diffusion-QL 的 ε_θ 网络结构、Q_φ critic 结构、IQL Bellman update
- **Trainable**: q_ξ + η_ψ_z + 微调 ε_θ
- **Training stages**:
  - Stage 1（前 ~30%）: warm up Diffusion-QL on KL-BC（v1 baseline 训练）
  - Stage 2（中 ~50%）: switch to mode-conditional W₂ loss，q_ξ 冻结于 Stage 1 trajectory clustering
  - Stage 3（后 ~20%）: 联合 fine-tune（q_ξ + η_ψ_z + ε_θ），learning rate 减小
- **Inference**: 仅用 ε_θ；q_ξ 与 η_ψ_z 都是 training artifacts

### Failure Modes and Diagnostics

| Failure | 检测 | Mitigation |
|---|---|---|
| Mode encoder collapse（所有 z = 单一值）| `H[m_θ(z|s)]` 监控；< 0.5 nat 警告 | (a) GMM K 减小至 2；(b) 增大 γ；(c) reset q_ξ 用更强的 trajectory clustering 初始化 |
| ICNN training instability（Brenier 估计噪声）| validation W₂ 与 Sinkhorn / sliced-W 对比 | (a) 增大 batch；(b) 短期切换 Sinkhorn warm-up；(c) 减小 ICNN 深度 |
| Rare mode 在 dataset 中 mass 太低（<2%）| 监控 effective mode count `exp(H[m_β])` | 把这种 task 标记 "low-multimodality regime"，不主张本方法在此优势 |
| return 不升反降 | 监控 average return + per-mode return | (a) 减小 α；(b) 切回 Stage 1 KL-BC backup；(c) 在 paper 中 honest report tradeoff |

### Novelty and Elegance Argument

| 最近工作 | 核心机制 | 与本提案差异 |
|---|---|---|
| Q-DOT (RLC 2025) | ICNN W₂ + IQL（point-estimate policy） | 本提案: diffusion + per-mode |
| OTPR (Feb 2025) | Q-cost OT + diffusion + online RL fine-tuning | 本提案: 不用 Q-cost；offline RL；per-mode |
| BDPO (ICML 2025) | reverse-kernel KL 累加 + diffusion | 本提案: W₂ 替代 KL；per-mode；CRITIQUE-02 是 BDPO 的反思 |
| Diffusion-QL (ICLR 2023) | KL-BC + diffusion | 本提案: per-mode W₂ replaces KL-BC |
| LOM (ICLR 2025) | GMM mode + 选最佳单 mode | 本提案: GMM mode + 保留所有 value-relevant mode |
| Latent Diffusion ORL (ICLR 2024) | latent skill + Q-learning | 本提案: 不压缩 action space；显式 mode preservation 约束 |
| P19 (KL-collapse, Oct 2025) | LM RL 上 KL collapse 定理 | 本提案: control-MDP 推广 + W₂ 修复 + 形式化定理 |
| SWFP (Oct 2025) | flow + W₂ JKO + online | 本提案: diffusion + offline + per-mode |

**Why elegant**: 仅在 Diffusion-QL 框架的 loss 中替换一项；inference-time 零 overhead；新 trainable 部分总共 < 1M 参数；定理与实现一一对应（每个 error term 都有 ablation）。

## Theoretical Grounding

### Part T1 — Formalizability Scan

| Component | Formal object | Draft |
|---|---|---|
| Mode encoder q_ξ | KL/GMM 似然 + EM 收敛 | `L_mode = -E[log Σ_z q_ξ(z|s,τ) · N(τ; μ_z, Σ_z)]`；EM converges to local optimum under standard conditions |
| Per-mode W₂ | ICNN-W₂ estimator error bound | `|Ŵ₂_ψ_z - W₂_true| ≤ ε_ICNN(N_z, d, depth)`，其中 N_z 是 mode z 内样本数；source: Makkuva 2020 Thm 3 + Vacher et al. 2021 |
| Mode-mass regularizer | TV bound | `D_TV(m_θ, m_β) ≤ ε_mass`；标准 TV |
| Mode-recall theorem | finite-sample lower bound | **Theorem (informal)**: 若 (a) 每对 mode 间 W₂ 距离 ≥ Δ，(b) 每个 mode z 的 ICNN-W₂ error ≤ ε_W (z)，(c) GMM mode-est error ≤ ε_mode-est，则 `Recall_mode(πθ; threshold τ) ≥ 1 - O(Σ_z ε_W(z) / Δ²) - O(ε_mode-est) - O(ε_mass)` 以高概率成立 |
| Diffusion score-matching loss | denoising matching consistency | `L_DSM` 标准 DDPM ELBO；非新颖 |
| Final policy improvement | offline RL safety bound（仅作 sanity check） | `J(β) - J(πθ) ≤ O(W₂(πθ, β)) + O(value error)`；本文不主张 occupancy-aware version (那是 IDEA-03 follow-up) |

### Part T2 — Assumption Inventory

| Claim | Assumption | Class | Empirical sanity check |
|---|---|---|---|
| Mode-recall theorem | 每对 mode separation ≥ Δ（mass center 间 ≥ Δ in W₂）| RESTRICTIVE — 在 dataset 真实多模态时通常满足；要求实际数据中 mode 分离度可测量 | 在 D4RL Kitchen 上量化 mode separation；若 < Δ_empirical 给 paper 的 disclaimer |
| ICNN-W₂ estimator bound | i.i.d. samples; finite VC of ICNN class | STANDARD（Makkuva 2020 + Vacher 2021）但在条件分布 setting 下需 careful 处理 | 通过 IDEA-07 风格压力测试（合成 Gaussian mixture）确认 estimator bias 可控 |
| GMM mode estimation | dataset behavior is approximately mixture of K modes | RESTRICTIVE — D4RL Kitchen 有 4-7 task-mode；AntMaze 有 multiple goal-conditioned paths；若 K 估计错误，mode-recall guarantee 退化 | 用 BIC/elbow 选 K；oracle-mode protocol 用 dataset 真 task-id 验证 |
| Q-function bounded error | Q error ≤ ε_Q（standard offline RL assumption）| STANDARD | 用 IQL-style 离线 critic 训练；report Q-residual |
| Trajectory embedding for clustering | LSTM/transformer 给的 embedding 足够分离 mode | UNVERIFIED | 在 Kitchen 上 visualize embedding (t-SNE/UMAP)；如果 mode 没分开则论文 invalid 该 task |

### Part T3 — Theory-Experiment Alignment Matrix

| Theoretical Claim | Claim Type | Standard Validation Protocol | Required Scale | Feasibility | Flag |
|---|---|---|---|---|---|
| Mode-recall lower bound `Recall_mode ≥ 1 - O((ε_W + ε_ICNN)/Δ²) - O(ε_mode-est)` | Generalization-bound-like / sample complexity | (a) Synthetic Gaussian mixture MDP (4-mode)：扫 mass scale Δ ∈ {0.5, 1, 2, 5}，扫 N_per_mode ∈ {100, 500, 2000, 10000}，对照 oracle-mode 与 learned mode；(b) D4RL Kitchen 上 mode separation 实测 + theorem 预测 | 4 Δ × 4 N × 3 seed = 48 runs synthetic + 5 D4RL tasks × 3 seed | FEASIBLE | none |
| Per-mode W₂ < global W₂ 反例 | Approximation ratio / construction | 解析构造 + 数值验证：2-mode mixture，证 global W₂ → 0 但 mode mass error → 0.5 | 1 closed-form + 5 numeric instances | FEASIBLE | none |
| ICNN-W₂ bias in finite samples + multi-mode | Information-theoretic / estimator bound | Synthetic Gaussian mixture：mode count m ∈ {2, 4, 8}，sample N ∈ {1k, 10k}，对照 ICNN-W₂ 与 closed-form W₂ | 6 conditions × 3 seed | FEASIBLE | none |
| Empirical hypothesis: rare-mode recall ≥ 90% on D4RL | Empirical hypothesis | D4RL Kitchen + AntMaze + Adroit，对比 Diffusion-QL/BDPO/Q-DOT-derived/global-W₂/no-mode-mass ablation；trajectory clustering 提供 mode label | 12 D4RL tasks × 5 baselines × 3 seed = 180 runs | FEASIBLE WITH CAVEATS — 双卡 6 周内只能跑约 80% 组合；按 priority 顺序：Kitchen-mixed > AntMaze-medium > Adroit-cloned > Hopper-medium > 余下 | none (但需 priority queue) |
| Empirical hypothesis: NeoRL-2 transfer 成功 | Empirical hypothesis | NeoRL-2 Pipeline + RocketRecovery 1-2 task：D4RL→NeoRL-2 zero-shot rare-mode preservation | 2 tasks × 3 seed | FEASIBLE | none |
| Inference-time overhead = 0 | Computational complexity | Wall-clock 测试：FPS 与 Diffusion-QL 对比 (5 batch sizes × 3 seed) | trivial | FEASIBLE | none |

无 NOT FEASIBLE 项；唯一 caveat 是 D4RL 全套 12 task × 5 baseline × 3 seed 在 6 周内仅完成 80%——优先 multi-modal heavy task，locomotion 单运动模式任务可放后。

## Evaluation Sketch

- **如何 validate**：synthetic mixture MDP（解析）→ D4RL multi-modal tasks（rare-mode recall + return）→ NeoRL-2 1-2 task（cross-domain transfer）三层链
- **关键指标**：(1) Rare-mode Recall@τ；(2) Average return；(3) Mode-mass Error；(4) Per-mode return spread
- **success looks like**：Kitchen-partial 上 rare-mode recall ≥ 90% 且 average return ≥ Diffusion-QL；toy figure 显示 global-W₂ 收敛后仍丢 rare mode 而 mode-conditional 不丢
- **failure looks like**：rare-mode recall 与 baseline 差异 < 10pp；或 average return 显著下降 (> 5%)；或 ICNN-W₂ training 不稳

## Resource Estimate

- **Scale**: MEDIUM（5-7 person-weeks 到 v1 paper）
- **Compute**: MEDIUM（2× RTX 4090，48GB；diffusion policy + critic + K=4-8 ICNN 显存 < 12GB；D4RL 全套 12 tasks × 3 seed 约需 5 天双卡）
- **Data**: 全部 available（D4RL public + NeoRL-2 public）
