# Round 1 Refinement

## Problem Anchor (verbatim)

- **Bottom-line problem**: 离线 RL 中 behavior policy 经常是真实多模态分布，但当前 diffusion policy 的 KL/score-BC 行为正则在多模态上系统失效（mode 漂移、稀有 high-value mode 被吞），导致 SOTA 在 multi-modal benchmark 上离 oracle 还有显著差距。
- **Must-solve bottleneck**: multi-modal 任务上稀有 value-relevant mode 的保留——每个有价值 mode 都被独立保住、其 mass 不被平均化或漂移到主导 mode。
- **Non-goals**: 不解决 inference 加速；不重写 Q-learning 框架；不做 online fine-tuning；不主张 "首次 Q-cost OT"。
- **Constraints**: 2× RTX 4090；6-8 周到 NeurIPS 2026；D4RL Kitchen/AntMaze (核心) + Adroit/NeoRL-2 (附录)。
- **Success condition**: 在 D4RL multi-modal task 上 rare-mode recall × return Pareto frontier 严格 dominate global W₂/KL baseline；finite-sample mode-recall theorem 含 ICNN-W₂ + GMM mode-est error；toy 反例 figure 可视化 global W₂ 漂移；inference overhead 可忽略 (< 5%)。

## Anchor Check

- **原 bottleneck 仍是**: 多模态 offline RL 中稀有 value-relevant mode 的保留
- **修订后方法仍解决**: per-mode W₂ + mode-mass + frozen mode classifier 在训练时显式约束 per-mode preservation
- **拒绝的 reviewer drift**:
  - 不接受用 LLM classifier 替 GMM——会引入 latent-mode 标注语义噪声、与 D4RL 几何结构脱节
  - 不接受 Q-cost OT 引入——会与 OTPR/P16 重叠 + drift to online RL framing
  - 不接受 "stratified batch constraints" 路线（reviewer 的 alternative）——它放弃了 per-mode W₂ 的形式化对象，retreat 到 batch-level marginal constraint，损失定理可证明性

## Simplicity Check

- **dominant contribution after revision**: Mode-Conditional W₂ 行为正则 + 形式化 finite-sample mode-recall 定理（保持，未变）
- **components removed**:
  - q_ξ 的 joint EM fine-tune **删除**——改为冻结的 frozen contrastive trajectory encoder + GMM (offline clustering once)
  - **ModeBench 降级为 evaluation protocol**（仅 Section 4 内的 protocol；不在 paper Contributions list）
  - **NeoRL-2 + Adroit 推到附录**——主实验仅 Kitchen + AntMaze
  - **P19 / BDPO critique** 从主线移除——仅在 Related Work 各 1 段
- **components merged**:
  - q_ξ + 显式 inference-time mode condition：一个 light FiLM head 同时承担 training-time mode-conditional W₂ 与 rollout-time mode-aware sampling
- **拒绝的 reviewer drift**:
  - reviewer 建议把 πθ(a|s,z) 重写为 batch-stratified marginal constraint——这会削弱定理可证明性；保持 z 是 explicit conditional
- **smallest adequate route**: 仅替换 Diffusion-QL loss 中一项（KL-BC → per-mode W₂ + mode-mass）+ 加 1 个 frozen mode classifier + 1 个 lightweight FiLM head。新参数 < 2M (非 trainable + < 1M trainable)。

## Top-2 Issues Diagnosed

### Issue 1: z 的内部不一致 + m_θ(z|s) 未定义（CRITICAL）
- **Reader experience**: 读到 `π_θ(a|s,z)` 与 `Σ_z m_β(z|s) · W₂²(π_θ(a|s,z), β(a|s,z))` 时合理推论 policy 是 conditional；但 inference 又称 zero overhead 仅 `a~π_θ(a|s)`——逻辑断裂。同时 m_θ(z|s) 在 generated action 上无 trajectory snippet，rollout 时如何赋 mode mass 完全不清。
- **Hostile reviewer critique**: "Section 'Loss' 写 W₂² 是 conditional on z，但 'Inference Path' 称 zero overhead；且 m_θ(z|s) 形式上是 Σ_a π_θ(a|s) q_ξ(z|a,s) 但 q_ξ 输入是 trajectory snippet τ 而 rollout 时只有当前 s——这个 marginalization 不能在 rollout 时计算，训练时也只在 batch 级别近似。请明确 inference 时 z 是否被采样、如何采样、overhead 多大。"
- **Structural impact**: 破坏 Skeleton Step 2 (Formalization)——若 mode preservation 的目标分布 m_θ(z|s) 都不能在 rollout 中定义，第三步定理也无意义。

### Issue 2: Contribution sprawl + experimental overload（IMPORTANT）
- **Reader experience**: 论文同时 promise (a) per-mode W₂ 方法、(b) finite-sample theorem、(c) ModeBench 协议、(d) NeoRL-2 transfer、(e) P19 control 推广、(f) BDPO path critique。读者不知道 paper 的中心是什么。
- **Hostile reviewer critique**: "This paper claims to do X (mode-conditional W₂) but tests Y (NeoRL-2), Z (Adroit), W (BDPO critique), introduces V (ModeBench)—pick one. Method papers in NeurIPS need one focused thesis."
- **Structural impact**: 破坏 Skeleton Step 4 (Empirical chain) 与 Step 5 (Differentiation)——副线分散读者注意力，主 mechanism 反而被淹。

## Skeleton Gap Check

| Skeleton Step | Round 0 Coverage | Assessment | Round 1 Action |
|---|---|---|---|
| Step 1 Diagnosis | "Technical Gap" 一节 | adequate | 新增明确的 toy reflection ("global W₂ → 0 但 mass error → 0.5") 作 Section 1 leading figure |
| Step 2 Formalization | "Core Mechanism" + Loss 段 | **superficial**（z 不一致；m_θ 未定义）| **重写**：明确 z 是 inference-time conditional via FiLM；frozen mode classifier c_ω 定义 m_θ on generated action chunks |
| Step 3 Theory | "Theoretical Grounding" | adequate but 需 ICNN error 显式 | 显式重写定理 statement，把 ε_ICNN(N_z, d, depth) 写进 bound；分离 oracle-mode 与 learned-mode 两版定理 |
| Step 4 Empirical | "Evaluation Sketch" + T3 matrix | adequate but **overloaded** | 收紧主实验到 synthetic + Kitchen + AntMaze；附录 Adroit/NeoRL-2 |
| Step 5 Differentiation | 差异表 | adequate | 保留差异表；ModeBench 从 contribution 降为 protocol |

## Changes Made

### 1. Method 段重写：z 改为 inference-time FiLM 条件 + frozen mode classifier
- **Reviewer said**: CRITICAL #1 + CRITICAL #2
- **Action**:
  - Inference path 显式说明 `z ~ m_β(z|s)` 后 `a ~ π_θ(a|s,z)` via FiLM；overhead = 1 lightweight GMM sample + 1 FiLM linear (~ 0.05ms on RTX 4090) ≈ "negligible (< 5% extra wall-clock)" 而非 "zero"
  - 引入 **frozen mode classifier c_ω(z | s, a_{0:H})**：输入 state + H=8 步 generated action chunk；输出 K-dim mode 概率；冻结后用于训练时计算 m_θ(z|s) on generated samples，rollout 时不需要 (z 已 sampled from m_β before action generation)
  - m_β(z|s) 通过 **frozen contrastive trajectory encoder**（small SimCLR-style sequence autoencoder, ~1M frozen params）+ GMM (K=4-8, BIC 选)，在 dataset 上 offline 一次性训练完成
- **Reasoning**: 满足 reviewer Modernization Opp #1+#2 (frozen contrastive encoder + FiLM)；同时保住 per-mode W₂ 的形式化对象不退化
- **Impact**: Method Specificity 从 6.5 升到 ≥ 8；Frontier Leverage 保持 8

### 2. Contribution 收紧 + 实验 scope 缩小
- **Reviewer said**: IMPORTANT #1 + #2
- **Action**:
  - **Dominant contribution** = Mode-Conditional W₂ + finite-sample mode-recall theorem (单一)
  - ModeBench → Section 4 内的 evaluation protocol（不在 Contributions list）
  - 主实验：synthetic mixture MDP + Kitchen + AntMaze
  - 附录：Adroit + NeoRL-2 1-2 task（如时间允许）；BDPO path critique → Related Work 1 段；P19 control extension → Discussion 1 段
- **Reasoning**: 满足 Simplification Opp #1-#3
- **Impact**: Contribution Quality 7.0 → ≥ 8；Validation Focus 6.0 → ≥ 8；Feasibility 6.5 → ≥ 8

### 3. Validation 改为 Pareto claim
- **Reviewer said**: IMPORTANT #3
- **Action**: Success condition 从 "rare-mode recall ≥ 90%" 改为 "rare-mode recall × return Pareto frontier 严格 dominate global W₂/KL"；附录附 hard 90% 数字仅作参考
- **Reasoning**: Pareto 不依赖 absolute threshold；reviewer 友好；负结果（recall 高但 return 跌）也可发表
- **Impact**: Validation Focus 6.0 → ≥ 8

### 4. mode encoder 冻结
- **Reviewer said**: Simplification Opp #1
- **Action**: 删除 Stage 3 联合 EM；q_ξ (用 contrastive encoder + GMM 实现) 在 Stage 0 离线训练好后**全程冻结**；只 Diffusion-QL ε_θ + ICNN η_ψ_z 与 FiLM head trainable
- **Reasoning**: 避免 "mode 定义动而 mode preservation 同时证明" 的逻辑漏洞
- **Impact**: 论文逻辑更干净；定理只需考虑 q_ξ-induced mode 是固定的

## Revised Proposal

# Mode-Conditional W-DiffPolicy: Wasserstein Diffusion Policies with Per-Mode Mass Preservation for Multi-Modal Offline Reinforcement Learning

## Problem Anchor
[verbatim — 见上]

## Skeleton
- State A: reviewer 相信 diffusion 自然保多模态，行为正则只是 distance 选择问题
- State B: reviewer 接受 "global W₂ 也吞稀有 mode" 必须 per-mode 约束 mass，本方法有 finite-sample 保证 + 实证证据
- Path: Step1 Diagnosis → Step2 Formalization (z=FiLM cond + frozen c_ω) → Step3 Theory → Step4 Empirical (synthetic+Kitchen+AntMaze) → Step5 Differentiation

## Technical Gap

[同 round 0，简化版]

- Diffusion-QL/EDP/SORL/BDPO 用 KL/score-BC 行为正则——multi-modal 上 mode 漂移
- Q-DOT 用 ICNN-W₂ 但仅 IQL global，仍可 mass 平移坍多模
- OTPR 用 Q-cost OT 但 online；P16 用 Maximin Q-OT 但 abstract
- LOM 放弃多模态选单 mode；P19 在 LM 上证 KL collapse 未推到 control

**最小充分干预**: Mode-Conditional W₂ 行为正则——以 frozen mode classifier 显式锚定 mode，per-mode conditional W₂ via ICNN-OT，FiLM 条件让 inference 仍然轻量。

## Method Thesis

**One-sentence thesis**: 我们证明 global Wasserstein 行为正则在 multi-modal offline RL 中可通过 mass 平移系统性丢失稀有 value-relevant mode；提出 Mode-Conditional W₂ Diffusion Policy（frozen mode classifier + per-mode conditional W₂ via ICNN-OT + FiLM-based mode conditioning），从 distribution-matching 升级为 mode-preserving，inference overhead 可忽略 (< 5%)。

## Contribution Focus

- **Dominant contribution**: Mode-Conditional W₂ 行为正则 + finite-sample mode-recall 定理（含 ICNN-W₂ estimator error 与 mode-est error 的显式 bound）
- **Optional supporting**: 无独立支持贡献。Section 4 用 rare-mode recall + mode-mass error + return 三指标作 evaluation protocol（不在 Contributions list）
- **Explicit non-contributions**: 不主张首个 W₂+diffusion；不主张 Q-cost OT；不引入 inference 加速；不做 online RL；不做完整 NeoRL-2 cross-domain 系统压测；不重写 evaluation benchmark

## Proposed Method

### Complexity Budget

- **Frozen / reused**:
  - Diffusion-QL backbone: K-step DDPM denoiser ε_θ(a_k, k, s, **z** via FiLM)
  - IQL-style critic Q_φ(s,a)
  - **Contrastive trajectory encoder f_ω**（小 SimCLR sequence autoencoder, ~1M params, frozen after Stage 0）
  - **GMM** on f_ω embedding 作 m_β(z|s)（K=4-8, BIC 选, frozen）
  - **Frozen mode classifier c_ω(z|s, a_{0:H})**: 用 f_ω + GMM 在 generated action chunks 上推断 mode；推断 only 不训练
  - ICNN OT potential η_ψ (Makkuva 2020)
- **New trainable** (≤ 2 components):
  1. **Per-mode ICNN potential η_ψ_z**: K 个 ICNN 共享前 2 层（state encoder），最后 1 层 mode-conditional；总参数 ~ 0.5M
  2. **FiLM mode-conditioning head** in ε_θ: 给每个 diffusion residual block 加 1 层 FiLM (γ_z, β_z)；总参数 ~ 0.3M
- **Tempting additions intentionally NOT used**:
  - Joint EM fine-tune of q_ξ（删除）
  - LLM mode classifier
  - Q-aware ground cost
  - online RL fine-tuning
  - consistency distillation
  - multi-critic ensemble

### System Overview

```
Stage 0 (one-shot offline, 几小时):
  Dataset D → contrastive trajectory encoder f_ω → GMM(K=4-8) on traj-emb → m_β(z|s) frozen

Stage 1 + 2 (training):
┌────────────────────────────────────────────────────────────────┐
│ Offline Dataset D = {(s, a, r, s', traj-id)}                    │
│         │                                                        │
│         ▼                                                        │
│ ┌──────────────┐    ┌────────────────────┐                     │
│ │ frozen f_ω   │    │ Q-Network Q_φ(s,a)  │                     │
│ │ + GMM        │    │ (IQL-style)         │                     │
│ │ → m_β(z|s)   │    └────────────────────┘                     │
│ └──────────────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│ ┌─────────────────────────────────────────────────────┐        │
│ │ Conditional Diffusion Policy π_θ(a|s,z)             │        │
│ │   K-step denoising: ε̂ = ε_θ(a_k, k, s, z) via FiLM  │        │
│ │   Sample a_diff for each z                          │        │
│ └─────────────────────────────────────────────────────┘        │
│         │                                                        │
│         ▼                                                        │
│ ┌─────────────────────────────────────────────────────┐        │
│ │ Per-mode ICNN OT η_ψ_z                              │        │
│ │   T_z = ∇η_ψ_z (Brenier)                            │        │
│ │   W₂²_ψ_z(π_θ(a|s,z), β(a|s,z))                     │        │
│ └─────────────────────────────────────────────────────┘        │
│         │                                                        │
│         ▼                                                        │
│ Loss = - E[Q(s, a_diff)]                                        │
│        + α · Σ_z m_β(z|s) · W₂²_ψ_z(π_θ(a|s,z), β(a|s,z))       │
│        + γ · D_TV(m_θ(z|s) , m_β(z|s))                          │
│        + β · L_DSM (z-conditional)                              │
│ where m_θ(z|s) = E_{a~π_θ(.|s)}[c_ω(z|s, a_{0:H})]              │
│                                                                  │
│ Inference (test time):                                           │
│   1. Sample z ~ m_β(z|s)            (1 categorical sample)     │
│   2. a ~ π_θ(a|s,z) via K-step DDPM (FiLM-conditioned)         │
│   3. (overhead vs Diffusion-QL ≈ 1 lookup + 1 FiLM lin/block)   │
└────────────────────────────────────────────────────────────────┘
```

### Core Mechanism

#### Stage 0: Offline mode discovery (one-shot, ~几小时)

- **Step 0a**: Train contrastive trajectory encoder f_ω on dataset trajectories using SimCLR-style InfoNCE loss with augmentations (sub-trajectory masking, time-shift)；输出 d=64 traj-embedding。**冻结后不再变。**
- **Step 0b**: Fit GMM (K=4..8 by BIC) on f_ω(τ) for τ in dataset; get state-conditional mode marginals m_β(z|s) by averaging GMM posterior over trajectories passing through s (within ε-neighborhood)。**冻结后不再变。**
- **Step 0c**: Define **frozen mode classifier** c_ω(z | s, a_{0:H}) = GMM-posterior-of-f_ω(s, a_{0:H})；用 H=8 步 action chunk + state 计算 trajectory embedding。**inference-only，不训练。**

> **Crucial design choice**: f_ω + GMM + c_ω 都冻结。论文证明 mode preservation under fixed mode definition；不证明 mode discovery 本身。

#### Stage 1: Conditional diffusion policy with FiLM

- ε_θ(a_k, k, s, z) 标准 DDPM 但每个 residual block 加 FiLM(γ_z, β_z)
- Stage 1 (前 30%): warm up with KL-BC + uniform z sampling（让 ε_θ 学会条件依赖 z）
- Stage 2 (后 70%): switch to mode-conditional W₂ loss

#### Loss

```
L_total(θ, ψ_{1:K}) = 
    L_RL:        - E_{s, z~m_β(z|s), a_diff~π_θ(.|s,z)} [ Q_φ(s, a_diff) ]
    + L_modeW2:  + α · E_s [ Σ_{z=1..K} m_β(z|s) · W₂²_ψ_z(π_θ(a|s,z), β(a|s,z)) ]
    + L_modeMass: + γ · E_s [ D_TV( m_θ(z|s) , m_β(z|s) ) ]
                  where m_θ(z|s) = E_{a~π_θ(.|s)} [ c_ω(z | s, a_{0:H}) ]   (over a Monte Carlo batch)
    + L_DSM:     + β · E_{z, a_β~D|z, t} [ || ε - ε_θ(a_β + σ_t·ε, t, s, z) ||² ]
```

注：L_DSM 用的是 dataset 中按 z 划分的 conditional behavior：β(a|s,z) ≈ {(s,a) ∈ D : c_ω(z|s, a_{0:H}) > 0.5}。

#### ICNN-OT 实现细节

- 每个 z 一个 lightweight ICNN η_ψ_z（共享前 2 层 state encoder, 最后 1 层 mode-specific）
- Dual formulation (Q-DOT eq. 6-9)：W₂²_ψ_z = E[||a - T_z(a')||²] for a~π_θ(.|s,z), a'~β(.|s,z), T_z = ∇η_ψ_z
- Discriminator-free (Brenier convex potential)；training stable
- Per-batch: 每 mode z 内 sub-batch ≥ 32（小 batch 时切回 Sinkhorn warm-up）

### Inference Path（明确 negligible 而非 zero overhead）

```python
def act(s):
    # Step 1: sample mode (1 categorical sample, ~ 0.001ms)
    z ~ Categorical(m_β(z|s))   # m_β is frozen, lookup-cached

    # Step 2: K-step diffusion sampling conditional on z (FiLM)
    a = sample_diffusion(epsilon_theta, s, z, K_steps)

    return a
```

**Overhead vs Diffusion-QL**:
- m_β(z|s): O(K) lookup + categorical sample ≈ 0.001 ms
- FiLM (γ_z, β_z) per residual block: K_steps × N_blocks × FiLM linear ≈ 0.05 ms total on RTX 4090
- 总额外开销 < 5% wall-clock vs Diffusion-QL

### Optional Supporting: Section 4 内的 evaluation protocol（非 contribution）

仅在 Section 4 描述协议；不在 Contributions list；附录提供 oracle-mode (用 dataset task-id) 与 unsupervised-mode (用 c_ω) 双 protocol 验稳定性。
- Rare-mode Recall@τ: dataset 中 mass < 20% 的 mode 在 N rollouts 中至少出现 1 次的比例
- Mode-mass Error: ||m_θ̂ - m_β||₁
- Per-mode Return distribution

### Modern Primitive Usage

- **Diffusion**: 多模态 expressive class（SOTA 标配）
- **ICNN-OT**: discriminator-free conditional W₂（核心新颖）
- **Frozen contrastive trajectory encoder**: 现代 self-supervised 方法做 mode discovery，比 raw GMM-on-action 稳定但仍轻量
- **FiLM 条件**: conditional generation 的标准现代 primitive；让 inference 仍然轻量

### Integration into Diffusion-QL

- **Frozen**: ε_θ 主结构、Q_φ critic 结构、IQL Bellman update、Diffusion-QL 训练流程
- **Trainable**: η_ψ_z (K 个 ICNN) + ε_θ 中的 FiLM head + ε_θ fine-tune
- **Stage 1 (前 30%)**: warm up KL-BC + uniform z；让 ε_θ 学条件依赖
- **Stage 2 (后 70%)**: mode-conditional W₂ loss + L_RL + L_modeMass

### Failure Modes and Diagnostics

| Failure | 检测 | Mitigation |
|---|---|---|
| GMM mode discovery 不分离 | t-SNE/UMAP visualize；silhouette < 0.2 警告 | 切换到 task-id (Kitchen 有 task labels) 作 oracle protocol；论文中 honest 报告 |
| ICNN training instability | validation W₂_ψ_z 与 Sinkhorn 对比；偏差 > 50% 警告 | 短期 Sinkhorn warm-up；减 ICNN 深度 |
| Rare mode mass < 2% | exp(H[m_β]) 监控 | 标 task 为 "low-multimodality regime"；不在该 task 主张优势 |
| Return 不升反降 | per-mode return 监控 | 减 α；切回 Stage 1 KL-BC backup |
| FiLM 条件失败（z 无效）| z conditional 与 z marginal 的 action 分布距离 → 0 | 增 β (DSM 强度) 或 α；强制 conditional 学习 |

### Novelty and Elegance Argument

| Work | Mechanism | Delta |
|---|---|---|
| Q-DOT (RLC 2025) | ICNN W₂ + IQL global | 本: diffusion + per-mode + frozen mode classifier |
| OTPR (Feb 2025) | Q-cost OT + diffusion + online RL fine-tune | 本: 不用 Q-cost；offline；per-mode |
| BDPO (ICML 2025) | reverse-kernel KL accumulated + diffusion | 本: per-mode W₂ + FiLM cond |
| Diffusion-QL (ICLR 2023) | KL-BC + diffusion | 本: per-mode W₂ replaces KL-BC |
| LOM (ICLR 2025) | GMM mode + 选单 mode | 本: GMM mode + 保所有 value-relevant mode |
| Latent Diffusion ORL (ICLR 2024) | latent skill compression | 本: FiLM-based per-mode condition；不压缩 action |
| P19 (KL-collapse, Oct 2025) | LM RL KL collapse | 本: control 实证版 (Discussion only)；W₂ 修复 |
| SWFP (Oct 2025) | flow + W₂ JKO + online | 本: diffusion + offline + per-mode |

**Why elegant**:
- 仅替换 Diffusion-QL loss 中一项
- Stage 0 一次性 offline mode discovery，之后 frozen
- 新 trainable < 1M 参数
- Inference overhead < 5% wall-clock
- 定理与实现一一对应（每个 error term 都有 ablation）

## Theoretical Grounding

### Part T1 — Formalizability Scan

| Component | Formal object | Draft |
|---|---|---|
| f_ω + GMM mode discovery | InfoNCE 收敛；GMM EM 收敛；BIC 选 K | 标准；不主张新理论 |
| Frozen mode classifier c_ω | GMM posterior bound on c_ω(z|s, a_{0:H}) | classifier 误差 ε_clf 由 GMM-posterior 在 holdout 上估计 |
| Per-mode ICNN-W₂ estimator | Brenier-OT estimator error bound | `|Ŵ₂_ψ_z - W₂²_true,z| ≤ ε_ICNN(N_z, d, depth)`，N_z 是 mode z 内样本数；source: Makkuva 2020 Thm 3 + Vacher et al. 2021 |
| Mode-mass regularizer | TV bound | `D_TV(m_θ, m_β) ≤ ε_mass` 经训练后 |
| **Mode-recall theorem (主定理)** | finite-sample lower bound | **Theorem (informal)**: 设 (a) 每对 mode 间 pairwise W₂ 距离 ≥ Δ；(b) 每个 mode z 的 ICNN-W₂ estimator error ≤ ε_W(z)；(c) frozen mode classifier 错误率 ≤ ε_clf；(d) D_TV(m_θ, m_β) ≤ ε_mass，则对 any rare mode z* with m_β(z*) ≥ ρ_min:  `Recall_mode(πθ, z*) ≥ 1 - O(ε_W(z*) / Δ²) - O(ε_clf / ρ_min) - O(ε_mass / ρ_min)` 以高概率成立 |
| Diffusion DDPM ELBO | 标准 | 非新颖 |

### Part T2 — Assumption Inventory

| Claim | Assumption | Class | Empirical sanity check |
|---|---|---|---|
| Mode-recall theorem | 每对 mode pairwise W₂ ≥ Δ | RESTRICTIVE — D4RL Kitchen 等多模态任务通常成立；可 measure | 在 Kitchen 上量化 pairwise mode W₂；若 < Δ_empirical 在 paper disclaimer |
| ICNN-W₂ estimator | i.i.d. samples; bounded VC of ICNN | STANDARD（Makkuva 2020）但 conditional 设定下需 careful | IDEA-07 风格压力测试（合成 Gaussian mixture）控制 |
| Frozen mode classifier c_ω | classifier err ≤ ε_clf measurable on holdout | STANDARD | held-out task-id consistency |
| GMM K 选择 | dataset behavior 大致是 K-mode mixture | RESTRICTIVE — D4RL Kitchen 4-7 task；BIC + oracle (task-id) 双重验 | BIC + oracle protocol 双 paper 报告 |
| Q error bounded | offline RL 标准 | STANDARD | report Q residual |
| **(removed)** trajectory embedding 分离 mode | (变成 frozen Stage 0 后)每 task 一次性验证 | UNVERIFIED but pre-checked | t-SNE Stage 0 后 visualize；不通过则该 task 标 invalid |

### Part T3 — Theory-Experiment Alignment Matrix

| Theoretical Claim | Claim Type | Standard Validation Protocol | Required Scale | Feasibility | Flag |
|---|---|---|---|---|---|
| Mode-recall lower bound | Sample complexity / generalization-bound-like | (a) Synthetic 4-mode Gaussian MDP：扫 Δ ∈ {0.5,1,2,5}, N ∈ {100,500,2000,10k}；oracle vs learned mode；(b) Kitchen 上 mode separation 实测 + theorem 预测 | 4×4×3 seed = 48 synthetic + 5 Kitchen tasks × 3 seed | FEASIBLE | none |
| Per-mode > global W₂ 反例 | Construction | 解析 closed-form 2-mode + 5 numeric (toy) | trivial | FEASIBLE | none |
| ICNN-W₂ bias multi-mode | Estimator bound | Synthetic mode count m ∈ {2,4,8} × N ∈ {1k, 10k} | 6 × 3 seed = 18 | FEASIBLE | none |
| Empirical: rare-mode recall × return Pareto dominate | Empirical hypothesis | D4RL Kitchen + AntMaze: Diffusion-QL / global W₂-DiffPolicy / KL-BDPO / ours，oracle-mode + unsup ablation | 8 D4RL multi-modal tasks × 4 baselines × 3 seed = 96 runs | FEASIBLE — 双卡 6 周内可完成（去掉 Adroit + NeoRL-2 后） | none |
| Inference overhead < 5% | Computational complexity | Wall-clock test 5 batch sizes × 3 seed | trivial | FEASIBLE | none |
| Adroit + NeoRL-2 transfer (附录) | Empirical hypothesis (stretch) | 2 NeoRL-2 + 2 Adroit task × 3 seed | 12 runs | FEASIBLE WITH CAVEATS — 附录；若时间紧可推迟到 v2 | none |

无 NOT FEASIBLE 项。

## Evaluation Sketch

- **如何 validate**：synthetic 4-mode MDP（解析）→ Kitchen + AntMaze（rare-mode recall × return Pareto）→ 附录 Adroit + 1-2 NeoRL-2 task
- **关键指标**：(1) Rare-mode Recall@τ；(2) Average return；(3) Mode-mass Error；(4) Per-mode return spread
- **success looks like**：rare-mode recall × return Pareto 严格 dominate global W₂/KL；toy figure 显示 global W₂ → 0 但 mass 漂移；inference overhead < 5%
- **failure looks like**：Pareto 不 dominate 或 dominate 区域 < 30%；或 ICNN training 不稳

## Resource Estimate

- **Scale**: MEDIUM (5-7 person-weeks 到 v1 paper)
- **Compute**: MEDIUM (双 4090, 48GB；Stage 0 几小时；Stage 1+2 单 task 4-6h；8 multi-modal tasks × 4 baselines × 3 seed = 96 runs ≈ 5 天双卡)
- **Data**: 全 public (D4RL + 附录 NeoRL-2)
