# Round 2 Refinement

## Problem Anchor (verbatim)

[同 round 1 — 不重复]

## Anchor Check

- 仍攻 rare value-relevant mode preservation
- 修订后所有公式以 **action chunk a_{0:H-1}** 为 unit；c_ω, W₂, DSM, theorem 全部对齐
- 拒绝的 reviewer drift：
  - 不接受 "single-action policy + 定理改 action-mode recall"——该 alternative 让定理与 trajectory-mode 失联，弱化核心 claim
  - 不接受 Adroit/NeoRL-2 重新进主 claim

## Simplicity Check

- **dominant**: Mode-Conditional W₂ + finite-sample mode-recall 定理（保持）
- **components removed/changed**:
  - **D_TV(m_θ, m_β) marginal 项 → 删除，替换为 conditional mode-consistency loss** L_cons = E[-log c_ω(z|s,a_chunk)]——直接、易解释、可用 reviewer 一眼懂
  - **Hard c_ω > 0.5 → soft GMM responsibilities**：β(a|s,z) 用 weighted dataset, weight = c_ω(z|s, a_chunk) ∈ [0,1]
  - **W₂ 项的 m_β 加权 → balanced mode replay (uniform over z)**——matched importance weighting，避免 rare mode 训练信号弱矛盾
  - 加 **conditioning dropout 5%** (classifier-free style)：鲁棒性补强，不在 contribution list
- **smallest adequate route**: 仍只替换 Diffusion-QL loss 一项；新 trainable 不变

## Top-2 Issues Diagnosed

### Issue 1: action vs action-chunk unit 不一致（CRITICAL）
- **Reader experience**: 公式中频繁出现 `c_ω(z|s, a_{0:H})` 与 `a_chunk`，但 Diffusion-QL 是 single-action diffusion——读者无法编程实现，立刻信任崩塌
- **Hostile reviewer critique**: "Diffusion-QL backbone 输出 single action a；整本 paper 的 c_ω/W₂/DSM/theorem 都用 a_{0:H}——要么扩展 ε_θ 输出 H 步 chunk (整体改写)，要么把所有 chunk 改回单 action (定理换粒度)。请明确"
- **Structural impact**: 破坏 Step 2 (Formalization)；编程不可能，定理无意义

**决策**: 选 **action-chunk diffusion policy** (H=4)，理由：
- Multi-modal mode 的本质是 trajectory-level 模式（不同烹饪策略、不同导航路径）；single-action 粒度无法捕捉
- Diffusion Policy by Chi et al (RSS 2023, robot manipulation) 已证 chunk diffusion 可行；Mixed-Density Diffuser (P07) 等 NeurIPS 2025 工作也是 chunk 风格
- ε_θ 输出 chunk 仅需把 last layer reshape 到 (H, action_dim)；总参数增 < 5%

### Issue 2: rare-mode 训练信号弱矛盾核心 claim（IMPORTANT）
- **Reader experience**: 读到 "本方法保 rare mode" 然后看到 W₂ 项被 m_β(z|s) 加权——rare mode 由定义 m_β 小，因此训练梯度小；逻辑矛盾
- **Hostile reviewer critique**: "你说 rare mode preservation 是 contribution，但 loss 给 rare mode 更小权重 (m_β(z|s))；更糟，定理误差项 ε_W(z*) 在 rare mode 上正是被欠优化的——你的实验 oracle-mode 能跑出 90%+ recall 才怪"
- **Structural impact**: 破坏 Step 3 (Theory)——定理 vacuous 因为 rare-mode 上 ICNN error 大；Step 4 (Empirical) 实验做不出 Pareto dominate

**决策**: **balanced mode replay** + **conditional consistency loss**：
- 训练 batch 按 z 平衡采样：每个 z 同样多 sample；W₂ 项不再用 m_β 加权而是 uniform_z
- 但 inference 时 z ~ m_β(z|s) 保持，因此 deployment policy 仍按真实 mode 频率混合
- 加 conditional consistency `L_cons = E_{z~Uniform, a~π_θ(.|s,z)}[-log c_ω(z|s,a_chunk)]`——保 mode 一致性而不依赖 marginal TV

## Skeleton Gap Check

| Skeleton Step | Round 1 Coverage | Assessment | Round 2 Action |
|---|---|---|---|
| Step 1 Diagnosis | "Technical Gap" + 计划 toy 反例 | adequate | 不变 |
| Step 2 Formalization | z + FiLM + frozen c_ω | **superficial**（unit 不一致）| **重写** unit = action chunk H=4；c_ω/W₂/DSM 全 chunk |
| Step 3 Theory | finite-sample bound | **被弱训练信号削弱** | balanced replay 让 ε_W(z*) 真小；定理保持 |
| Step 4 Empirical | 96 runs FEASIBLE | **schedule 乐观** | core = 4 tasks × 4 baselines × 3 seeds = 48 runs；8 tasks 是 stretch |
| Step 5 Differentiation | 差异表 | adequate | 不变 |

## Changes Made

### 1. Unit 统一到 action chunk H=4（CRITICAL #1 fix）
- **Reviewer said**: c_ω 与 π_θ unit 不一致
- **Action**:
  - Policy 输出从 single action 改为 action chunk a_{0:H-1}, H=4
  - ε_θ last layer reshape 到 (H, action_dim)；总参数增 < 5%
  - c_ω(z | s, a_{0:H-1}): 输入 state + chunk；输出 K-dim mode 概率
  - W₂²_ψ_z 在 chunk 空间：a, a' ∈ R^{H × d}
  - DSM loss 在 chunk 上 (DDPM 标准做法)
  - Theorem 改为 chunk-mode recall（更强 — chunk 比 single-action 更能区分 mode）
- **Reasoning**: chunk 是 mode-meaningful unit；与 RSS 2023 / NeurIPS 2025 (Chi et al, P07) 一致；编程可行
- **Impact**: Method Specificity 6.8 → ≥ 9

### 2. Balanced mode replay 替代 m_β 加权（IMPORTANT #1 fix）
- **Reviewer said**: rare mode 训练信号弱
- **Action**:
  - Training: batch 内按 z 等量采样；W₂ 项 `(1/K) Σ_z W₂²_ψ_z(...)` 而非 `Σ_z m_β(z|s) W₂²_ψ_z`
  - Inference: 仍按真实分布 z ~ m_β(z|s) 采样
  - 这是 importance-weighting 视角：训练用 uniform z 让 estimator 对每个 mode 平均高质，部署时按 m_β 混合即得 unbiased policy
- **Reasoning**: 让 ε_W(z*) 在 rare mode 上真小，定理 vacuity 风险消除
- **Impact**: Contribution Quality 8.0 → ≥ 9；Theory consistency 重建

### 3. Conditional consistency loss 替代 marginal TV（Simplification #1）
- **Reviewer said**: marginal TV 不直接
- **Action**:
  - 删除 `γ · D_TV(m_θ(z|s), m_β(z|s))`
  - 加 `γ · E_{z~Uniform, a~π_θ(.|s,z)} [-log c_ω(z | s, a_{0:H-1})]`
  - 直接问 "条件 z 下 sampled chunk 是否仍在 mode z"
- **Reasoning**: 更直接、更易估计、更可读；reviewer Modernization Opp #1
- **Impact**: Method Specificity 与 Contribution Quality 双升

### 4. Soft GMM responsibilities + classifier-free dropout（Modernization）
- **Action**:
  - β(a_chunk | s, z) ≈ weighted dataset，weight = c_ω(z | s, a_chunk) (soft, ∈ [0,1])
  - Training: 5% 概率 z = null token, ε_θ 学 fallback to marginal behavior
- **Reasoning**: soft 比 hard 稳；classifier-free dropout 增鲁棒性
- **Impact**: Frontier Leverage 8.3 → ≥ 9

### 5. Schedule realism（IMPORTANT #2 fix）
- **Action**:
  - **Core schedule**: 4 multi-modal tasks (Kitchen-mixed, AntMaze-medium-play, AntMaze-large, Adroit-cloned) × 4 baselines (Diffusion-QL / global-W₂-DiffPolicy / BDPO / ours) × 3 seeds = 48 runs；约 7-8 天双卡（含失败重跑）
  - **Stretch**: 余下 4 tasks (Kitchen-partial/complete, AntMaze-medium-diverse, Adroit-expert) — 时间允许时跑
  - **Appendix**: 1-2 NeoRL-2 task × 3 seed
- **Reasoning**: 真实 schedule；reviewer 不会怀疑可行性
- **Impact**: Feasibility 6.8 → ≥ 8

## Revised Proposal

# Mode-Conditional W-DiffPolicy: Wasserstein-Regularized Action-Chunk Diffusion Policies with Per-Mode Mass Preservation for Multi-Modal Offline Reinforcement Learning

## Problem Anchor [verbatim, unchanged]

## Skeleton

- State A → State B 同前；Path: Diagnosis → Formalization (unit=chunk) → Theory → Empirical (4+4 tasks, 3 seeds) → Differentiation

## Technical Gap

[同 round 1，简化]

## Method Thesis

**Single sentence**: 我们证明 global Wasserstein 行为正则在 multi-modal offline RL 中可通过 mass 平移系统性丢失稀有 value-relevant mode；提出 **Mode-Conditional W-DiffPolicy**——以 action chunk (H=4) 为 unit、frozen contrastive trajectory encoder + GMM 定义 K=4..8 个 mode、per-mode conditional W₂ via ICNN-OT 加 balanced mode replay 与 conditional consistency loss、FiLM-based mode conditioning + classifier-free dropout——给出 finite-sample mode-recall 保证，inference overhead < 5%。

## Contribution Focus

- **Dominant**: Mode-Conditional W₂ + finite-sample mode-recall 定理 (含 ICNN-W₂ + c_ω error 显式 bound)
- **Optional supporting**: 无独立支持贡献。Section 4 用 rare-mode chunk recall + per-mode chunk-return + return-recall Pareto 三指标作 evaluation protocol（不在 Contributions list）
- **Non-contributions**: 不主张首个 W₂+diffusion；不主张 Q-cost OT；不引入 inference 加速；不做 online；不做完整 NeoRL-2；不重写 evaluation benchmark

## Proposed Method

### Complexity Budget

- **Frozen / reused**:
  - Diffusion-QL backbone (chunk variant): K-step DDPM denoiser ε_θ(a_k_chunk, k, s, z) via FiLM；output (H, action_dim)
  - IQL critic Q_φ(s, a_first)（对 chunk 第一个 action 评 Q）
  - **Frozen contrastive trajectory encoder f_ω** (~1M params，SimCLR-style)
  - **Frozen GMM** on f_ω embedding (K=4..8 by BIC) → m_β(z|s)
  - **Frozen mode classifier c_ω(z | s, a_{0:H-1}) = soft GMM-posterior of f_ω(s, a_{0:H-1})** — soft, not threshold
  - ICNN OT potential η_ψ
- **New trainable** (≤ 2):
  1. Per-mode ICNN potentials η_ψ_z (K 个 ICNN, shared first 2 layers, mode-conditional last layer; chunk-input)
  2. FiLM mode-conditioning head + null-token (classifier-free dropout) in ε_θ
- **Tempting additions intentionally NOT**:
  - Joint EM fine-tune of f_ω/GMM
  - LLM mode classifier
  - Q-cost OT
  - Online RL
  - Consistency distillation
  - Multi-critic ensemble
  - Marginal D_TV term

### Stage 0: Offline Mode Discovery (one-shot, ~ 3-6 hours)

- **0a**: Train f_ω on dataset trajectories with InfoNCE loss + augmentations (sub-trajectory masking, time-shift). Frozen.
- **0b**: Fit GMM(K = 4..8 by BIC) on f_ω(τ); compute m_β(z|s) via local averaging within ε-neighborhood of s. Frozen.
- **0c**: Define c_ω(z | s, a_{0:H-1}) = GMM-posterior(f_ω(s, a_{0:H-1})), **soft** (no threshold). Frozen, inference-only.
- **0d (sanity)**: t-SNE/UMAP visualize f_ω embedding; if silhouette < 0.2, switch to oracle protocol (dataset task-id) for that benchmark.

### Stage 1: Conditional Diffusion Policy with FiLM + Classifier-Free Dropout

- ε_θ(a_k_chunk, k, s, z) — FiLM(γ_z, β_z) per residual block
- 5% training-time z = null (classifier-free style); rollout always uses sampled z
- Stage 1 (~ 30%): warm up KL-BC + uniform z

### Stage 2: Mode-Conditional W₂ Training (~ 70%)

```
L_total(θ, ψ_{1:K}) = 
    L_RL:        - E_{s, z~Uniform, a_chunk~π_θ(.|s,z)} [Q_φ(s, a_chunk[0])]   ← balanced replay over z
    + L_modeW2:  + α · (1/K) Σ_{z=1..K} E_s [W₂²_ψ_z(π_θ(a_chunk|s,z), β(a_chunk|s,z))]   ← uniform_z
    + L_cons:    + γ · E_{s, z~Uniform, a_chunk~π_θ(.|s,z)} [- log c_ω(z | s, a_chunk)]   ← conditional mode consistency
    + L_DSM:     + β · E_{z~Uniform, a_β_chunk~D|z, t} [|| ε - ε_θ(a_β_chunk + σ_t·ε, t, s, z) ||²]
                  where β(a_chunk|s,z) ≈ weighted dataset, weight = c_ω(z | s, a_β_chunk) (soft)
    (5% probability replace z with null token in all four terms)
```

**关键设计点**:
- 训练 z ~ Uniform：每个 mode 在 batch 中等权出现 → rare mode 得充分训练信号
- 推理 z ~ m_β(z|s)：deployment 仍按真实 mode 分布混合 → 不偏离 dataset 行为
- Balanced replay 与 importance-weighted estimation 等价但实现简单
- L_cons 替代 marginal TV：直接问 "条件 z 下 sampled chunk 是否仍在 mode z"

### Per-mode ICNN-OT 实现

- 每个 z 一个 lightweight ICNN η_ψ_z（chunk-input，shared first 2 layers state encoder, last layer mode-specific）
- Dual formulation (Q-DOT eq. 6-9): W₂²_ψ_z = E[||a_chunk - T_z(a'_chunk)||²], T_z = ∇η_ψ_z (Brenier)
- Discriminator-free; per-batch sub-batch ≥ 32 per mode (用 balanced replay 自然保证)
- Sinkhorn warm-up only if validation W₂_ψ_z 相对 closed-form/Sinkhorn 偏差 > 50%

### Inference Path（< 5% wall-clock overhead）

```python
def act(s):
    z ~ Categorical(m_β(z|s))                    # 1 categorical sample, ~ 0.001ms
    a_chunk = sample_diffusion(eps_theta, s, z, K_steps)   # FiLM-conditioned, ~ 0.05ms extra/block
    return a_chunk[0]    # apply first action; chunk replanning every step (or every H steps for cheaper)
```

(可选：若希望更便宜，每 H 步 replan 一次，节约 H-1 步 diffusion 调用)

### Section 4 内的 evaluation protocol（非 contribution）

仅在 Section 4 描述；附录提供 oracle-mode (task-id) 与 unsupervised-mode (c_ω) 双 protocol。
- **Rare-mode chunk recall@τ**: dataset 中 mass < 20% 的 mode 在 N rollouts 中至少出现 1 次的比例
- **Mode-mass error**: ||m_θ_emp - m_β||₁
- **Per-mode chunk return** distribution

### Modern Primitive Usage

- Chunk diffusion: 与 RSS 2023 Diffusion Policy / NeurIPS 2025 Mixed-Density Diffuser 一致
- Frozen contrastive trajectory encoder: 现代 self-supervised
- FiLM + classifier-free dropout: conditional generation 标准
- ICNN-OT: discriminator-free conditional W₂

### Integration

- **Frozen**: ε_θ 主结构 + Q_φ critic + IQL Bellman + f_ω + GMM + c_ω
- **Trainable**: η_ψ_z (K 个) + ε_θ FiLM head + ε_θ fine-tune
- 总 trainable 参数 ~ 0.8M (新) + ε_θ 微调
- 总 frozen ~ 1M (encoder) + GMM lookups

### Failure Modes

| Failure | 检测 | Mitigation |
|---|---|---|
| GMM 不分离 | silhouette < 0.2 | 切 oracle (task-id)；honest report |
| ICNN unstable | val W₂_ψ_z vs Sinkhorn > 50% bias | Sinkhorn warm-up; reduce ICNN depth |
| Rare mode mass < 2% | exp(H[m_β]) | 标 "low-multimodality regime"；不主张优势 |
| Return 退 | per-mode return | 减 α；backup KL-BC |
| FiLM 失败 | conditional vs marginal action 距离 → 0 | 增 β 或 α；查 dropout 是否过高 |
| Chunk DSM unstable | DSM loss 跳跃 | 减 H (4→2)；增 K_steps |

### Novelty and Elegance Argument

| Work | Mechanism | Delta |
|---|---|---|
| Q-DOT (RLC 2025) | ICNN W₂ + IQL global | 本: chunk diffusion + per-mode + balanced replay |
| OTPR (Feb 2025) | Q-cost OT + diffusion + online | 本: 不用 Q-cost；offline；per-mode |
| BDPO (ICML 2025) | reverse-kernel KL + diffusion | 本: per-mode W₂ + FiLM + balanced replay |
| Diffusion-QL (ICLR 2023) | KL-BC + diffusion (single action) | 本: chunk + per-mode W₂ + FiLM |
| Diffusion Policy (RSS 2023, Chi et al) | chunk diffusion + BC | 本: + per-mode W₂ behavior reg + finite-sample theorem |
| LOM (ICLR 2025) | GMM + 选单 mode | 本: GMM + 保多 mode + 形式化定理 |
| Latent Diffusion ORL (ICLR 2024) | latent skill | 本: chunk action 不压缩；显式 mode preservation |
| P19 (Oct 2025) | LM RL KL collapse | 本: control-MDP 实证版 (Discussion only)；W₂ 修复 |
| SWFP (Oct 2025) | flow + W₂ JKO + online | 本: chunk diffusion + offline + per-mode |

**Why elegant**:
- 仅替换 Diffusion-QL chunk-variant loss 中一项 + 加 1 个 frozen offline stage
- Inference overhead < 5%
- 新 trainable < 1M 参数
- 定理与实现 1:1 对应

## Theoretical Grounding

### T1 Formalizability

| Component | Formal object | Draft |
|---|---|---|
| f_ω + GMM | InfoNCE + EM 收敛标准 | 不主张新理论 |
| c_ω | soft posterior error on holdout | ε_clf measurable on held-out chunks |
| Per-mode ICNN-W₂ on chunks | Brenier OT estimator error | `|Ŵ₂_ψ_z - W₂_true_z| ≤ ε_W(N_z, d·H, depth)` |
| Conditional consistency | NLL bound | E[-log c_ω(z|s, a_chunk)] ≤ ε_cons |
| **Mode-recall theorem (主)** | finite-sample lower bound | **Theorem (informal)**: 设 (a) chunk-space 中每对 mode pairwise W₂ ≥ Δ；(b) ε_W(z) ≤ ε_W(N_z, dH, depth)（**balanced replay 保 N_z = N/K**, 因此 ε_W 对所有 z 一致小）；(c) c_ω 错误率 ≤ ε_clf；(d) ε_cons ≤ ε_c；则对 m_β(z*) ≥ ρ_min 的 rare mode z*：`Recall_chunk_mode(πθ, z*) ≥ 1 - O(ε_W / Δ²) - O(ε_clf / ρ_min) - O(ε_c)` 高概率成立。**关键**: balanced replay 让 ε_W 不依赖 m_β(z*)，rare mode 保 recall。 |
| DSM ELBO on chunks | 标准 | 非新颖 |

### T2 Assumptions

| Claim | Assumption | Class | Sanity check |
|---|---|---|---|
| Mode-recall theorem | chunk pairwise W₂ ≥ Δ | RESTRICTIVE — 多模态任务通常成立 | Kitchen 上量化 |
| ICNN-W₂ on chunks | i.i.d. chunks; bounded VC | STANDARD | IDEA-07 风格压力测试 |
| **Balanced replay = uniform N_z** | training procedure design | DESIGN CHOICE — 由 batch sampler 保证 | unit test |
| c_ω err measurable | held-out chunks | STANDARD | per-task held-out |
| GMM K | dataset is K-mixture | RESTRICTIVE — BIC + oracle 双 verify | Kitchen task-id |
| Q err bounded | offline RL 标准 | STANDARD | report Q residual |

### T3 Theory-Experiment Alignment

| Claim | Type | Validation Protocol | Scale | Feasibility |
|---|---|---|---|---|
| Mode-recall lower bound | Sample complexity | Synthetic 4-mode chunk-MDP: Δ ∈ {0.5,1,2,5} × N ∈ {100,500,2k,10k} × oracle/learned mode | 4×4×3 = 48 runs | FEASIBLE |
| Per-mode > global counter-example | Construction | closed-form 2-mode + 5 numeric chunk instances | trivial | FEASIBLE |
| ICNN bias on chunks (multi-mode) | Estimator bound | mode count m ∈ {2,4,8} × N ∈ {1k, 10k} | 6 × 3 = 18 | FEASIBLE |
| **Empirical: rare-mode recall × return Pareto dominate (CORE)** | Empirical hypothesis | 4 multi-modal D4RL tasks × 4 baselines × 3 seeds | **48 runs** (核心) | FEASIBLE 双卡 7-8 天 |
| **Stretch**: 余下 4 D4RL tasks | Empirical hypothesis | + 48 runs | 48 stretch | FEASIBLE WITH CAVEATS |
| **Appendix**: NeoRL-2 1-2 task | Empirical hypothesis | × 3 seed | 6 runs | FEASIBLE WITH CAVEATS |
| Inference overhead < 5% | Computational complexity | wall-clock 5 batch × 3 seed | trivial | FEASIBLE |

无 NOT FEASIBLE。Core schedule 现实：48 runs 含失败重跑 + ICNN tuning + variance ≈ 7-8 天双卡。

## Evaluation Sketch

- **如何 validate**: synthetic 4-mode chunk-MDP → Kitchen + AntMaze (core, 4 tasks 4 baselines 3 seeds) → stretch (4 more tasks) → appendix Adroit + NeoRL-2
- **关键指标**: rare-mode chunk recall × return Pareto；mode-mass error；per-mode return spread；inference wall-clock
- **success**: Pareto strict dominate global-W₂/KL on 3+ of 4 core tasks；toy figure 显示 global W₂ → 0 但 mass 漂移；inference < 5%
- **failure**: Pareto 不 dominate 或 dominate 区域 < 30%；ICNN 训练不稳

## Resource Estimate

- **Scale**: MEDIUM (5-7 weeks)
- **Compute**: MEDIUM (双 4090, 48GB; Stage 0 ~ 6h; Stage 1+2 single task 4-6h; core 48 runs ≈ 7-8 天双卡，含失败重跑)
- **Data**: 全 public
