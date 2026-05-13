# Mode-Conditional W-DiffPolicy: Wasserstein Diffusion Policies with Per-Mode Mass Preservation for Multi-Modal Offline Reinforcement Learning

> **Final Proposal** — IDEA-01 经 3 轮 review-revise + Phase 5.5 expansion 后版本
> **Score Trajectory**: Round 1 = 7.45 → Round 2 = 7.71 → Round 3 = 7.93 (REVISE)
> **Verdict**: REVISE (MAX_ROUNDS=3 触达；自动收敛；用 Round 2 refinement 作 base + Round 3 surgical fixes 在 Phase 5.5 整合)
> **目标会议**: NeurIPS 2026
> **硬件预算**: 2× RTX 4090 (48GB)
> **时间预算**: 5-7 person-weeks 到 v1 paper

---

## Novelty Positioning

> **Global Wasserstein 行为正则可以做到整体距离很小但稀有 behavior mode 消失——因为 transport mass 可以在 mode 间漂移。Mode-Conditional W-DiffPolicy 通过 (a) 一次性离线固定 behavior modes、(b) 把 diffusion policy 条件在固定 modes 上、(c) 对每个 valid mode 单独正则，修复这一 failure。新颖性不是 "K 个 ICNN" 也不是 "W₂ + diffusion"，而是 global W₂ 的 failure mode 与对应的 per-mode recall 形式化保证。**

## Problem Anchor

- **Bottom-line**: 离线 RL 中 behavior policy 经常是真实多模态分布（Kitchen 多种烹饪策略、AntMaze 多种导航路径），但当前 diffusion policy 的 KL/score-BC 行为正则在多模态上系统失效——mode 漂移、稀有 high-value mode 被吞——导致 SOTA 在 multi-modal benchmark 上离 oracle 还有显著差距。
- **Must-solve**: multi-modal 任务上 rare value-relevant mode 的保留——每个有价值 mode 都被独立保住、其 mass 不被平均化或漂移到主导 mode。
- **Non-goals**: 不解决 inference 加速；不重写 Q-learning 框架；不做 online fine-tuning；不主张 "首次 Q-cost OT"（OTPR 已占）。
- **Constraints**: 2× RTX 4090；6-8 周 NeurIPS 2026；D4RL Kitchen/AntMaze (核心) + Adroit/NeoRL-2 (附录)。
- **Success**: chunk-mode recall × return Pareto frontier 严格 dominate global-W₂/KL on 3/4 core tasks；finite-sample theorem 含 N_eff(z) + ε_W + ε_clf + ε_c；toy 反例 figure；inference overhead < 5%。

## Skeleton (State A → B Path)

- **State A**: reviewer 相信 diffusion policy 已能自然保多模态，行为正则只是 distance 选择问题（KL → W₂）
- **State B**: reviewer 接受 "global W₂ 也会通过 mass 漂移吞稀有 mode"，必须 per-mode 形式约束 mass，且本方法有 finite-sample 保证 + 实证证据
- **Path**:
  1. **Diagnosis**: 全局 W₂ 也吞 rare value-relevant mode（toy 反例 figure）
  2. **Formalization**: chunk-unit + frozen mode classifier + per-mode W₂ + 条件 consistency loss
  3. **Theory**: finite-sample chunk-mode recall bound 含 N_eff(z) 与 ε_W/ε_clf/ε_c 显式 error term
  4. **Empirical**: synthetic 4-mode → D4RL multi-modal core (4 tasks × 4 baselines × 3 seeds) → stretch + appendix
  5. **Differentiation**: vs OTPR / Q-DOT / BDPO / P19 / LOM / Diffusion-QL / SWFP / Latent Diffusion ORL 差异表

## Technical Gap

### 当前 Diffusion offline RL 的失效

- **Diffusion-QL (P01)**：score-matching BC 近似 KL；multi-modal 上稀有 mode 的 score 梯度被淹
- **BDPO (P04, ICML 2025)**：把 KL 严格化为反向 transition kernel KL 累加；reverse-path 累积 KL ≤ ε 与 terminal 多模态 mass error 解耦（CRITIQUE-02）
- **Q-DOT (P14, RLC 2025)**：ICNN-W₂ 替 KL 但仅 IQL；其全局 W₂ 仍可通过 mass 平移坍多模态
- **OTPR (2502.12631, Feb 2025)**：Q-cost OT + diffusion，但 **online RL fine-tuning** 而非 offline；Q-cost 不解决 rare-mode preservation
- **LOM (P20, ICLR 2025)**：直接放弃多模态选最佳单 mode
- **P19 (Oct 2025)**：在 LM RL 上证 KL collapse；未推到 control，无 W₂ 替代定理

### 最小充分干预

**Mode-Conditional W₂ 行为正则**：以 chunk 为 unit、frozen mode classifier 锚定 mode、per-mode conditional W₂ via ICNN-OT + balanced valid-(s,z) sampler + conditional consistency loss + FiLM-based mode conditioning + classifier-free dropout。

仅替换 Diffusion-QL chunk-variant loss 中一项；新 trainable 总参数 < 1M；inference overhead < 5%。

## Method

### Stage 0 — Offline Mode Discovery (one-shot, ~3-6 hours)

#### 0a. Contrastive Trajectory Encoder f_ω

**Architecture** (frozen after Stage 0):
- 2-layer Transformer encoder over chunk-tokens
- Input: state s ∈ R^{d_s}, action chunk a_chunk ∈ R^{H × d_a} (H=4)
- Tokenization: `[CLS, Linear_s(s), Linear_a(a_0)+PE_0, ..., Linear_a(a_{H-1})+PE_{H-1}]`
- d_model=128, n_heads=4, FFN=256
- Output: h ∈ R^{d_e=64}, L2-normalized

**Loss**: $\mathcal{L}_f = \mathcal{L}_{NCE} + \lambda_{rec} \mathcal{L}_{rec}$ where $\mathcal{L}_{NCE}$ is InfoNCE with τ=0.1.

**Augmentations**: sub-chunk masking (mask_prob=0.15), action jitter (σ=0.01), time-shift (±2 step).

**Hyperparams**: H=4, d_e=64, batch=512, lr=3e-4, steps=20k. (Range/default/justification 表见 round-3-expanded.md)

#### 0b. GMM + state-conditional m_β(z|s)

For each chunk i: $h_i = f_\omega(s_i, a_{i:i+H-1})$.

Per-task GMM: $p(h) = \sum_{z=1}^K \pi_z \mathcal{N}(h; \mu_z, \Sigma_z)$. K ∈ {4..8} by BIC. Diagonal Σ.

Soft responsibilities: $r_i(z) = \pi_z \mathcal{N}(h_i; \mu_z, \Sigma_z) / \sum_{z'} \pi_{z'} \mathcal{N}(h_i; \mu_{z'}, \Sigma_{z'})$.

State-conditional prior (kernel-smoothed kNN, k=200, σ_s = median kNN dist):
$$m_\beta(z|s) = \frac{\sum_{i \in \mathcal{N}_k(s)} K_\sigma(s, s_i) r_i(z)}{\sum_{i \in \mathcal{N}_k(s)} K_\sigma(s, s_i)}$$

Low-density shrinkage: $\tilde m_\beta = \lambda(s) m_\beta + (1-\lambda(s)) \bar m_\beta$, $\lambda(s) = n_{eff}(s) / (n_{eff}(s) + 50)$.

**Frozen.**

#### 0c. Frozen Mode Classifier

$$c_\omega(z | s, a_{0:H-1}) = \text{soft posterior of GMM}(f_\omega(s, a_{0:H-1}))$$

**Frozen, inference-only.**

#### 0d. Sanity

- Silhouette < 0.2 警告 → 切 dataset task-id oracle protocol（Kitchen 有 task labels）
- t-SNE/UMAP visualize 作 paper figure

### Stage 1 — Conditional Chunk Diffusion Warm-up (~30%)

ε_θ(a_k_chunk, k, s, z): chunk variant of Diffusion-QL with FiLM(γ_z, β_z) per residual block. 5% prob z = null token (classifier-free).

Loss = chunk DSM only:
$$\mathcal{L}_{S1} = \beta \cdot \mathbb{E}_{(s,a)\sim D, z\sim r_i(z), t} \left[ \| \epsilon - \epsilon_\theta(a_{chunk} + \sigma_t \epsilon, t, s, z) \|^2 \right]$$

**Switch to Stage 2** when: $t \geq \min(0.3 T_{total}, t : \text{ValDSM}_{t-5:t} \text{ improves} < 2\%)$, min warmup 15k.

### Stage 2 — Mode-Conditional W₂ Training (~70%)

```
L_total(θ, ψ_{1..K}) = 
    -λ_Q · E_{s, z~U_valid, a_chunk~π_θ(.|s,z)} [Q_φ(s, a_chunk[0])]               ← balanced replay over valid modes
    + α · L_W2_actor                                                                  ← per-mode W₂ surrogate
    + γ · E_{s, z~U_valid, a_chunk~π_θ(.|s,z)} [-log c_ω(z | s, a_chunk)]            ← conditional consistency
    + β · L_DSM_chunk (z-conditional, z~r_i(z) on dataset)                           ← chunk DSM
    (5% prob z=null in all 4 terms)
```

with:
- $\mathcal{L}_{W2}^\pi = \mathbb{E}_{x_\theta \sim \pi_\theta(\cdot|s,z)} [\| x_\theta - \text{sg}(\nabla_x u_{\psi,z}(s, x_\theta)) \|^2]$ (Brenier-map identity)
- IQL critic Q_φ trained simultaneously, expectile τ ∈ {0.7 Kitchen, 0.9 AntMaze}

#### Per-mode ICNN η_{ψ,z}

Architecture (non-negative weights via softplus reparameterization):
$$h_0 = \text{softplus}(W_x^{(0)} x + W_s^{(0)} e_s + b_0)$$
$$h_{\ell+1} = \text{softplus}(W_{h,+}^{(\ell)} h_\ell + W_x^{(\ell)} x + W_s^{(\ell)} e_s + b_\ell)$$
$$u_{\psi,z}(s,x) = w_{h,+}^{(z)\top} h_L + w_x^{(z)\top} x + w_s^{(z)\top} e_s + b^{(z)} + \frac{\epsilon_{sc}}{2} \|x\|^2$$

Strong-convexity ε_sc = 1e-4. Depth=3, width=256. lr=1e-4. n_icnn_steps_per_policy=2. batch_per_mode=32.

Semi-dual training:
$$\mathcal{L}_{ICNN}(\psi; z) = \mathbb{E}_{x_\theta \sim P_{\theta,z}}[u_{\psi,z}(s, x_\theta)] + \mathbb{E}_{x_\beta \sim P_{\beta,z}}[u_{\psi,z}^*(s, x_\beta)]$$

with $u^*(s, y) = \max_{x \in [-1,1]^D} \langle x, y \rangle - u(s, x)$ via inner gradient ascent (10 steps, lr=5e-2).

#### Balanced Valid (s, z) Sampler

```python
def build_mode_samplers(D, R, n_min=128):
    samplers = {}
    for z in range(K):
        weights = R[:, z]
        n_eff = weights.sum()**2 / (weights**2).sum()
        if n_eff >= n_min:
            samplers[z] = AliasSampler(normalize(weights))
    return samplers   # only valid modes

def balanced_mode_batch(samplers, batch_size=512):
    valid_modes = list(samplers.keys())
    b_per_z = batch_size // len(valid_modes)
    batch = []
    for z in valid_modes:
        idx = samplers[z].sample(b_per_z)
        s, a_chunk = D.states[idx], D.chunks[idx]
        batch.append((z, s, a_chunk, R[idx, z]))
    return concat(batch)
```

**关键**: 不在任意 (s, z) 对上做 W₂；只对 valid 模式（N_eff ≥ n_min）按 soft weight 采样。

### Inference: Receding-Horizon Only (overhead < 5%)

```python
def act(s_t):
    mode_probs = m_beta(s_t)                      # [K]
    z_t = categorical_sample(mode_probs)          # ~ 0.001ms

    a_chunk = ddpm_sample_chunk(
        epsilon_theta, state=s_t, mode=z_t,
        H=4, n_steps=K_eval=10, film=True
    )                                              # [H, d_a]; ~0.05ms extra/block

    return a_chunk[0]                             # receding horizon ONLY
```

**Chunk 仅作 mode-conditioning + DSM 稳定化对象，不作 open-loop plan**。

## Theoretical Grounding

### Main Theorem (with N_eff)

**Setup**:
- chunk space $\mathcal{X} = [-1, 1]^{H d_a}$
- frozen $c_\omega(z|s,x)$, $r_i(z) = c_\omega(z | s_i, x_i)$
- effective sample size: $N_{eff}(z) = (\sum_i r_i(z))^2 / \sum_i r_i(z)^2$

**Definition (chunk-mode recall)**: $\text{Recall}_z(\pi_\theta) = \Pr_{s\sim D_z, x\sim \pi_\theta(\cdot|s,z)}[c_\omega(z|s,x) \geq \tau_c]$

**Theorem (informal)**. Assume:
1. **Mode separation Δ**: chunks in mode z 与 z' 在 chunk-space 上 W₂/classifier-margin ≥ Δ
2. **ICNN OT estimator error**:
   $$|\widehat W_{2,z}^2 - W_{2,z}^2| \leq \epsilon_{stat}(z, \delta) + \epsilon_{opt}(z) + \epsilon_{conj}, \quad \epsilon_{stat}(z, \delta) = C R^2 \sqrt{\frac{\text{Pdim}(\mathcal{U}_\psi) + \log(K/\delta)}{N_{eff}(z)}}$$
   with prob ≥ 1-δ, $R = O(\sqrt{H d_a})$
3. **Classifier err on holdout**: $\epsilon_{clf}(z) = \Pr_{(s,x)\sim\beta_z}[c_\omega(z|s,x) < \tau_c]$
4. **Consistency NLL bound**: $\mathcal{L}_{cons}(z) \leq \epsilon_c(z)$

**Conclusion**: For modes with $m_\beta(z) \geq \rho_{min}$ and $N_{eff}(z) \geq n_{min}$:
$$\text{Recall}_z(\pi_\theta) \geq 1 - O\left(\frac{\epsilon_{stat}(z,\delta) + \epsilon_{opt}(z) + \epsilon_{conj}}{\Delta^2}\right) - O\left(\frac{\epsilon_{clf}(z)}{\rho_{min}}\right) - O\left(\frac{\epsilon_c(z)}{\log 2}\right)$$
with prob ≥ 1-δ.

**关键 clarification**: Balanced replay equalizes optimization budget but **NOT** statistical sample size. 统计误差仍取决于 N_eff(z)。ρ_min 与 n_min 显式条件：满足不了的 rare mode 在论文中诚实标注为 "low-multimodality regime"，不主张优势。

### Theory-Experiment Alignment Matrix

| Theoretical Claim | Type | Validation Protocol | Scale | Feasibility |
|---|---|---|---|---|
| Mode-recall lower bound (with N_eff) | Sample complexity | Synthetic 4-mode chunk-MDP: Δ ∈ {0.5,1,2,5} × N_eff ∈ {64,256,1024,4096} × oracle/learned mode | 4×4×3 = 48 | FEASIBLE |
| Per-mode > global counter-example | Construction | closed-form 2-mode + 5 numeric | trivial | FEASIBLE |
| ICNN bias on chunks (multi-mode) | Estimator bound | mode count m ∈ {2,4,8} × N_eff ∈ {1k, 10k} | 6 × 3 = 18 | FEASIBLE |
| **NEW: ε_stat scales with N_eff(z)** | Estimator scaling | log-log plot of ICNN-W₂ error vs N_eff (replay-controlled) | reuse 48 synthetic | FEASIBLE |
| **NEW: consistency-to-recall via log 2** | Information-theoretic | Synthetic mixture; classifier err on held-out vs ε_c | 6 × 3 | FEASIBLE |
| **NEW: chunk-mode scope clarification** | Scope | explicit narrative in paper | trivial | FEASIBLE |
| **CORE: Pareto dominate D4RL** | Empirical hypothesis | 4 tasks (Kitchen-mixed, AntMaze-medium-play, AntMaze-large-play, Adroit-cloned) × 4 baselines × 3 seeds | 48 runs ≈ 7-8 days | FEASIBLE |
| Stretch: 4 more tasks | Empirical | + Kitchen-partial/complete + AntMaze-medium-diverse + Adroit-expert | 48 stretch | FEASIBLE WITH CAVEATS |
| Appendix NeoRL-2 | Empirical | RocketRecovery + Pipeline × 3 seed | 6 runs | FEASIBLE WITH CAVEATS |
| Inference overhead < 5% | Comp. complexity | 5 batch × 3 seed wall-clock | trivial | FEASIBLE |

无 NOT FEASIBLE。3 个新理论 claim 在现有 budget 内 absorb。

## Failure Modes

| Failure | 检测 | Mitigation |
|---|---|---|
| GMM 不分离 | silhouette < 0.2 | task-id oracle protocol；honest report |
| ICNN unstable | val W₂_ψ_z vs Sinkhorn 偏差 > 50% | Sinkhorn warm-up；reduce ICNN depth |
| Rare mode mass < ρ_min | exp(H[m_β]) | "low-multimodality regime" disclaimer |
| Return 退 | per-mode return | 减 α；backup KL-BC |
| FiLM 失效 | conditional vs marginal action 距离 → 0 | 增 β 或 α；查 dropout |
| Chunk DSM 跳跃 | DSM loss 大幅波动 | 减 H (4→2)；增 K_steps |
| N_eff(z) 过小 | < n_min for z | 报告该 mode unverified；不主张优势 |

## Differentiation Table (Novelty Argument)

| Work | Mechanism | Delta vs Mode-Conditional W-DiffPolicy |
|---|---|---|
| Q-DOT (RLC 2025) | ICNN W₂ + IQL global | 本: chunk diffusion + per-mode + N_eff theorem |
| OTPR (Feb 2025) | Q-cost OT + diffusion + online | 本: 不用 Q-cost；offline；per-mode |
| BDPO (ICML 2025) | reverse-kernel KL + diffusion | 本: per-mode W₂ + FiLM + balanced replay |
| Diffusion-QL (ICLR 2023) | KL-BC + diffusion (single action) | 本: chunk + per-mode W₂ |
| Diffusion Policy (Chi RSS 2023) | chunk diffusion + BC | 本: + per-mode W₂ + finite-sample theorem |
| LOM (ICLR 2025) | GMM + 选单 mode | 本: GMM + 保多 mode + 形式化定理 |
| Latent Diffusion ORL (ICLR 2024) | latent skill 压缩 | 本: chunk action 不压缩；显式 mode preservation |
| P19 (Oct 2025) | LM RL KL collapse | 本: control 实证版（Discussion）；W₂ 修复 |
| SWFP (Oct 2025) | flow + W₂ JKO + online | 本: diffusion + offline + per-mode |

## Resource Estimate

- **Scale**: MEDIUM (5-7 person-weeks 到 v1)
- **Compute**: MEDIUM (双 4090, 48GB; Stage 0 ~ 6h; Stage 1+2 single task 4-6h; core 48 runs ≈ 7-8 天双卡 含失败重跑)
- **Data**: 全 public (D4RL public; NeoRL-2 public)

## 90-Day Timeline (粗规划)

| 周 | 任务 |
|---|---|
| W1 | Stage 0 复现 + Diffusion-QL chunk-variant baseline + ICNN η_ψ_z 实现 |
| W2 | synthetic 4-mode chunk-MDP 实验 + counter-example figure |
| W3 | core D4RL: Kitchen-mixed + AntMaze-medium-play (4 baselines, 3 seeds each) |
| W4 | core D4RL: AntMaze-large-play + Adroit-cloned (完成 48 runs) |
| W5 | theorem 形式化 + ICNN bias 压力测试 + N_eff 实验 |
| W6 | 写作 + ablation (α/β/γ sweep, oracle/unsupervised, chunk H sweep) |
| W7 | stretch 4 task 或 NeoRL-2 appendix + 改稿 |
| W8 | 投 NeurIPS 2026 |
