# Round 3 Expanded — Final Implementation-Precision Proposal

> 基础: round-2-refinement.md（最高分版本，Round 3 review 评分 7.93/10）
> 整合 Round 3 surgical fixes (N_eff theorem, receding-horizon, valid (s,z) sampler, novelty positioning) + 全 [EXPAND] 段填充至 implementation precision

---

# Mode-Conditional W-DiffPolicy: Wasserstein Diffusion Policies with Per-Mode Mass Preservation for Multi-Modal Offline Reinforcement Learning

## Novelty Positioning（论文 Contribution narrative）

> Global Wasserstein 行为正则可以做到整体距离很小但稀有 behavior mode 消失——因为 transport mass 可以在 mode 间漂移。Mode-Conditional W-DiffPolicy 通过 (a) 一次性离线固定 behavior modes、(b) 把 diffusion policy 条件在固定 modes 上、(c) 对每个 valid mode 单独正则，修复这一 failure。**新颖性不是 "K 个 ICNN" 也不是 "W₂ + diffusion"，而是 global W₂ 的 failure mode 与对应的 per-mode recall 形式化保证。**

## Problem Anchor [verbatim]

- **Bottom-line**: 离线 RL 中 behavior policy 多模态，diffusion policy 的 KL/score-BC 行为正则在多模态上系统失效（mode 漂移、稀有 high-value mode 被吞）
- **Must-solve**: multi-modal 任务上 rare value-relevant mode 的保留
- **Non-goals**: 不解决 inference 加速；不重写 Q-learning；不做 online；不主张首个 Q-cost OT
- **Constraints**: 2× RTX 4090；6-8 周 NeurIPS 2026；D4RL Kitchen/AntMaze (核心) + Adroit/NeoRL-2 (附录)
- **Success**: chunk-mode recall × return Pareto dominate global-W₂/KL on 3/4 core tasks；finite-sample theorem 含 N_eff(z) + ε_W + ε_clf + ε_c；toy 反例 figure；inference overhead < 5%

## Skeleton [unchanged]

- State A → State B；Path: Diagnosis → Formalization (chunk unit + receding-horizon) → Theory (N_eff bound) → Empirical (4 core + 4 stretch tasks, 3 seeds) → Differentiation

## Method

### Stage 0: Offline Mode Discovery (one-shot, ~3-6 hours)

#### 0a. Contrastive Trajectory Encoder f_ω

**Architecture**: 2-layer Transformer encoder over chunk-tokens
- Input: `s ∈ R^{d_s}` (state), `a_chunk ∈ R^{H × d_a}` (action chunk, H=4)
- Tokenization: `[CLS, Linear_s(s), Linear_a(a_0) + PE_0, ..., Linear_a(a_{H-1}) + PE_{H-1}]`
- d_model=128, n_heads=4, FFN width=256
- Output: `h = LayerNorm(out_proj(CLS_token))` ∈ R^{d_e=64}, L2-normalized

**Loss**: 
$$\mathcal{L}_f = \mathcal{L}_{NCE} + \lambda_{rec} \mathcal{L}_{rec}$$

with InfoNCE:
$$\mathcal{L}_{NCE} = -\frac{1}{B}\sum_i \log \frac{\exp(sim(h_i^{(1)}, h_i^{(2)})/\tau_{NCE})}{\sum_j \exp(sim(h_i^{(1)}, h_j^{(2)})/\tau_{NCE})}$$

**Augmentations**: sub-trajectory masking (mask_prob=0.15), action jitter (jitter_std=0.01), time-shift (±2 step)

**Hyperparameters** (range / default / why):
- H: 2-8 / 4 / mode-meaningful but not open-loop risky
- d_e: 32-128 / 64 / GMM separation without overfitting
- τ_NCE: 0.05-0.2 / 0.1 / standard
- mask_prob: 0.05-0.3 / 0.15 / chunk-level invariance
- λ_rec: 0-1 / 0.1 / NCE dominant
- batch: 256-1024 / 512 / contrastive negatives
- lr: 1e-4 — 1e-3 / 3e-4 / stable Transformer
- steps: 10k-50k / 20k / hours

**Frozen after Stage 0.**

#### 0b. GMM + state-conditional m_β(z|s)

For each chunk i:
$$h_i = f_\omega(s_i, a_{i:i+H-1})$$

Per-task GMM:
$$p(h) = \sum_{z=1}^K \pi_z \mathcal{N}(h; \mu_z, \Sigma_z)$$

K selected by BIC over {4..8}. Diagonal covariance (full unstable for rare modes).

Soft responsibilities:
$$r_i(z) = \frac{\pi_z \mathcal{N}(h_i; \mu_z, \Sigma_z)}{\sum_{z'} \pi_{z'} \mathcal{N}(h_i; \mu_{z'}, \Sigma_{z'})}$$

State-conditional prior via kernel-smoothed kNN (k=200, σ_s = median kNN dist):
$$m_\beta(z|s) = \frac{\sum_{i \in \mathcal{N}_k(s)} K_\sigma(s, s_i) r_i(z)}{\sum_{i \in \mathcal{N}_k(s)} K_\sigma(s, s_i)}$$

Low-density shrinkage:
$$\tilde m_\beta(z|s) = \lambda(s) m_\beta(z|s) + (1-\lambda(s)) \bar m_\beta(z), \quad \lambda(s) = \frac{n_{eff}(s)}{n_{eff}(s) + k_0}$$

with $k_0 = 50$.

**Frozen.**

#### 0c. Mode classifier c_ω(z | s, a_chunk)

$$c_\omega(z | s, a_{0:H-1}) = \text{soft posterior of GMM}(f_\omega(s, a_{0:H-1}))$$

**Frozen, inference-only.**

#### 0d. Sanity check
- Silhouette < 0.2 警告 → 切 task-id oracle protocol
- t-SNE/UMAP visualize for paper figure

### Stage 1: Conditional Chunk Diffusion Warm-up (前 30%)

ε_θ(a_k_chunk, k, s, z) — chunk variant of Diffusion-QL with FiLM:
- Backbone: U-Net 1D (chunk-axis as sequence)
- FiLM(γ_z, β_z) per residual block (γ_z, β_z ∈ R^d_h)
- Output: noise prediction in shape (H, d_a)
- 5% prob z → null token (classifier-free dropout)

Stage 1 loss = chunk DSM only:
$$\mathcal{L}_{S1} = \beta \cdot \mathbb{E}_{(s,a_chunk) \sim D, z \sim r_i(z), t} \left[ \| \epsilon - \epsilon_\theta(a_{chunk} + \sigma_t \epsilon, t, s, z) \|^2 \right]$$

Switch criterion to Stage 2:
$$t_{switch} = \min(0.3 T_{total}, t : \text{ValDSM}_{t-5:t} \text{ improves} < 2\%)$$
with min warmup 15k updates.

### Stage 2: Mode-Conditional W₂ Training (后 70%)

```
L_total(θ, ψ_{1..K}) = 
    -λ_Q · E_{s, z~U_valid, a_chunk~π_θ(.|s,z)}[Q_φ(s, a_chunk[0])]               ← balanced replay over valid modes
    + α · L_W2_actor                                                                 ← per-mode W₂ surrogate
    + γ · E_{s, z~U_valid, a_chunk~π_θ(.|s,z)}[-log c_ω(z | s, a_chunk)]            ← conditional consistency
    + β · L_DSM_chunk (z-conditional, z~r_i(z) on dataset)                          ← chunk DSM
    (5% prob z=null in all 4 terms)
```

with:
- L_W2_actor = E_{x_θ ~ π_θ(.|s,z)} [|| x_θ - sg(∇_x u_{ψ,z}(s, x_θ)) ||²]  (Brenier-map identity)
- IQL critic Q_φ trained simultaneously with expectile τ (0.7 Kitchen / 0.9 AntMaze)

#### Per-mode ICNN η_{ψ,z}

**Architecture**: ICNN with non-negative residual weights (softplus reparameterization)
- Input: (s, x ∈ R^{H d_a}, z one-hot)
- Hidden:
  $$h_0 = \text{softplus}(W_x^{(0)} x + W_s^{(0)} e_s + b_0)$$
  $$h_{\ell+1} = \text{softplus}(W_{h,+}^{(\ell)} h_\ell + W_x^{(\ell)} x + W_s^{(\ell)} e_s + b_\ell)$$
- Output:
  $$u_{\psi,z}(s, x) = w_{h,+}^{(z)\top} h_L + w_x^{(z)\top} x + w_s^{(z)\top} e_s + b^{(z)} + \frac{\epsilon_{sc}}{2} \|x\|^2$$
- Convexity: W_{h,+}^{(\ell)} = softplus(W_{h,raw}^{(\ell)})
- Strong-convexity term: ε_sc = 1e-4

**ICNN training (semi-dual)**:
$$\mathcal{L}_{ICNN}(\psi; z) = \mathbb{E}_{x_\theta \sim P_{\theta,z}}[u_{\psi,z}(s, x_\theta)] + \mathbb{E}_{x_\beta \sim P_{\beta,z}}[u_{\psi,z}^*(s, x_\beta)]$$

with conjugate via inner gradient ascent (10 steps, lr=5e-2)

**Hyperparameters**:
- depth: 2-4 / 3
- width: 128-512 / 256
- ICNN lr: 1e-5 — 3e-4 / 1e-4
- n_icnn_steps per policy update: 1-5 / 2
- batch per mode: 16-128 / 32

#### Balanced Valid (s, z) Sampler

```python
def build_mode_samplers(D, R, n_min=128):
    samplers = {}
    for z in range(K):
        weights = R[:, z]   # soft GMM responsibility
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

### Inference: Receding-Horizon Only

```python
def act(s_t):
    # 1. Sample mode (overhead ~ 0.001ms)
    mode_probs = m_beta(s_t)             # [K]
    z_t = categorical_sample(mode_probs)
    
    # 2. K-step DDPM chunk sampling with FiLM (overhead ~ 0.05ms extra/block)
    a_chunk = ddpm_sample_chunk(
        epsilon_theta, state=s_t, mode=z_t,
        H=4, n_steps=K_eval=10, film=True
    )    # [H, d_a]
    
    # 3. Receding-horizon: execute first action only
    return a_chunk[0]
```

**chunk 仅作 mode-conditioning + DSM 稳定化对象，不作 open-loop plan。**

## Theoretical Grounding

### Main Theorem (with N_eff)

**Setup**: 
- chunk space $\mathcal{X} = [-1, 1]^{H d_a}$
- frozen mode classifier $c_\omega(z | s, x)$
- soft responsibilities $r_i(z) = c_\omega(z | s_i, x_i)$
- effective sample size:
  $$N_{eff}(z) = \frac{(\sum_{i=1}^N r_i(z))^2}{\sum_{i=1}^N r_i(z)^2}$$

**Definition (chunk-mode recall)**:
$$\text{Recall}_z(\pi_\theta) = \Pr_{s \sim D_z, x \sim \pi_\theta(\cdot | s, z)}[c_\omega(z|s,x) \geq \tau_c]$$

**Theorem**. Assume:
1. **Mode separation**: ∀ z' ≠ z, chunks in mode z 与 z' 在 chunk-space W₂ / classifier-margin 上分离 ≥ Δ
2. **ICNN OT estimator error**:
   $$|\widehat W_{2,z}^2 - W_{2,z}^2| \leq \epsilon_{stat}(z, \delta) + \epsilon_{opt}(z) + \epsilon_{conj}$$
   with prob ≥ 1-δ, where
   $$\epsilon_{stat}(z, \delta) = C R^2 \sqrt{\frac{\text{Pdim}(\mathcal{U}_\psi) + \log(K/\delta)}{N_{eff}(z)}}$$
   and $R = O(\sqrt{H d_a})$
3. **Classifier error on holdout**: $\epsilon_{clf}(z) = \Pr_{(s,x) \sim \beta_z}[c_\omega(z|s,x) < \tau_c]$
4. **Consistency loss bound**: $\mathcal{L}_{cons}(z) = \mathbb{E}_{x \sim \pi_\theta(\cdot|s,z)}[-\log c_\omega(z|s,x)] \leq \epsilon_c(z)$

**Conclusion**: For modes with $m_\beta(z) \geq \rho_{min}$ and $N_{eff}(z) \geq n_{min}$:
$$\text{Recall}_z(\pi_\theta) \geq 1 - O\left(\frac{\epsilon_{stat}(z,\delta) + \epsilon_{opt}(z) + \epsilon_{conj}}{\Delta^2}\right) - O\left(\frac{\epsilon_{clf}(z)}{\rho_{min}}\right) - O\left(\frac{\epsilon_c(z)}{\log 2}\right)$$
with prob ≥ 1-δ.

**Critical clarification**: Balanced replay equalizes optimization budget but **NOT** statistical sample size. Statistical error depends on $N_{eff}(z)$. ρ_min and n_min are explicit conditions; rare modes with $m_\beta < \rho_{min}$ are excluded from the guarantee (and the paper honestly reports this in "low-multimodality regime").

### Assumption Inventory

| Claim | Assumption | Class | Sanity |
|---|---|---|---|
| Mode separation Δ | chunk-space pairwise W₂ ≥ Δ | RESTRICTIVE measurable | Kitchen 实测 |
| ICNN bound | Pdim of u_ψ + i.i.d. weighted samples | STANDARD | Makkuva 2020 + Vacher 2021 |
| Effective sample size | balanced replay = optim ≠ statistics | DESIGN CHOICE | unit test; theorem honest |
| Classifier err on holdout | ε_clf measurable | STANDARD | per-task held-out |
| Consistency NLL bound | training converges | STANDARD | track during training |
| GMM K | dataset is K-mixture | RESTRICTIVE | BIC + oracle |
| Q err bounded | offline RL standard | STANDARD | Q residual report |

### Theory-Experiment Alignment Matrix (with new claims)

| Claim | Type | Validation Protocol | Scale | Feasibility |
|---|---|---|---|---|
| Mode-recall lower bound (with N_eff) | Sample complexity | Synthetic 4-mode chunk-MDP: Δ ∈ {0.5,1,2,5} × N_eff ∈ {64,256,1024,4096} × oracle/learned mode | 4×4×3 = 48 | FEASIBLE |
| Per-mode > global counter-example | Construction | closed-form 2-mode + 5 numeric chunk instances | trivial | FEASIBLE |
| ICNN bias on chunks (multi-mode) | Estimator bound | mode count m ∈ {2,4,8} × N_eff ∈ {1k, 10k} | 6 × 3 = 18 | FEASIBLE |
| **NEW: ε_stat scales with N_eff(z) not replay** | Estimator scaling | log-log plot of ICNN-W₂ error vs N_eff (replay-controlled) | reuse existing 48 synthetic | FEASIBLE |
| **NEW: consistency-to-recall via log 2 threshold** | Information-theoretic | Synthetic mixture; classifier error on held-out vs ε_c bound | 6 conditions × 3 seed | FEASIBLE |
| **NEW: chunk-mode scope (not trajectory/occupancy)** | Scope clarification | not new experiments needed; explicit narrative in paper | trivial | FEASIBLE |
| **CORE: Pareto dominate on D4RL multi-modal** | Empirical | 4 tasks (Kitchen-mixed, AntMaze-medium-play, AntMaze-large-play, Adroit-cloned) × 4 baselines × 3 seeds | 48 runs ≈ 7-8 days dual-4090 | FEASIBLE |
| Stretch: 4 more D4RL tasks | Empirical | + Kitchen-partial/complete + AntMaze-medium-diverse + Adroit-expert | 48 runs | FEASIBLE WITH CAVEATS |
| Appendix: NeoRL-2 1-2 task | Empirical | RocketRecovery + Pipeline × 3 seed | 6 runs | FEASIBLE WITH CAVEATS |
| Inference overhead < 5% | Computational complexity | 5 batch sizes × 3 seeds wall-clock | trivial | FEASIBLE |

无 NOT FEASIBLE。3 个新理论 claim 在现有 budget 内 absorb。

## Failure Modes & Diagnostics

| Failure | 检测 | Mitigation |
|---|---|---|
| GMM 不分离 | silhouette < 0.2 | task-id oracle；honest report |
| ICNN unstable | val W₂_ψ_z vs Sinkhorn 偏差 > 50% | Sinkhorn warm-up；reduce depth |
| Rare mode mass < 2% | m_β(z*) < ρ_min | "low-multimodality regime" disclaimer |
| Return 退 | per-mode return | 减 α；backup KL-BC |
| FiLM 失效 | conditional vs marginal action 距离 → 0 | 增 β 或 α；查 dropout |
| Chunk DSM 跳跃 | DSM loss 大幅波动 | 减 H (4→2)；增 K_steps |
| N_eff(z) 过小 | < n_min for some z | 报告该 mode unverified；不主张优势 |

## Novelty & Differentiation

| Work | Mechanism | Delta vs Mode-Conditional W-DiffPolicy |
|---|---|---|
| Q-DOT (RLC 2025) | ICNN W₂ + IQL global | 本: chunk diffusion + per-mode + N_eff theorem |
| OTPR (Feb 2025) | Q-cost OT + diffusion + online | 本: 不用 Q-cost；offline；per-mode |
| BDPO (ICML 2025) | reverse-kernel KL + diffusion | 本: per-mode W₂ + FiLM + balanced replay |
| Diffusion-QL (ICLR 2023) | KL-BC + diffusion (single action) | 本: chunk + per-mode W₂ |
| Diffusion Policy (Chi RSS 2023) | chunk diffusion + BC | 本: + per-mode W₂ + finite-sample theorem |
| LOM (ICLR 2025) | GMM + 选单 mode | 本: GMM + 保多 mode + 形式化定理 |
| Latent Diffusion ORL (ICLR 2024) | latent skill compression | 本: chunk action 不压缩；显式 mode preservation |
| P19 (Oct 2025) | LM RL KL collapse | 本: control 实证版 (Discussion)；W₂ 修复 |
| SWFP (Oct 2025) | flow + W₂ JKO + online | 本: diffusion + offline + per-mode |

**Why elegant**: 仅替换 Diffusion-QL chunk-variant loss 一项；Stage 0 一次性；新 trainable < 1M；inference overhead < 5%；定理与实现 1:1 对应。

## Hyperparameters Table (Summary)

| Name | Range | Default | Note |
|---|---:|---:|---|
| K (GMM modes) | 4-8 | BIC | per-task |
| H (chunk length) | 2-8 | 4 | mode-meaningful |
| α (W₂) | 0.03-3.0 | 0.3 | mode preservation |
| β (DSM) | 0.5-2.0 | 1.0 | diffusion BC |
| γ (consistency) | 0.03-1.0 | 0.3 | mode drift prevention |
| λ_Q | 0.1-1.0 | 0.5 | Q exploitation cap |
| classifier-free dropout | 0.02-0.15 | 0.05 | marginal fallback |
| policy lr | 1e-4 — 5e-4 | 3e-4 | Diffusion-QL compat |
| critic lr | 1e-4 — 5e-4 | 3e-4 | IQL standard |
| ICNN lr | 1e-5 — 3e-4 | 1e-4 | stable convex |
| encoder lr | 1e-4 — 1e-3 | 3e-4 | Stage 0 only |
| batch | 256-1024 | 512 | balanced replay |
| per-mode batch | 16-128 | 32 | ICNN stability |
| ICNN depth/width | 2-4 / 128-512 | 3 / 256 | Q-DOT-scale |
| conjugate steps | 5-20 | 10 | semi-dual |
| Stage 1 updates | 15k-50k | 30k | DSM warmup |
| Stage 2 updates | 50k-150k | 70k | mode-cond W₂ |
| IQL τ (expectile) | 0.7-0.9 | task-dep | 0.7 Kitchen, 0.9 AntMaze |
| discount γ_RL | 0.99-0.995 | 0.99 | D4RL |
| target update τ_t | 0.001-0.01 | 0.005 | stable Q target |
| ρ_min | 0.01-0.1 | 0.03 | rare mode floor |
| n_min (N_eff) | 64-512 | 128 | theorem validity |
| τ_c (classifier threshold) | 0.4-0.7 | 0.5 | recall threshold |
| δ (high-prob) | 0.01-0.1 | 0.05 | theorem |

## Evaluation Sketch

- **如何 validate**: synthetic 4-mode chunk-MDP → Kitchen + AntMaze (core 4 × 4 × 3) → stretch (4 more) → appendix Adroit + 1-2 NeoRL-2
- **关键指标**: rare-mode chunk recall × return Pareto；mode-mass error；per-mode return spread；inference wall-clock；N_eff(z) reported
- **Success**: Pareto strict dominate global-W₂/KL on 3/4 core tasks；toy figure：global W₂ → 0 但 mass 漂移；inference overhead < 5%
- **Failure**: Pareto 不 dominate 或 dominate 区域 < 30%；ICNN training 不稳；返回 honest report

## Resource Estimate

- **Scale**: MEDIUM (5-7 person-weeks)
- **Compute**: MEDIUM (双 4090, 48GB; Stage 0 ~ 6h; Stage 1+2 single task 4-6h; core 48 runs ≈ 7-8 days dual-card 含失败重跑)
- **Data**: 全 public (D4RL public; NeoRL-2 public)
