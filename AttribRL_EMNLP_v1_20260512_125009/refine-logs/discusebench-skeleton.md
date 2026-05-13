# Skeleton — DiscUseBench (Plan B)

## State A
- A1: Anchor P01 在 2 benchmark × 4 model 上发现 97.54% vs 0.53% discovery-use gap
- A2: 没人质疑 final accuracy 是否掩盖 D/U/C 三元 failure composition
- A3: LLM judge 判定 "use" 被默认可靠
- A4: closed-model leaderboards 通常不冻结版本，难复现

## State B
- B1: 14 model × 5 benchmark D/U/C 三维分布是稳定 empirical regularity；不同 model family 显示 systematically different conversion bottleneck
- B2: outcome-only RL 的 published gain 来自更频繁 search 而非更好 utilization（success metric 掩盖 failure composition 的实证）
- B3: LLM judge 在 "use" 判定上 false-positive rate > κ；causal counterfactual oracle 给出更可证伪的 use definition
- B4: closed-model 版本冻结 + dockerized harness + raw trajectory 公开 = reproducible diagnostic benchmark
- B5: ≥2 个新 conversion laws：(i) observation salience → conversion 单调；(ii) parametric knowledge leakage isolation → use rate inflation 量化

## Skeleton Path (5 步)
1. Defining D/U/C with causal counterfactual oracle (not LLM judge alone)
2. Building 14×5 grid + frozen closed-model versions + dockerized harness
3. Mechanism decomposition: 3 failure modes (overconfidence / unreadability / delayed-integration) clustered cross-model
4. Reproducibility: judge agreement κ ≥ 0.75; bootstrap CI; raw trajectory release; open-model-only degraded subset
5. ≥2 conversion laws (empirical regularities) that change downstream RL design assumption
