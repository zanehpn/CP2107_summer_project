# Socratic Turn 2 — Answers (Claude)

[See main conversation for full detailed answers. Key upgrades:]

## A1. EMA on non-probe step: carry forward EMA(t+1)=EMA(t), no spurious update.

## A2. Two-layer KL calibration:
- Layer 1: absolute floor κ=0.05 nats (pre-trained random distractor 95-percentile)
- Layer 2: null-distractor baseline subtraction: $R_{cf}^{\text{net}} = \max(0, R_{cf}^{\text{matched}} - R_{cf}^{\text{null}})$
- 只奖励 evidence-specific sensitivity，不奖励 generic obs-change sensitivity

## A3. Two-stage token localization:
- Function-choice token + Argument-slot token (schema-aware extraction)
- $R_{cf,t} = w_{\text{fn}} \cdot \text{KL}_{\text{fn}} + w_{\text{arg}} \cdot \text{KL}_{\text{arg}}$, default $w_{\text{fn}}=0.3, w_{\text{arg}}=0.7$
- Validation: argument-slot detection accuracy ≥ 0.85 on 50 human-annotated trajectories

## A4. D_t prompt v2: judges *obs content has usable evidence*, decoupled from agent action
- Two-judge consensus + 200 human spot check (κ ≥ 0.75 target)
- boilerplate copy → D_t = 0 → U_t = 0 (filter at D_t not Mask)

## A5. 10-variant ablation matrix
- V0 (outcome only) / V1 (dense lex) / V2 (R_lex+Mask+D_t) / V3 (R_cf+Mask+D_t) / V4 (no schedule) / V5 (no Mask) / V6 (no D_t) / V7 (full AttribRL) / V8 (CST-style) / V9 (AgenTracer-style)
- V7 vs V1 排除 dense shaping; V7 vs V8 排除 CoT predictability; V7 vs V2/V3 双源; V7 vs V4/5/6 gating necessity
- Paired t-test with Bonferroni correction
- Budget: ~18 GPU-day, fit 2×4090 4-week plan

---

## GPT 最终回应

"我已充分理解这个方法"

→ 进入 final scoring phase。
