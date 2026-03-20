# AGPO Training Guide

This directory provides `CustomAGPOTrainer`, an extension of the PPO training stack for **Adaptive Group Policy Optimization (AGPO)**. When you launch reinforcement learning with `llamafactory-cli train` under `stage: ppo` and set `rl_algo: agpo`, the AGPO workflow is enabled automatically. This README explains the implementation structure, the training loop, and how the code corresponds to the AGPO method.

## Method Overview

AGPO is designed around **one probe and two controllers**:

- A **probe phase** samples grouped rollouts at a base temperature and extracts a shared statistical state.
- An **adaptive temperature controller (ATS)** adjusts rollout diversity based on uncertainty.
- An **adaptive clipping controller** sets the PPO/GRPO clip radius based on uncertainty and optimization stability signals.

In the method formulation, the key statistics are:

- **Reward dispersion** `\hat{\sigma}`: measures how spread the grouped rewards are.
- **Reward skewness** `|\tilde{\kappa}_3(r)|`: captures asymmetric tails in the reward distribution.
- **Probe vote entropy** `\mathcal{E}_{probe}`: measures answer-level disagreement among probe rollouts.
- **Policy entropy** `H(\pi)`: reflects exploration capacity of the current policy.
- **Step-wise KL drift** `\widehat{D}_{KL}^{step}`: tracks how far the new policy moves from the previous one.

The overall idea is:

1. Use a probe rollout group to estimate uncertainty.
2. Increase or decrease sampling temperature around `tau_base` so that uncertain batches explore more and easier batches exploit more.
3. Adapt the clip radius so that updates are larger when exploration is needed and smaller when instability risk is high.

This shared-probe design is the central idea of AGPO: **the same probe statistics jointly control exploration and update magnitude**.

## How the Code Maps to the Method

### 1. Grouped rollouts and group-normalized advantage

For each prompt, AGPO samples a **group** of candidate responses from the **same policy** rather than using multiple agents. Each response receives a reward, and the trainer normalizes rewards within the group to produce rollout-level advantages. In practice, those rollout-level advantages are broadcast to token positions when computing the loss.

This corresponds to the grouped-rollout reinforcement learning setup in the method section, where AGPO inherits the GRPO-style surrogate and replaces fixed clipping with an adaptive controller.

### 2. Probe phase

Before the actual training rollouts are collected, the trainer performs a **probe sampling** step at `agpo_tau_base`. The probe responses are used only to estimate controller statistics and are not reused for gradient updates.

The probe stage computes:

- reward dispersion (`std`, `MAD`, or `IQR` depending on configuration),
- vote entropy over extracted answers,
- optional skewness statistics,
- uncertainty scores for temperature adaptation.

This implements the method's probe--train loop.

### 3. Adaptive Temperature Sampling (ATS)

The temperature controller uses the uncertainty score built from:

- reward dispersion,
- probe vote entropy,
- reward skewness.

The raw uncertainty score is centered by a running EMA baseline. If the current batch is more uncertain than the running baseline, temperature is increased; otherwise it is decreased, while always staying inside `[agpo_tau_min, agpo_tau_max]`.

Conceptually, this matches the ATS rule in the method section:

- higher-than-baseline uncertainty `=>` hotter sampling,
- lower-than-baseline uncertainty `=>` cooler sampling.

### 4. Adaptive clipping

Instead of using a fixed PPO clip radius, AGPO computes an adaptive clipping value from:

- a base clip value,
- reward dispersion,
- optional skewness,
- policy entropy,
- detached step-wise KL drift.

The goal is to enlarge the feasible update region when the policy should remain exploratory, but shrink it when reward variability or policy drift suggests instability risk.

In other words:

- **more uncertainty / entropy** can justify a wider update window,
- **more variance / asymmetry / KL drift** tighten the trust region.

### 5. AGPO objective

The final training objective keeps the GRPO/PPO clipped-surrogate structure, but substitutes the fixed `epsilon` with `epsilon_adaptive`. The reference-policy KL regularizer still acts as an anchor, while the detached step-wise KL is used only as a controller signal.

This separation is important:

- `D_KL(pi_old || pi_new)` is used as a **drift signal** for control,
- `D_KL(pi_theta || pi_ref)` is used as a **regularizer** in optimization.

## Architecture Overview

- `workflow.py` loads the tokenizer, dataset, policy model with value head, reference model, and reward model, then instantiates `CustomAGPOTrainer` and runs training.
- `trainer.py` extends `trl.PPOTrainer` with AGPO-specific grouped sampling, reward normalization, adaptive temperature scheduling, adaptive clipping, and logging.
- `tuner.py` dispatches to the AGPO workflow when `stage=ppo` and `rl_algo=agpo`; otherwise it falls back to the standard PPO path.

## Training Loop

A full AGPO update step follows this sequence:

1. **Sync the old policy**  
   At the beginning of each gradient step, the current policy is copied so that importance ratios and detached KL drift can be measured against the previous policy snapshot.

2. **Probe sampling**  
   For each prompt, the trainer samples one probe group at `agpo_tau_base` and computes reward dispersion, vote entropy, and optional skewness.

3. **Adaptive temperature update**  
   The trainer converts the probe statistics into an uncertainty score, centers it using an EMA baseline, and updates the rollout temperature within the configured bounds.

4. **Training rollout sampling and advantage normalization**  
   The trainer resamples responses using the adaptive temperature, evaluates rewards, and normalizes grouped rewards into advantages.

5. **Adaptive clipping update**  
   The PPO/GRPO clip radius is adjusted using reward dispersion and historical step-wise KL drift, together with the configured controller bounds.

6. **Optimization and logging**  
   The policy is updated, and metrics such as temperature, dispersion, KL, and rewards are recorded for debugging and visualization.

> Note: the current AGPO trainer does not load a validation dataset directly. If you need evaluation, run a separate inference or benchmark step after training.

## Quick Start

1. Prepare a policy model from supervised fine-tuning (SFT) and a reward model.
2. Copy a PPO configuration file such as `examples/train_lora/llama3_lora_ppo.yaml` and update the AGPO-specific fields:

```yaml
stage: ppo
rl_algo: agpo
reward_model: <path-to-reward-model>
reward_model_type: lora   # or full / api
agpo_group_size: 8
```

3. Launch training:

```bash
llamafactory-cli train path/to/your_agpo_config.yaml
```

If you need distributed or Ray-based training, you can reuse the standard launch patterns already supported by the project.

## Key Hyperparameters

| Parameter | Default | Description |
| --- | --- | --- |
| `agpo_group_size` | 8 | Number of responses sampled per prompt. Larger groups provide more stable group statistics. |
| `agpo_tau_base` / `agpo_tau_min` / `agpo_tau_max` | 1.0 / 0.5 / 1.5 | Base temperature and allowable temperature range for probe and training sampling. |
| `agpo_lambda_temp` | 0.15 | Scales how strongly centered uncertainty changes the temperature. |
| `agpo_use_robust_dispersion` | `std` | Reward dispersion estimator: standard deviation, MAD, or IQR. |
| `agpo_w_r` / `agpo_w_e` / `agpo_w_k` | 1.0 / 1.0 / 0.0 | Weights for reward dispersion, vote entropy, and skewness in the ATS uncertainty score. |
| `agpo_eps_base` / `agpo_eps_min` / `agpo_eps_max` | 0.2 / 0.05 / 0.4 | Base clip radius and controller bounds for adaptive clipping. |
| `agpo_alpha_var` / `agpo_gamma_stepkl` | 1.0 / 0.5 | Sensitivity of adaptive clipping to reward variance and step-wise KL drift. |
| `agpo_beta_ref_kl` | 0.03 | Weight of the reference-policy KL regularizer. |
| `agpo_log_probe_metrics` | True | Whether to log probe-stage statistics for diagnostics. |
| `agpo_count_probe_tokens_in_budget` | True | Whether probe tokens count toward the rollout token budget. |

## Logging and Visualization

- The trainer periodically records metrics such as `loss`, `reward`, and `learning_rate`.
- AGPO-specific metrics are logged under names such as `agpo/tau`, `agpo/sigma`, and `agpo/step_kl`.
- If `plot_loss: true` is enabled, the workflow can generate training curves after training finishes.

Because AGPO depends heavily on controller dynamics, these logs are especially useful for diagnosing whether the model is over-exploring, collapsing to low-diversity outputs, or drifting too aggressively.

## Practical Recommendations

- If the reward model is noisy, consider increasing `agpo_w_r` or `agpo_alpha_var` to make the controller more conservative.
- If you want more response diversity, consider increasing `agpo_w_e` or raising `agpo_tau_max`.
- If you use an API reward model (`reward_model_type: api`), make sure the serving endpoint matches the format expected by `get_rewards_from_server`.
- AGPO still depends on the usual PPO settings such as learning rate, gradient clipping, precision mode, and rollout budget, so tune them together rather than in isolation.

## In One Sentence

AGPO combines **grouped rollouts**, **adaptive temperature sampling**, and **adaptive clipping** into a single probe--train reinforcement learning loop so that the model explores more when uncertainty is high and updates more conservatively when optimization risk rises.
