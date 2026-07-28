# AGPO 训练指南

本目录提供的 `CustomAGPOTrainer` 直接实现适应性群体策略优化（Adaptive Group Policy Optimization, AGPO）的采样与更新循环。它由 `llamafactory-cli train` 命令在强化学习阶段（`stage: ppo`）下、将 `rl_algo` 设为 `agpo` 时自动启用。下文概述代码结构、训练流程以及关键超参数设置。

## 架构概览

- `workflow.py` 在 API/full 奖励模式下加载普通 causal-LM 策略，不再创建未使用的 value head；只有本地 LoRA/OFT 奖励适配器确实需要奖励头时才保留该包装。
- `trainer.py` 直接管理 AGPO 群体采样、奖励归一化、自适应裁剪、温度调度与优化，同时复用 Accelerate、Transformers 和 LLaMA-Factory 的优化器、调度器及回调体系。
- 当 `stage` 为 `ppo` 且 `rl_algo` 设为 `agpo` 时，主训练入口 `tuner.py` 会调度上述工作流；否则退回到传统 PPO。【F:src/llamafactory/train/tuner.py†L63-L67】

## 训练流程

一次完整的 AGPO 步骤包含下列阶段：

1. **保存旧策略概率**：采样时保存当前策略的 detached token log-probability，作为裁剪比率的旧策略侧，不复制第二个可训练策略。
2. **探测采样 (Probe Sampling)**：针对每条提示，使用基础温度 `agpo_tau_base` 生成一组候选回复，计算奖励分散度、投票熵与可选偏度，用于评估不确定性。【F:src/llamafactory/train/agpo/trainer.py†L467-L490】
3. **温度自适应**：所有 prompt 都相对于同一个 minibatch EMA 快照计算控制量；收集结束后只用 minibatch 均值更新一次 EMA，从而消除固定统计量下的 prompt 顺序依赖。
4. **训练采样与优势归一化**：使用自适应温度重新生成响应，按所选分散度指标（标准差、MAD 或 IQR）归一化奖励，得到群体内优势值。【F:src/llamafactory/train/agpo/trainer.py†L502-L514】
5. **自适应裁剪**：结合奖励分散度与历史 KL (`step_kl`) 调节 PPO 裁剪范围，实现更稳健的策略更新。【F:src/llamafactory/train/agpo/trainer.py†L524-L544】
6. **统计、预算与日志**：记录温度、饱和率、分散度、KL、奖励和全局 generated-token 计数。若配置预算，最后一组会按剩余额度缩短每条序列，保证不超额。

> 注意：AGPO 训练器当前不支持显式的验证数据集加载，如需评估需在训练结束后单独运行推理或评测脚本。【F:src/llamafactory/train/agpo/trainer.py†L81-L84】

## 快速开始

1. 准备监督微调（SFT）产出的策略模型与奖励模型（LoRA 或全量权重均可）。
2. 复制一个 PPO 示例配置（如 `examples/train_lora/llama3_lora_ppo.yaml`），并修改以下关键字段：

```yaml
stage: ppo
rl_algo: agpo            # 启用 AGPO
reward_model: <path-to-reward-model>
reward_model_type: lora  # 或 full/api，需与实际奖励模型类型一致
agpo_group_size: 8       # 每个 prompt 采样的回复数量
agpo_count_probe_tokens_in_budget: true
agpo_max_generated_tokens: 300000000  # 论文协议；按 step 训练时可省略
```

3. 使用 CLI 启动训练：

```bash
llamafactory-cli train path/to/your_agpo_config.yaml
```

如需分布式或 Ray 训练，可沿用官方 README 中关于 `FORCE_TORCHRUN` 或 `USE_RAY` 的启动方式；AGPO 与现有基础设施兼容。【F:src/llamafactory/train/tuner.py†L63-L98】

## 关键超参数

| 参数 | 默认值 | 作用 | 代码位置 |
| --- | --- | --- | --- |
| `agpo_group_size` | 8 | 每个 prompt 生成的响应数量，决定群体统计的稳定性。 | 【F:src/llamafactory/hparams/finetuning_args.py†L227-L230】 |
| `agpo_tau_base` / `agpo_tau_min` / `agpo_tau_max` | 1.0 / 0.5 / 1.5 | 探测与训练时的温度上下界，控制采样多样性。 | 【F:src/llamafactory/hparams/finetuning_args.py†L259-L270】 |
| `agpo_lambda_temp` | 0.15 | 将不确定性映射为温度增益的比例系数。 | 【F:src/llamafactory/hparams/finetuning_args.py†L271-L274】 |
| `agpo_use_robust_dispersion` | `std` | 奖励分散度度量方式（标准差、MAD 或 IQR）。 | 【F:src/llamafactory/hparams/finetuning_args.py†L255-L258】 |
| `agpo_w_r` / `agpo_w_e` / `agpo_w_k` | 1.0 / 1.0 / 0.0 | 组合奖励分散度、投票熵与偏度的权重，用于估计不确定性。 | 【F:src/llamafactory/hparams/finetuning_args.py†L275-L286】 |
| `agpo_eps_base` / `agpo_eps_min` / `agpo_eps_max` | 0.2 / 0.05 / 0.4 | PPO 裁剪区间的基础值与上下限。 | 【F:src/llamafactory/hparams/finetuning_args.py†L231-L242】 |
| `agpo_alpha_var` / `agpo_gamma_stepkl` | 1.0 / 0.5 | 依据奖励方差与历史 KL 动态调整裁剪半径。 | 【F:src/llamafactory/hparams/finetuning_args.py†L243-L249】 |
| `agpo_beta_ref_kl` | 0.03 | 参考策略 KL 正则项权重。 | 【F:src/llamafactory/hparams/finetuning_args.py†L287-L289】 |
| `agpo_log_probe_metrics` | True | 是否记录探测阶段统计，便于诊断。 | 【F:src/llamafactory/hparams/finetuning_args.py†L295-L297】 |
| `agpo_count_probe_tokens_in_budget` | True | 是否将探测样本的 token 计入预算，避免超长序列。 | 【F:src/llamafactory/hparams/finetuning_args.py†L291-L293】 |
| `agpo_max_generated_tokens` | `None` | 可选的全局 continuation-token 预算；预留 probe/train 两组容量、缩短最终组并保证不超额。 | `finetuning_args.py` |

## 日志与可视化

- 训练过程中会周期性输出 `loss`、`reward`、`learning_rate` 等指标，同时在 `agpo_trainer.state.log_history` 中追加详细记录，便于对接 W&B / TensorBoard 等外部工具。【F:src/llamafactory/train/agpo/trainer.py†L412-L440】
- 每个 prompt 的统计信息会写入 `agpo/*` 命名空间，例如 `agpo/tau`、`agpo/sigma`、`agpo/step_kl`、`agpo/tau_saturated` 与 `agpo/budget/*`。
- 输出目录中的 `agpo_token_audit.json` 会原子更新，记录 probe/train token、计费总量、剩余额度、world size 与最终组规则。
- `agpo_controller_trace.rank<N>.jsonl` 为每个完成的 prompt 写入一行控制器记录，并用 token ID 的 SHA-256 指纹对齐样本，便于复算相关性与饱和率而不保存 prompt 明文。
- 若在配置中启用 `plot_loss: true`，训练结束后会自动生成包含损失与奖励曲线的图像文件。【F:src/llamafactory/train/agpo/workflow.py†L72-L74】

## 实践建议

- 当奖励模型波动较大时，可适当调高 `agpo_w_r` 或 `agpo_alpha_var` 以增强稳定性；若希望鼓励多样化响应，可提高 `agpo_w_e` 与 `agpo_tau_max`。
- 若使用 API 奖励模型（`reward_model_type: api`），请确保服务端实现了 `get_rewards_from_server` 所需的接口格式。【F:src/llamafactory/train/agpo/trainer.py†L611-L638】
- 复现论文的 token-matched 协议时，必须同时设置 `agpo_count_probe_tokens_in_budget: true` 与 `agpo_max_generated_tokens: 300000000`，并保留每次运行的 `agpo_token_audit.json`。
- AGPO 仍依赖基础 PPO 设置（学习率、梯度裁剪、混合精度等），可沿用既有经验调整这些参数。【F:src/llamafactory/train/agpo/trainer.py†L85-L154】【F:src/llamafactory/train/agpo/trainer.py†L407-L610】

欢迎在 Issues 中反馈使用体验或贡献新的示例配置，帮助我们进一步完善 AGPO 生态。
