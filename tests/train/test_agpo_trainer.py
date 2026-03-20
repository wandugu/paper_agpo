import math

from llamafactory.train.agpo.trainer import CustomAGPOTrainer, load_agpo_config


def test_load_agpo_config():
    config = load_agpo_config()
    assert config['controller']['ema_alpha'] == 0.1
    assert config['logging']['debug_reward_samples'] is True


def test_compute_centered_uncertainty_and_temperature():
    centered, updated_ema = CustomAGPOTrainer.compute_centered_uncertainty(2.0, 1.0, 0.25)
    assert math.isclose(updated_ema, 1.25)
    assert math.isclose(centered, 0.75)

    tau = CustomAGPOTrainer.compute_adaptive_temperature(
        tau_base=1.0,
        lambda_temp=0.2,
        centered_uncertainty=0.75,
        tau_min=0.5,
        tau_max=1.5,
    )
    assert math.isclose(tau, 1.15, rel_tol=1e-6)


def test_compute_adaptive_clip_uses_entropy_and_skew():
    eps = CustomAGPOTrainer.compute_adaptive_clip(
        eps_base=0.2,
        eps_min=0.05,
        eps_max=0.4,
        reward_dispersion=0.4,
        abs_skew=0.3,
        policy_entropy=1.5,
        step_kl=0.2,
        alpha_var=1.0,
        gamma_stepkl=0.5,
        zeta_skew=0.2,
        entropy_scale=0.1,
        entropy_floor=1e-8,
    )
    expected = 0.2 * (1 + 0.15) / (1 + 0.4 + 0.1 + 0.06)
    assert math.isclose(eps, expected, rel_tol=1e-6)
