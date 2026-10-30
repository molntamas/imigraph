import pytest

from imigraph.experiments import ExperimentConfig, run_experiment


def test_default_experiment_passes_the_comparison():
    result = run_experiment()
    assert result.passed, result.report.to_markdown()
    assert result.reference.num_frames == result.graph_projection.num_frames == 61
    assert result.graph_history.num_walkers == result.config.particles


def test_config_derives_diffusion_and_validates():
    config = ExperimentConfig(spacing=0.5, step_time=0.5)
    assert config.diffusion == pytest.approx(0.125)
    assert config.to_dict()["diffusion"] == pytest.approx(0.125)
    with pytest.raises(ValueError):
        ExperimentConfig(steps=100, record_every=30)


def test_seed_makes_the_experiment_reproducible():
    first = run_experiment(ExperimentConfig(particles=50, steps=100, record_every=10, seed=5))
    second = run_experiment(ExperimentConfig(particles=50, steps=100, record_every=10, seed=5))
    assert (first.reference.positions == second.reference.positions).all()
    assert (first.graph_history.anchors == second.graph_history.anchors).all()
