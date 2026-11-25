from hydra import compose, initialize_config_dir


def test_dspo_uses_distinct_condition_labels(project_root) -> None:
    with initialize_config_dir(
        config_dir=str(project_root / "configs"), version_base="1.3"
    ):
        cfg = compose(config_name="train_dspo_ldm.yaml")

    assert cfg.diffusion_module.denoiser.num_datasets == 5
