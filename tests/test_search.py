from ml.search import generate_all_configs


def test_generated_configs_are_unique():
    configs = generate_all_configs()

    keys = [
        (
            tuple(sorted(model_config.items())),
            tuple(sorted(training_config.items())),
        )
        for model_config, training_config in configs
    ]

    assert len(keys) == len(set(keys))