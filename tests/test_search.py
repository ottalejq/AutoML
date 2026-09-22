from ml.search import generate_all_configs


def test_generated_configs_are_unique():
    configs = generate_all_configs(
        model_search_space={"alpha": [0.01, 0.1], "l1_ratio": [0.25, 0.75]},
        fit_search_space={"batch_size": [16, 32]},
    )

    keys = [
        (
            tuple(sorted(model_config.items())),
            tuple(sorted(training_config.items())),
        )
        for model_config, training_config in configs
    ]

    assert len(keys) == len(set(keys)) == 8
    assert ({"alpha": 0.01, "l1_ratio": 0.25}, {"batch_size": 16}) in configs
    assert generate_all_configs({}, {}) == [({}, {})]