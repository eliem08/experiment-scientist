from prompt_experiment import DEFAULT_EXPERIMENT

def test_experiment_pairs_are_complete():
    assert DEFAULT_EXPERIMENT
    for case in DEFAULT_EXPERIMENT:
        assert case["id"]
        assert case["baseline"].strip()
        assert case["treatment"].strip()
