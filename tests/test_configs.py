from shared.config import config_for, merge
from shared.models import SUPPORTED


def test_config_merge():
    assert merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}
    for model in SUPPORTED:
        config = config_for(model)
        assert config["model"] == model
        assert config["dataset"]["num_classes"] == 102
        assert config["training"]["epochs"] > 0
