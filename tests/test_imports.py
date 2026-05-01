def test_imports():
    from src.models.multitask import DeepGuardMultiTask
    from src.utils.config import load_config

    assert DeepGuardMultiTask is not None
    assert load_config is not None
