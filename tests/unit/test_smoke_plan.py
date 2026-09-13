import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "scripts/smoke/run.py"
spec = importlib.util.spec_from_file_location("smoke_driver", path)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_smoke_accepts_environment_and_is_bounded(tmp_path):
    configs, stages = driver.build_plan("custom-run", 3, 2, tmp_path / ".env", "my-context")
    assert configs["train"]["training"]["max_updates"] == 2
    assert {cfg["run_id"] for cfg in configs.values()} == {"custom-run"}
    assert all(stage["status"] == "NOT_RUN" for stage in stages)
    assert stages[0]["command"][:3] == ["docker", "--context", "my-context"]
    assert not tmp_path.joinpath(".env").exists()
    with pytest.raises(ValueError, match="smoke permits"):
        driver.build_plan("custom-run", 101, 2, tmp_path / ".env")
