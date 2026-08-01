from pathlib import Path

import pytest

from direct_s2st.config import RootPaths, load_config


def test_profile_defaults_to_smoke() -> None:
    assert load_config(None)["profile"] == "smoke"


def test_profile_override_is_merged(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("value: 1\nprofiles:\n  pilot:\n    value: 2\n", encoding="utf-8")
    assert load_config(path, profile="pilot")["value"] == 2


def test_model_requires_revision(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("asr:\n  model: example/model\n", encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        load_config(path)


def test_roots_come_from_environment() -> None:
    roots = RootPaths.from_environment(
        {
            "CORPUS_ROOT": "/corpus",
            "EXPERIMENT_DATA_ROOT": "/experiments",
            "RUNS_ROOT": "/runs",
            "CACHE_ROOT": "/cache",
        }
    )
    assert roots.corpus == Path("/corpus")


def test_common_profile_is_merged_from_config_tree(tmp_path: Path) -> None:
    common = tmp_path / "configs" / "common"
    system = tmp_path / "configs" / "s2ut"
    common.mkdir(parents=True)
    system.mkdir(parents=True)
    (common / "pilot.yaml").write_text("target_hours: 10\n", encoding="utf-8")
    path = system / "train.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")
    resolved = load_config(path, profile="pilot")
    assert resolved["target_hours"] == 10
