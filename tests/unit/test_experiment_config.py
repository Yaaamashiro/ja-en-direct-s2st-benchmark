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
            "EXPERIMENT_DATA_ROOT": "/benchmark",
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


def test_training_profiles_control_update_counts() -> None:
    root = Path(__file__).resolve().parents[2]
    config = root / "configs" / "s2ut" / "train.yaml"
    smoke = load_config(config, profile="smoke")
    pilot = load_config(config, profile="pilot")
    full = load_config(config, profile="full")

    assert smoke["training"]["max_updates"] == 100
    assert pilot["training"]["max_updates"] == 10_000
    assert full["training"]["max_updates"] == 100_000
    assert full["confirm_full"] is True


def test_fairseq_training_commands_use_the_pinned_s2s_interfaces() -> None:
    root = Path(__file__).resolve().parents[2]
    s2ut = load_config(root / "configs" / "s2ut" / "train.yaml")
    translatotron2 = load_config(
        root / "configs" / "translatotron2" / "train.yaml"
    )

    s2ut_command = s2ut["training"]["command"]
    assert s2ut_command[s2ut_command.index("--task") + 1] == "speech_to_speech"
    assert "--target-is-code" in s2ut_command
    assert s2ut_command[s2ut_command.index("--target-code-size") + 1] == "100"
    assert s2ut_command[s2ut_command.index("--criterion") + 1] == "speech_to_unit"
    assert "--optimizer" in s2ut_command
    assert "--lr-scheduler" in s2ut_command
    assert "--max-tokens" in s2ut_command

    t2_command = translatotron2["training"]["command"]
    assert t2_command[t2_command.index("--task") + 1] == "speech_to_speech"
    assert (
        t2_command[t2_command.index("--criterion") + 1]
        == "speech_to_spectrogram"
    )
    assert t2_command[t2_command.index("--n-frames-per-step") + 1] == "5"
    assert "--optimizer" in t2_command
    assert "--lr-scheduler" in t2_command
    assert "--max-tokens" in t2_command
