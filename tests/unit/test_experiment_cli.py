from pathlib import Path

from direct_s2st.cli import _default_config, build_parser, command_key, main


def test_cli_exposes_required_command_tree() -> None:
    args = build_parser().parse_args(["s2ut", "extract-units", "--dry-run"])
    assert command_key(args) == "s2ut.extract-units"
    assert args.profile == "smoke"


def test_dry_run_does_not_execute_models(capsys: object, monkeypatch: object) -> None:
    for name in ("CORPUS_ROOT", "EXPERIMENT_DATA_ROOT", "RUNS_ROOT", "CACHE_ROOT"):
        monkeypatch.setenv(name, "/fixture")
    assert main(["cascade", "run", "--dry-run"]) == 0
    assert main(["s2ut", "fetch-artifacts", "--dry-run"]) == 0


def test_invalid_shard_is_rejected() -> None:
    assert main(["s2ut", "prepare", "--shard-index", "2", "--num-shards", "2"]) == 2


def test_default_config_root_can_be_set_by_container(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.setenv("S2ST_CONFIG_ROOT", str(tmp_path))
    args = build_parser().parse_args(["s2ut", "extract-units", "--dry-run"])
    assert _default_config(args) == tmp_path / "s2ut" / "prepare.yaml"
