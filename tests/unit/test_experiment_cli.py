from direct_s2st.cli import build_parser, command_key, main


def test_cli_exposes_required_command_tree() -> None:
    args = build_parser().parse_args(["s2ut", "extract-units", "--dry-run"])
    assert command_key(args) == "s2ut.extract-units"
    assert args.profile == "smoke"


def test_dry_run_does_not_execute_models(capsys: object, monkeypatch: object) -> None:
    for name in ("CORPUS_ROOT", "EXPERIMENT_DATA_ROOT", "RUNS_ROOT", "CACHE_ROOT"):
        monkeypatch.setenv(name, "/fixture")
    assert main(["cascade", "run", "--dry-run"]) == 0


def test_invalid_shard_is_rejected() -> None:
    assert main(["s2ut", "prepare", "--shard-index", "2", "--num-shards", "2"]) == 2
