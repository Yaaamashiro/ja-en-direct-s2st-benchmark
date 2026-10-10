from pathlib import Path
import yaml

from .multitask import validate_prepared, read_tsv


def validate_training(root: Path, command: list[str]) -> dict:
    def option(name: str, default=None):
        return command[command.index(name) + 1] if name in command else default
    if option("--multitask-config-yaml") != "config_multitask.yaml":
        raise ValueError("main S2UT baseline requires --multitask-config-yaml config_multitask.yaml")
    if option("--train-subset") != "train" or option("--valid-subset") != "dev":
        raise ValueError("S2UT training must use train, validation must use dev")
    result = validate_prepared(root, clusters=int(option("--target-code-size", 100)))
    from fairseq.models import ARCH_CONFIG_REGISTRY
    from argparse import Namespace
    architecture = option("--arch")
    if architecture not in ARCH_CONFIG_REGISTRY:
        raise ValueError(f"unregistered fairseq architecture: {architecture}")
    args = Namespace()
    for key in ("encoder_layers", "decoder_layers"):
        value = option("--" + key.replace("_", "-"))
        if value is not None:
            setattr(args, key, int(value))
    ARCH_CONFIG_REGISTRY[architecture](args)
    if architecture == 's2ut_transformer_fisher':
        fields = ('encoder_layers', 'encoder_embed_dim', 'encoder_ffn_embed_dim', 'encoder_attention_heads',
                  'decoder_layers', 'decoder_embed_dim', 'decoder_ffn_embed_dim', 'decoder_attention_heads')
        expected = (12, 256, 2048, 4, 6, 256, 2048, 8)
        for key, value in zip(fields, expected):
            override = option('--' + key.replace('_', '-'))
            if getattr(args, key) != value or (override is not None and int(override) != value):
                raise ValueError('Fisher architecture dimension mismatch: ' + key)
        result['model_dimensions'] = dict(zip(fields, expected))
    config = yaml.safe_load((root / "config_multitask.yaml").read_text(encoding="utf-8"))
    if architecture == 's2ut_transformer_fisher':
        from .multitask import load_settings
        for task, expected_task in load_settings().items():
            if any(config[task].get(key) != value for key, value in expected_task.items()):
                raise ValueError('Fisher auxiliary task/loss configuration mismatch: ' + task)
    for task, cfg in config.items():
        side = "decoder" if "decoder_layer" in cfg else "encoder"
        if not 1 <= cfg[f"{side}_layer"] <= getattr(args, f"{side}_layers"):
            raise ValueError(f"attachment layer out of range: {task}")
    for split in ("train", "dev", "test"):
        main = read_tsv(root / f"{split}.tsv", ("id", "src_audio", "src_n_frames", "tgt_audio", "tgt_n_frames"))
        aux = read_tsv(Path(config["decoder_target_ctc"]["data"]) / f"{split}.tsv", ("id", "tgt_text"))
        for pair_id, row in aux.items():
            tokens = row["tgt_text"].split()
            minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
            if int(main[pair_id]["tgt_n_frames"]) + 1 < minimum:
                raise ValueError(f"impossible decoder CTC alignment: {split}/{pair_id}; requires {minimum} steps")
    return result
