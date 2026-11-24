from pathlib import Path


def validate_checkpoint(path: Path, *, include_lineage=False) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"real training checkpoint required: {path}")
    import torch
    # fairseq checkpoints contain trusted local OmegaConf/Namespace training state.
    state = torch.load(path, map_location="cpu", weights_only=False)
    history = state.get("optimizer_history", [])
    if not history or int(history[-1].get("num_updates", 0)) < 1 or not state.get("last_optimizer_state"):
        raise ValueError("checkpoint has no completed optimizer update/state")
    parameters = state.get("model", {})
    if not parameters or not all(torch.is_tensor(value) and torch.isfinite(value).all().item() for value in parameters.values()):
        raise ValueError("checkpoint has empty or non-finite model state")
    for name in ("source_letter_decoder", "target_letter_decoder", "decoder_target_ctc_decoder"):
        if not any(key.startswith(name + ".") for key in parameters):
            raise ValueError(f"checkpoint is missing auxiliary parameters: {name}")
    def finite(value):
        import math
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return True
    if not finite(state['last_optimizer_state']):
        raise ValueError('checkpoint has non-finite optimizer state')
    if not state['last_optimizer_state'].get('state'):
        raise ValueError('checkpoint has no saved optimizer parameter state')
    scale = state['last_optimizer_state'].get('loss_scale')
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0):
        raise ValueError('checkpoint has invalid FP16 loss scale')
    monitor = state.get('extra_state', {}).get('s2st_monitor')
    if monitor is not None:
        from .monitor import S2UTMonitor
        monitor_class = S2UTMonitor
        if monitor.get('version') == 's2ut-paper-report-v1':
            from .paper_monitor import PaperMonitor
            monitor_class = PaperMonitor
        monitor_class.restore(None, monitor, int(history[-1]['num_updates']), monitor['warmup'], path.parent.parent)
    result = {"num_updates": int(history[-1]["num_updates"]), "checkpoint": str(path), "finite_parameters": True}
    if include_lineage:
        import hashlib
        import json
        import re
        from ..journal import digest
        from ..hashing import sha256_file
        fingerprint = state.get('extra_state', {}).get('s2st_recipe_fingerprint')
        root = path.parent.parent
        try:
            metadata = (root/'research-metadata.json').read_bytes()
            lock = (root/'s2ut-recipe-lock.json').read_bytes()
            recipe = json.loads(lock)
            locked = recipe['recipe'].pop('training_fingerprint')
            meta = json.loads(metadata)
            if (isinstance(fingerprint, str) and re.fullmatch('[0-9a-f]{64}', fingerprint)
                    and fingerprint == locked == digest(recipe) == meta.get('training_fingerprint')
                    and meta.get('completed_updates') == result['num_updates']):
                result['s2ut_lineage'] = {
                    'training_fingerprint': fingerprint, 'updates': result['num_updates'],
                    'checkpoints/checkpoint_last.pt': sha256_file(path),
                    'research-metadata.json': hashlib.sha256(metadata).hexdigest(),
                    's2ut-recipe-lock.json': hashlib.sha256(lock).hexdigest(),
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Older or incomplete optional evidence must not stop training.
    return result
