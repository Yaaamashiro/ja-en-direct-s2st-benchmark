from pathlib import Path


def validate_checkpoint(path: Path) -> dict:
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
    return {"num_updates": int(history[-1]["num_updates"]), "checkpoint": str(path), "finite_parameters": True}
