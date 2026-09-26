"""Shared Whisper loading; task semantics belong to ASR/S2T adapters."""
def load_whisper_pipeline(model_id, revision, device=0):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    dtype = torch.float16 if device >= 0 else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    if device >= 0:
        model = model.to(f"cuda:{device}")
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )

