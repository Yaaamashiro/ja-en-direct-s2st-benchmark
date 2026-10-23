from __future__ import annotations

import os
import json
import re
import tempfile
from pathlib import Path
from ..progress import operation


REQUIRED_FILES = (
    'config.json', 'generation_config.json', 'preprocessor_config.json',
    'tokenizer_config.json', 'vocab.json', 'merges.txt', 'model.safetensors',
    'speech_tokenizer/config.json', 'speech_tokenizer/preprocessor_config.json',
    'speech_tokenizer/model.safetensors',
)


@operation('Qwen TTS: download/check one complete pinned snapshot')
def pinned_snapshot(model_id: str, revision: str) -> Path:
    """qwen-tts 0.1.1 drops revision in nested cached_file/processor calls.

Give the SDK a complete LOCAL directory, never a repo ID. Its nested loads
then cannot resolve a second (main) revision, including with a partial cache.
"""
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Qwen TTS requires a full immutable revision')
    from huggingface_hub import snapshot_download, constants
    cache = Path(constants.HF_HUB_CACHE)
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and cache.resolve().is_relative_to(Path(corpus).resolve()):
        raise ValueError('Qwen download cache must be outside CORPUS_ROOT')
    # No allow_patterns: include root processor/tokenizer AND speech_tokenizer.
    # Hub downloads are resumable and reuse existing blobs at the same revision.
    root = Path(snapshot_download(repo_id=model_id, revision=revision))
    if root.name != revision or root.parent.name != 'snapshots':
        raise ValueError('Qwen snapshot path does not match requested revision')
    missing = [name for name in REQUIRED_FILES if not (root/name).is_file() or (root/name).stat().st_size == 0]
    if missing:
        raise ValueError('incomplete pinned Qwen snapshot; no model loaded: '+', '.join(missing))
    for name in REQUIRED_FILES:
        if name.endswith('.json'):
            value = json.loads((root/name).read_text(encoding='utf-8'))
            if not isinstance(value, dict) or not value:
                raise ValueError('invalid Qwen JSON artifact: '+name)
    return root


class QwenTTS:
    @operation('model: load Qwen TTS')
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        speaker: str,
        language: str,
        device: str = "cuda:0",
    ) -> None:
        snapshot = pinned_snapshot(model_id, revision)
        import torch
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            str(snapshot),
            local_files_only=True,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.speaker = speaker
        self.language = language

    def __call__(self, text: str, output_path: Path) -> None:
        import soundfile as sf

        waveforms, sample_rate = self.model.generate_custom_voice(
            text=text,
            language=self.language,
            speaker=self.speaker,
            do_sample=False,
            subtalker_dosample=False,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent, prefix=f".{output_path.stem}.", suffix=".wav"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            sf.write(temporary, waveforms[0], sample_rate, subtype="PCM_16")
            os.replace(temporary, output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def main():
    """Model-free artifact preflight, before expensive ASR/MT stages."""
    import argparse
    from ..inference_logging import configure_logging
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    print(json.dumps(dict(model=args.model, revision=args.revision,
                          snapshot=str(pinned_snapshot(args.model, args.revision)))), flush=True)


if __name__ == '__main__':
    main()
