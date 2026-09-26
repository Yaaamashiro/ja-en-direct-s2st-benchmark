from pathlib import Path
import re
from ..cascade.whisper import load_whisper_pipeline


def validate_s2t_config(config):
    values = config['s2t']
    if values.get('task') != 'translate' or not values.get('language'):
        raise ValueError('S2T requires translate task and an explicit source language')
    for name in ('s2t', 'tts'):
        if not re.fullmatch(r'[0-9a-f]{40}', config[name].get('revision', '')):
            raise ValueError(f'{name} requires an immutable commit SHA')


class WhisperS2T:
    def __init__(self, *, model_id, revision, language, task, device=0):
        if task != 'translate':
            raise ValueError('WhisperS2T requires task=translate')
        self.language, self.task = language, task
        self.pipeline = load_whisper_pipeline(model_id, revision, device)

    def __call__(self, audio_path: Path) -> str:
        result = self.pipeline(str(audio_path), generate_kwargs=dict(
            language=self.language, task=self.task, do_sample=False, condition_on_prev_tokens=False))
        text = str(result['text']).strip()
        if not text:
            raise ValueError('empty speech translation')
        return text
