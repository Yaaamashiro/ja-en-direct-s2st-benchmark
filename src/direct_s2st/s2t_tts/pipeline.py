from ..cascade.pipeline import _run_text_pipeline
from ..cascade.tts import QwenTTS
from .s2t import WhisperS2T, validate_s2t_config


def run_pipeline(common_manifest, output_root, *, run_id, s2t, tts, **kwargs):
    return _run_text_pipeline(common_manifest, output_root, run_id=run_id, tts=tts,
        text_stages=[('s2t_en_text', 's2t_seconds', s2t)], system_id='s2t_tts', strict=True, **kwargs)


def components_from_config(config):
    validate_s2t_config(config)
    speech, voice = config['s2t'], config['tts']
    return (WhisperS2T(model_id=speech['model'], revision=speech['revision'],
                language=speech['language'], task=speech['task']),
            QwenTTS(model_id=voice['model'], revision=voice['revision'],
                speaker=voice['speaker'], language=voice['language']))
