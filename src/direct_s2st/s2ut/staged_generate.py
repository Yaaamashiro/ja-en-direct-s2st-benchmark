"""Full-split pinned fairseq generation with verified local WAV mapping."""


def main():
    from ..inference_logging import configure_logging
    configure_logging()
    from fairseq_cli import generate
    from fairseq.data.audio import speech_to_text_dataset as audio
    from ..drive_staging import local_path
    original = audio.get_features_or_waveform
    def staged(path, *args, **kwargs):
        return original(str(local_path(path)), *args, **kwargs)
    try:
        audio.get_features_or_waveform = staged
        generate.cli_main()
    finally:
        audio.get_features_or_waveform = original


if __name__ == '__main__':
    main()
