"""Offline compatibility gate, not a real-speech E2E/quality test."""
import argparse
import sys
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-extensions', action='store_true')
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 13), sys.version
    from fairseq import options
    from fairseq.dataclass.configs import FairseqConfig
    from fairseq.models.transformer.transformer_config import TransformerConfig
    from fairseq.models import ARCH_CONFIG_REGISTRY
    from fairseq.models.text_to_speech.codehifigan import CodeGenerator
    from fairseq.criterions.speech_to_speech_criterion import SpeechToUnitMultitaskTaskCriterion
    from fairseq.data.audio.audio_utils import get_mel_filters, TTSSpectrogram, TTSMelScale
    import examples.speech_synthesis.data_utils
    from omegaconf import OmegaConf
    a, b = FairseqConfig(), FairseqConfig()
    assert a.common is not b.common
    x, y = TransformerConfig(), TransformerConfig()
    assert x.encoder is not y.encoder and x.quant_noise is not y.quant_noise
    OmegaConf.structured(a)
    options.get_training_parser()
    assert 's2ut_transformer' in ARCH_CONFIG_REGISTRY
    assert get_mel_filters(16000, 512, 80, 0, 8000).shape == (80, 257)
    waveform = torch.linspace(-1, 1, 16000)[None]
    mel = TTSMelScale(80, 16000, 0, 8000, 257)(TTSSpectrogram(512, 400, 160)(waveform))
    assert torch.isfinite(mel).all()
    from direct_s2st.s2ut.extract_units import assign_kmeans_units
    centers = np.arange(400, dtype=np.float32).reshape(100, 4)
    assert assign_kmeans_units(centers[[0, 3, 99]], centers).tolist() == [0, 3, 99]
    if args.require_extensions:
        from fairseq.data import data_utils_fast
        result = data_utils_fast.batch_by_size_vec(np.arange(4, dtype=np.int64),
            np.ones(4, dtype=np.int64), 4, 4, 1)
        assert len(result) == 1
    print('Python 3.13 fairseq config/CLI/frontend compatibility: PASS')


if __name__ == '__main__':
    main()
