"""Pinned official HuBERT-100 LJSpeech Code HiFi-GAN artifacts."""
import os
from pathlib import Path
from ..artifacts import download_artifact

BASE = 'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/'
CHECKPOINT_SHA256 = 'b46cc9778377a3c1b72b44ff815126469511676083d728f1c0677a1be0f6bc36'
CONFIG_SHA256 = '769cd70effba4278d2d7eef7d5c42177651b9273f0678b418e7d48e2665f6043'
UNIT_CONFIGURATION = dict(
    hubert_model='facebook/hubert-base-ls960',
    hubert_revision='dba3bb02fda4248b6e082697eee756de8fe8aa8a',
    hubert_layer=6, kmeans_clusters=100,
    kmeans_sha256='f14a3104615485381fc489701d6761c9abbdbb0d43607cf55518a1e2891023fe')


def validate_reference_units(configuration):
    # K=100 alone is insufficient: a different k-means model permutes the IDs.
    if any(configuration.get(key) != value for key, value in UNIT_CONFIGURATION.items()):
        raise ValueError('official Unit vocoder requires the pinned HuBERT L6 / KM100 unit configuration')


def ensure_reference(checkpoint, config):
    for path, name, checksum in [(checkpoint, 'g_00500000', CHECKPOINT_SHA256),
                                 (config, 'config.json', CONFIG_SHA256)]:
        path = Path(path)
        corpus = os.environ.get('CORPUS_ROOT')
        if corpus and path.resolve().is_relative_to(Path(corpus).resolve()):
            raise ValueError('reference vocoder must be outside CORPUS_ROOT')
        download_artifact(BASE + name, path, sha256=checksum)
