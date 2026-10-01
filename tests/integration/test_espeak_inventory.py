"""Optional real pinned-engine audit: S2ST_TEST_ESPEAK=/path/to/espeak-ng."""
import os

import pytest

from direct_s2st.translatotron2.phonemize import EspeakNgPhonemizer


def test_pinned_engine_definition_inventory_and_lexical_text():
    executable = os.environ.get('S2ST_TEST_ESPEAK')
    if not executable:
        pytest.skip('explicit real eSpeak executable not configured')
    engine = EspeakNgPhonemizer(executable=executable, expected_version='1.52.0')
    vocabulary = set(engine.fixed_vocabulary())
    for text in (
        "Police organization ⋯ prefecture's police department",
        "don't 3.14 U.S.A. dogs' well-known",
        'science hour fire player cure loch genre rouge café déjà naïve',
        'zero one two three four five six seven eight nine 1,000 12:30 -5',
    ):
        phones = engine(text).split()
        assert phones and set(phones) <= vocabulary, (text, set(phones) - vocabulary)
    assert {'a\u200dɪ\u200də', 'a\u200dɪ\u200dɚ', 'o', 'r', 'ɐ', 'ɑ̃', 'ɔ'} <= vocabulary
    legacy = EspeakNgPhonemizer(executable=executable, expected_version='1.52.0',
                                text_processing_version='legacy-v1')
    assert engine("don't 3.14") != legacy("don't 3.14")
