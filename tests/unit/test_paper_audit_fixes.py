import copy
import json
from pathlib import Path

import pytest
import torch

from direct_s2st.s2ut.multitask import prepare_labels, read_tsv
from direct_s2st.s2ut.ctc_tokenizer import build, canonical_text, load


def rows():
    texts = ['Police organization and the police department', 'The police organization is here']
    return {split: [dict(pair_id=f'{split}-{i}', ja_text='日本語', en_text=text)
                   for i, text in enumerate(texts)] for split in ('train', 'dev', 'test')}


def test_unigram_is_train_only_independent_and_resume_immutable(tmp_path):
    data = rows()
    lock = prepare_labels(data, tmp_path/'prepared')
    processor, tokenizer = load(tmp_path/'prepared')
    assert tokenizer['requested_vocab_size'] == 1000
    assert tokenizer['model_type'] == 'unigram' and tokenizer['vocabulary_source'] == 'train_only'
    assert lock['ctc_version'] == tokenizer['version']
    letters = read_tsv(tmp_path/'prepared/target_letter/train.tsv', ('id', 'tgt_text'))
    ctc = read_tsv(tmp_path/'prepared/decoder_target_ctc/train.tsv', ('id', 'tgt_text'))
    assert letters != ctc
    assert ctc['train-0']['tgt_text'].split() == processor.encode(canonical_text(data['train'][0]['en_text']), out_type=str)
    original = (tmp_path/'prepared/ctc.model').read_bytes()
    assert prepare_labels(data, tmp_path/'prepared', resume=True) == lock
    assert (tmp_path/'prepared/ctc.model').read_bytes() == original
    altered = copy.deepcopy(data['train'])
    altered[0]['en_text'] += ' changed'
    with pytest.raises(ValueError, match='identity changed'):
        build(altered, tmp_path/'prepared')
    changed = copy.deepcopy(data)
    changed['dev'][0]['en_text'] = data['train'][1]['en_text']
    prepare_labels(changed, tmp_path/'other')
    assert (tmp_path/'other/ctc.model').read_bytes() == original  # No held-out training leakage.
    (tmp_path/'prepared/ctc.model').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        load(tmp_path/'prepared')


@pytest.mark.parametrize('legacy_ctc', [True, False])
def test_migration_reuses_tsvs_preserves_old_data_and_resumes(tmp_path, monkeypatch, legacy_ctc):
    from test_s2ut_units import _common
    from direct_s2st.s2ut.extract_units import extract_units
    from direct_s2st.s2ut.prepare_fairseq import prepare_fairseq
    from direct_s2st.s2ut.migrate_labels import migrate, paper_data_root
    from direct_s2st.io import atomic_write_json
    common, units, old = tmp_path/'common', tmp_path/'s2ut/units', tmp_path/'s2ut/fairseq'
    _common(common)
    extract_units(common, units, extractor=lambda _: [1, 2, 3], split=None, clusters=100,
                  hubert_model='fixture', hubert_revision='a'*40, hubert_layer=6, kmeans_sha256='b'*64)
    prepare_fairseq(common, units, old)
    lock = json.loads((old/'data-lock.json').read_text())
    del lock['linguistic']['character_vocab_version']
    lock['linguistic']['vocabulary_source'] = 'train_only'
    if legacy_ctc:
        for key in ('ctc_version', 'ctc_tokenizer'):
            del lock['linguistic'][key]
        # True legacy fixtures have character CTC labels and no SentencePiece model.
        for name in ('train.tsv', 'dev.tsv', 'test.tsv', 'dict.txt'):
            (old/'decoder_target_ctc'/name).write_bytes((old/'target_letter'/name).read_bytes())
        (old/'ctc.model').unlink()
        (old/'ctc-tokenizer.json').unlink()
    atomic_write_json(old/'data-lock.json', lock, overwrite=True)
    before = {str(p): p.read_bytes() for root in (old, units, common) for p in root.rglob('*') if p.is_file()}
    assert paper_data_root(tmp_path, planned=True).name == 'fairseq-unigram-allchars-v1'
    target = migrate(common, old)
    assert target != old and paper_data_root(tmp_path) == target
    assert (target/'train.tsv').read_bytes() == (old/'train.tsv').read_bytes()
    if not legacy_ctc:
        assert (target/'ctc.model').read_bytes() == (old/'ctc.model').read_bytes()
    assert all(Path(p).read_bytes() == data for p, data in before.items())
    assert migrate(common, old) == target
    (target/'decoder_target_ctc/test.tsv').write_text('corrupted')
    with pytest.raises(ValueError, match='checksum'):
        migrate(common, old)


def test_throughput_selection_does_not_assume_full_vram_is_fastest(tmp_path):
    from test_batch_calibration import config, measured, HW
    from direct_s2st.batch_calibration import calibrate
    from direct_s2st.recipes import option
    cfg = config()
    cfg['batch_calibration']['objective'] = 'throughput'
    def measure(c, size, parent):
        result = measured(size)
        # Batch 8 has best speed; larger batches fit but run slower per sample.
        result['steps'] = [dict(s, seconds=size/(100 if size == 8 else 70)) for s in result['steps']]
        result['steps'][0]['seconds'] = 999  # First-step warmup is not scored.
        return result
    selected = calibrate(cfg, {}, tmp_path/'record', tmp_path/'local', measure=measure, get_hardware=lambda: HW)
    assert option(selected['command'], '--batch-size') == '8'
    assert option(selected['command'], '--update-freq') == '128'
    record = json.loads((tmp_path/'record').read_text())
    assert record['selection_objective'] == 'throughput'
    assert record['stress_samples_per_second']['8'] == pytest.approx(100)


def test_teacher_prenet_vectorization_keeps_eval_outputs_and_gradients():
    from direct_s2st.translatotron2.model import ModelConfig, Translatotron2
    model = Translatotron2(ModelConfig.smoke(), 7).eval()
    values = torch.randn(2, 8, model.config.mel_dim, requires_grad=True)
    vectorized = model.prenet(values)
    sequential = torch.stack([model.prenet(values[:, t]) for t in range(values.size(1))], 1)
    torch.testing.assert_close(vectorized, sequential)
    left = torch.autograd.grad(vectorized.sum(), values, retain_graph=True)[0]
    right = torch.autograd.grad(sequential.sum(), values)[0]
    torch.testing.assert_close(left, right)


def test_gpu_telemetry_unavailable_is_not_zero(monkeypatch):
    from direct_s2st.train_runtime import gpu_telemetry
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert gpu_telemetry() is None


def test_gradient_audit_is_bounded_and_rejects_nonfinite():
    from direct_s2st.s2ut.fairseq_train import TrainingAudit, GROUPS
    audit = TrainingAudit()
    logs = {'multitask': {name: {'loss': 1.} for name in ('source_letter', 'target_letter', 'decoder_target_ctc')}}
    for _ in range(200):
        audit.record(torch.tensor(1.), 1., logs)
    assert len(audit.losses) == 128 and audit.loss_count == 200
    audit.gradients = {name: torch.tensor(1.) for name in GROUPS}
    assert audit.result()['status'] == 'PASS'
    audit.gradients['encoder'] = torch.tensor(float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        audit.result()


def test_hubert_early_layer_pruning_preserves_features_without_downloads():
    from transformers import HubertConfig, HubertModel
    from direct_s2st.s2ut.extract_units import retain_hubert_layers
    torch.manual_seed(2)
    cfg = HubertConfig(hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
        intermediate_size=24, conv_dim=(8, 8, 8), conv_stride=(5, 2, 2), conv_kernel=(10, 3, 3),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, hidden_dropout=0.,
        attention_dropout=0., activation_dropout=0., feat_proj_dropout=0., layerdrop=0.)
    full = HubertModel(cfg).eval()
    trimmed = retain_hubert_layers(copy.deepcopy(full), 2).eval()
    wave = torch.randn(2, 400)
    with torch.inference_mode():
        expected = full(wave, output_hidden_states=True).hidden_states[2]
        actual = trimmed(wave, output_hidden_states=True).hidden_states[2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(trimmed.encoder.layers) == 2


@pytest.mark.gpu
def test_hubert_pruning_cuda_parity_and_tt2_cuda_resume(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from transformers import HubertConfig, HubertModel
    from direct_s2st.s2ut.extract_units import retain_hubert_layers
    torch.manual_seed(3)
    cfg = HubertConfig(hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
        intermediate_size=24, conv_dim=(8, 8, 8), conv_stride=(5, 2, 2), conv_kernel=(10, 3, 3),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, layerdrop=0.)
    full = HubertModel(cfg).eval().cuda()
    trimmed = retain_hubert_layers(copy.deepcopy(full), 2)
    wave = torch.randn(2, 400, device='cuda')
    with torch.inference_mode():
        torch.testing.assert_close(trimmed(wave, output_hidden_states=True).hidden_states[2],
                                   full(wave, output_hidden_states=True).hidden_states[2], rtol=0, atol=0)
    from direct_s2st.translatotron2.model import ModelConfig, Translatotron2
    from direct_s2st.translatotron2.engine import optimization_step, save_checkpoint, load_checkpoint
    model = Translatotron2(ModelConfig.smoke(), 7).cuda()
    batch = dict(source=torch.randn(2, 24, 80), source_lengths=torch.tensor([24, 17]),
        phones=torch.tensor([[3, 4, 5, 2], [4, 3, 2, 0]]), phone_lengths=torch.tensor([4, 3]),
        target=torch.randn(2, 12, 80), target_lengths=torch.tensor([12, 9]))
    optimizer = torch.optim.Adam(model.parameters())
    optimization_step(model, optimizer, batch)
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, model, optimizer, 1, list('abcdefg'), {'fixture': True})
    expected = optimization_step(model, optimizer, batch)
    restored, state = load_checkpoint(path, device='cuda', restore_rng=True)
    other = torch.optim.Adam(restored.parameters())
    other.load_state_dict(state['optimizer'])
    assert optimization_step(restored, other, batch) == expected
