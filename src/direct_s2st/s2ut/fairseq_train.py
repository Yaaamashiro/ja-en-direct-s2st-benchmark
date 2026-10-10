"""Run the pinned fairseq trainer with loss/gradient evidence; no model changes."""
import os
import json
from ..progress import operation, track
from pathlib import Path
import sys
import torch
from ..io import atomic_write_json

GROUPS = ('encoder', 'decoder', 'source_letter_decoder', 'target_letter_decoder', 'decoder_target_ctc_decoder')


@operation('s2ut: validate and lock paper recipe')
def prepare_research_metadata(command):
    """Validate before fairseq loads optimizer state; wrapper flags never reach it."""
    from ..recipes import (option, validate_s2ut_paper, validate_s2ut_runtime, training_metadata,
                           prepared_metadata, MODES, S2UT_VOCODERS)
    from ..hashing import sha256_file
    from ..journal import digest
    command = list(command)
    mode, vocoder = option(command, '--reproduction-mode', 'smoke'), option(command, '--vocoder-mode', 'trained')
    for flag in ('--reproduction-mode', '--vocoder-mode'):
        if flag in command:
            index = command.index(flag)
            del command[index:index + 2]
    if mode not in MODES or vocoder not in S2UT_VOCODERS:
        raise ValueError('invalid S2UT recipe/vocoder mode')
    if mode == 'smoke':
        return command, None
    if option(command, '--arch') != 's2ut_transformer_fisher':
        raise ValueError('S2UT paper recipe requires Fisher architecture')
    validate_s2ut_paper(command, exact=mode == 'paper_exact')
    validate_s2ut_runtime(mode,
        fixed_microbatch=int(os.environ.get('S2ST_TRAIN_FIXED_MICROBATCH', '0')),
        adaptive_batch=os.environ.get('S2ST_TRAIN_ADAPTIVE_BATCH') == '1',
        precision=os.environ.get('S2ST_TRAIN_PRECISION', 'default'))
    if mode == 'paper_exact':
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('this Fisher update_freq=4 recipe simulates four GPUs on one process; use an explicit practical recipe for other world sizes')
    from .preflight import validate_training
    root = Path(command[1])
    evidence = validate_training(root, command)
    metadata = training_metadata(command, 's2ut', mode, vocoder, world_size=int(os.environ.get('WORLD_SIZE', '1')))
    if int(os.environ.get('S2ST_TRAIN_FIXED_MICROBATCH', '0')):
        metadata['physical_batch_cap'] = int(os.environ['S2ST_TRAIN_FIXED_MICROBATCH'])
        metadata['batch_policy'] = 'startup-calibrated then fixed; logical token batches unchanged'
        metadata['deviations'].append('additional physical microbatch splitting differs from official Fisher batch execution')
    metadata.update(prepared_metadata(root))
    lock = root.parent / 'units/unit-lock.json'
    unit_identity = json.loads(lock.read_text(encoding='utf-8'))
    from ..config import load_config
    repository = Path(__file__).resolve().parents[3]
    cfg = load_config(repository / 'configs/s2ut/prepare.yaml')
    expected_units = dict(hubert_revision=cfg['hubert']['revision'], kmeans_sha256=cfg['kmeans']['sha256'],
                          hubert_layer=cfg['hubert_layer'], kmeans_clusters=cfg['kmeans_clusters'])
    if any(unit_identity.get(k) != v for k, v in expected_units.items()) or metadata['dataset_fingerprint']['unit_configuration'] != unit_identity:
        raise ValueError('S2UT unit artifact identity mismatch')
    metadata.update(**expected_units, reduce_consecutive_units=True)
    metadata.update(unit_artifacts=unit_identity, unit_lock_sha256=sha256_file(lock),
                    model_dimensions=evidence['model_dimensions'])
    normalized = command[:]
    for flag in ('--restore-file', '--max-update'):
        if flag in normalized:
            index = normalized.index(flag)
            del normalized[index:index + 2]
    identity = dict(recipe=metadata, command=normalized,
                    data={str(p.relative_to(root)): sha256_file(p) for p in
                          track(sorted(root.rglob('*')), 's2ut: hash recipe inputs') if p.is_file()})
    metadata['training_fingerprint'] = digest(identity)
    run_root = Path(option(command, '--save-dir')).parent
    marker = run_root / 's2ut-recipe-lock.json'
    if '--restore-file' in command:
        saved = json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else None
        if saved != identity:
            raise ValueError('S2UT recipe identity mismatch or legacy checkpoint; use a new run')
    atomic_write_json(marker, identity, resume=True)
    import subprocess
    repository = Path(__file__).resolve().parents[3]
    metadata['repository_revision'] = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    metadata.update(max_updates=int(option(command, '--max-update')), completed_updates=0)
    metadata['paper_training_length_reached'] = metadata['max_updates'] >= 400000
    if not metadata['paper_training_length_reached']:
        print('[recipe] S2UT short training: target updates below paper 400000; not a completed paper-length run',
              file=sys.stderr, flush=True)
    return command, (run_root, metadata)


class TrainingAudit:
    def __init__(self):
        self.handles, self.models, self.losses = [], set(), []
        self.gradients = {name: 0.0 for name in GROUPS}
        self.loss_count = 0

    def attach(self, model):
        if id(model) in self.models:
            return
        self.models.add(id(model))
        for name, parameter in model.named_parameters():
            group = next((key for key in GROUPS if name.startswith(key+'.')), None)
            if group and parameter.requires_grad:
                def inspect(gradient, key=group):
                    value = gradient.detach().abs().max()
                    previous = self.gradients[key]
                    self.gradients[key] = (torch.maximum(previous, value) if torch.is_tensor(previous) else value)
                    return gradient
                self.handles.append(parameter.register_hook(inspect))

    def record(self, total, main, logs):
        tensors = [total.detach(), torch.as_tensor(main, device=total.device).detach()]
        names = ['total', 'main']
        for task in ('source_letter', 'target_letter', 'decoder_target_ctc'):
            if task not in logs.get('multitask', {}):
                raise ValueError('missing S2UT auxiliary loss: '+task)
            tensors.append(torch.as_tensor(logs['multitask'][task]['loss'], device=total.device).detach())
            names.append(task)
        values = dict(zip(names, torch.stack(tensors).tolist()))
        import math
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError('nonfinite S2UT loss')
        self.losses.append(values)
        self.loss_count += 1
        if len(self.losses) > 128:
            del self.losses[0]

    def result(self):
        device = next((v.device for v in self.gradients.values() if torch.is_tensor(v)), 'cpu')
        values = torch.stack([torch.as_tensor(v, device=device) for v in self.gradients.values()]).tolist()
        import math
        if any(not math.isfinite(v) for v in values):
            raise ValueError('nonfinite S2UT gradient')
        gradients = dict(zip(GROUPS, values))
        if not self.losses or any(value <= 0 for value in gradients.values()):
            raise ValueError('S2UT loss/gradient evidence is incomplete')
        return {'status': 'PASS', 'losses': self.losses, 'loss_count': self.loss_count,
                'loss_history_limit': 128, 'max_abs_gradient_by_group': gradients}


@operation('s2ut/fairseq_train: main')
def main():
    command, research = prepare_research_metadata(sys.argv)
    sys.argv[:] = command
    from fairseq.criterions.speech_to_speech_criterion import SpeechToUnitMultitaskTaskCriterion as Criterion
    from fairseq_cli.train import cli_main
    audit = TrainingAudit()
    original_forward, original_compute = Criterion.forward, Criterion.compute_loss
    last_main = []
    def compute(self, *args, **kwargs):
        output = original_compute(self, *args, **kwargs)
        last_main[:] = [output[0].detach()]
        return output
    def forward(self, model, *args, **kwargs):
        if model.training:
            audit.attach(model)
        output = original_forward(self, model, *args, **kwargs)
        if model.training:
            audit.record(output[0], last_main[0], output[2])
        return output
    Criterion.compute_loss, Criterion.forward = compute, forward
    from fairseq.trainer import Trainer
    original_save = Trainer.save_checkpoint
    def save(self, filename, extra_state):
        extra_state = dict(extra_state)
        if research:
            extra_state['s2st_recipe_fingerprint'] = research[1]['training_fingerprint']
            atomic_write_json(research[0] / 'research-metadata.json',
                              dict(research[1], completed_updates=self.get_num_updates()), overwrite=True)
        return original_save(self, filename, extra_state)
    if research:
        # Bind the actual restored file too, not only a sibling lock document.
        restore_file = command[command.index('--restore-file') + 1] if '--restore-file' in command else None
        if restore_file:
            checkpoint = torch.load(restore_file, map_location='cpu', weights_only=False)
            if checkpoint.get('extra_state', {}).get('s2st_recipe_fingerprint') != research[1]['training_fingerprint']:
                raise ValueError('S2UT checkpoint recipe fingerprint mismatch')
        Trainer.save_checkpoint = save
    try:
        from .performance import runtime_hooks
        with runtime_hooks(audit):
            cli_main()
        directory = Path(sys.argv[sys.argv.index('--save-dir')+1])
        rank = os.environ.get('RANK', '0')
        atomic_write_json(directory.parent / ('gradient-audit-rank-'+rank+'.json'), audit.result(), overwrite=True)
    finally:
        Trainer.save_checkpoint = original_save
        Criterion.compute_loss, Criterion.forward = original_compute, original_forward
        for handle in audit.handles:
            handle.remove()


if __name__ == '__main__':
    main()
