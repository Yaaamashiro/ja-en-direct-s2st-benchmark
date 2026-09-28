"""Run the pinned fairseq trainer with loss/gradient evidence; no model changes."""
import os
from ..progress import operation
from pathlib import Path
import sys
import torch
from ..io import atomic_write_json

GROUPS = ('encoder', 'decoder', 'source_letter_decoder', 'target_letter_decoder', 'decoder_target_ctc_decoder')


class TrainingAudit:
    def __init__(self):
        self.handles, self.models, self.losses = [], set(), []
        self.gradients = {name: 0.0 for name in GROUPS}

    def attach(self, model):
        if id(model) in self.models:
            return
        self.models.add(id(model))
        for name, parameter in model.named_parameters():
            group = next((key for key in GROUPS if name.startswith(key+'.')), None)
            if group and parameter.requires_grad:
                def inspect(gradient, key=group):
                    if not torch.isfinite(gradient).all():
                        raise ValueError('nonfinite S2UT gradient: '+key)
                    self.gradients[key] = max(self.gradients[key], float(gradient.detach().abs().max()))
                    return gradient
                self.handles.append(parameter.register_hook(inspect))

    def record(self, total, main, logs):
        values = {'total': float(total.detach()), 'main': float(main)}
        for task in ('source_letter', 'target_letter', 'decoder_target_ctc'):
            if task not in logs.get('multitask', {}):
                raise ValueError('missing S2UT auxiliary loss: '+task)
            values[task] = float(logs['multitask'][task]['loss'])
        if any(not torch.isfinite(torch.tensor(v)) for v in values.values()):
            raise ValueError('nonfinite S2UT loss')
        self.losses.append(values)

    def result(self):
        if not self.losses or any(value <= 0 for value in self.gradients.values()):
            raise ValueError('S2UT loss/gradient evidence is incomplete')
        return {'status': 'PASS', 'losses': self.losses, 'max_abs_gradient_by_group': self.gradients}


@operation('s2ut/fairseq_train: main')
def main():
    from fairseq.criterions.speech_to_speech_criterion import SpeechToUnitMultitaskTaskCriterion as Criterion
    from fairseq_cli.train import cli_main
    audit = TrainingAudit()
    original_forward, original_compute = Criterion.forward, Criterion.compute_loss
    last_main = []
    def compute(self, *args, **kwargs):
        output = original_compute(self, *args, **kwargs)
        last_main[:] = [float(output[0].detach())]
        return output
    def forward(self, model, *args, **kwargs):
        if model.training:
            audit.attach(model)
        output = original_forward(self, model, *args, **kwargs)
        if model.training:
            audit.record(output[0], last_main[0], output[2])
        return output
    Criterion.compute_loss, Criterion.forward = compute, forward
    try:
        from .performance import runtime_hooks
        with runtime_hooks(audit):
            cli_main()
        directory = Path(sys.argv[sys.argv.index('--save-dir')+1])
        rank = os.environ.get('RANK', '0')
        atomic_write_json(directory.parent / ('gradient-audit-rank-'+rank+'.json'), audit.result(), overwrite=True)
    finally:
        Criterion.compute_loss, Criterion.forward = original_compute, original_forward
        for handle in audit.handles:
            handle.remove()


if __name__ == '__main__':
    main()
