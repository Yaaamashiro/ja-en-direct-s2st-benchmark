"""Teacher-forced dev objectives only; never substitutes for free-running E2E."""
import torch
from ..progress import track
from .batching import collate


@torch.no_grad()
def validate(model, dataset, device, limit=None):
    training = model.training
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    totals = dict(mel_loss=0., phone_loss=0., duration_loss=0.)
    counts = [0, 0, 0]
    try:
        model.eval()
        for index in track(range(min(len(dataset), limit) if limit else len(dataset)), 'tt2: dev validation'):
            sample = {key: value.to(device) for key, value in collate([dataset[index]]).items()}
            output = model(**sample)
            sizes = [int(sample['target_lengths'].sum()), int(sample['phone_lengths'].sum()), sample['source'].size(0)]
            for position, name in enumerate(totals):
                if not torch.isfinite(output[name]):
                    raise ValueError('nonfinite dev loss: '+name)
                totals[name] += float(output[name])*sizes[position]
                counts[position] += sizes[position]
        if not all(counts):
            raise ValueError('empty dev dataset')
        result = {name: total/count for (name, total), count in zip(totals.items(), counts)}
        result['loss'] = result['mel_loss'] + model.config.phone_weight*result['phone_loss'] + model.config.duration_weight*result['duration_loss']
        return result
    finally:
        model.train(training)
        torch.set_rng_state(rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
