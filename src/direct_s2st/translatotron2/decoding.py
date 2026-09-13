"""Beam search over linguistic hypotheses; synthesizes the selected hidden states."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def beam_generate(model, source, source_lengths, beam_size, max_phones, max_frames, length_penalty):
    memory, lengths = model.encode(source, source_lengths)
    outputs = []
    for index in range(source.size(0)):
        acoustic = memory[index:index+1]
        acoustic_lengths = lengths[index:index+1]
        initial = dict(score=0., tokens=[], states=[], recurrent=None,
                       context=acoustic.new_zeros(1, model.config.context_dim), done=False)
        beams = [initial]
        def rank(hypothesis):
            return hypothesis['score'] / max(1, len(hypothesis['tokens']))**length_penalty
        for step in range(max_phones+1):
            candidates = []
            for hypothesis in beams:
                if hypothesis['done']:
                    candidates.append(hypothesis)
                    continue
                token = hypothesis['tokens'][-1] if hypothesis['tokens'] else model.BOS
                scores, hidden, context, recurrent, _ = model.linguistic_step(
                    torch.tensor([token], device=source.device), hypothesis['context'],
                    hypothesis['recurrent'], acoustic, acoustic_lengths)
                if not torch.isfinite(scores).all():
                    raise ValueError('nonfinite phoneme scores')
                scores = scores.log_softmax(-1)[0]
                scores[[model.PAD, model.BOS]] = -torch.inf
                if step == 0:
                    scores[model.EOS] = -torch.inf
                values, tokens = scores.topk(min(beam_size, scores.numel()))
                for value, token in zip(values.tolist(), tokens.tolist()):
                    if value == -float('inf'):
                        continue
                    done = token == model.EOS
                    candidates.append(dict(score=hypothesis['score']+value,
                        tokens=hypothesis['tokens']+[token], done=done,
                        states=hypothesis['states']+([] if done else [hidden]),
                        recurrent=recurrent, context=context))
            beams = sorted(candidates, key=rank, reverse=True)[:beam_size]
            if beams and all(hypothesis['done'] for hypothesis in beams):
                break
        completed = [hypothesis for hypothesis in beams if hypothesis['done']]
        if not completed:
            raise ValueError('phoneme decoding reached max_phones without EOS')
        chosen = max(completed, key=rank)
        phone_lengths = torch.tensor([len(chosen['states'])], device=source.device)
        output = model.synthesize(torch.stack(chosen['states'], 1), phone_lengths, max_frames=max_frames)
        output.update(phones=torch.tensor([chosen['tokens']], device=source.device), phone_lengths=phone_lengths)
        outputs.append(output)
    result = {}
    for key in ('mel', 'post_mel', 'durations', 'phones'):
        result[key] = torch.nn.utils.rnn.pad_sequence([output[key][0] for output in outputs], batch_first=True)
    for key in ('phone_lengths', 'frame_lengths'):
        result[key] = torch.cat([output[key] for output in outputs])
    frames, phones = int(result['frame_lengths'].max()), int(result['phone_lengths'].max())
    result['upsampling'] = torch.cat([F.pad(output['upsampling'],
        (0, phones-output['upsampling'].size(2), 0, frames-output['upsampling'].size(1))) for output in outputs])
    return result
