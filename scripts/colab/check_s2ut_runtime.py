"""Exercise real pinned CTC train/eval paths before corpus staging or optimization."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from direct_s2st.progress import operation


@operation('s2ut runtime: actual CTC train and dev forward')
def check():
    import editdistance
    import torch
    import torchaudio
    import sentencepiece
    import soundfile
    import librosa
    from fairseq.data import Dictionary
    from fairseq.models import FairseqEncoderModel
    from fairseq.criterions.ctc import CtcCriterion
    from fairseq.models.speech_to_speech.modules.ctc_decoder import CTCDecoder
    assert editdistance.eval([1, 2], [1, 3]) == 1
    assert editdistance.eval(['hello', 'world'], ['hello']) == 1
    dictionary = Dictionary()
    a, b = dictionary.add_symbol('▁hello'), dictionary.add_symbol('▁world')
    task = SimpleNamespace(target_dictionary=dictionary)
    cfg = SimpleNamespace(post_process='sentencepiece', sentence_avg=False, zero_infinity=False,
                          wer_args=None, wer_kenlm_model=None)
    criterion = CtcCriterion(cfg, task)
    with torch.random.fork_rng():
        torch.manual_seed(1)
        model = FairseqEncoderModel(CTCDecoder(dictionary, 4))
        sample = dict(id=torch.arange(2), target=torch.tensor([[a, b], [a, dictionary.pad()]]),
            target_lengths=torch.tensor([2, 1]), ntokens=3,
            net_input=dict(src_tokens=torch.randn(6, 2, 4), src_lengths=torch.tensor([6, 5])))
        loss, size, train = criterion(model.train(), sample)
        assert torch.isfinite(loss) and size == 3
        loss.backward()
        assert torch.isfinite(model.encoder.proj.weight.grad).all()
        with torch.no_grad():
            loss, size, dev = criterion(model.eval(), sample)
        assert torch.isfinite(loss) and size == 3 and dev['c_total'] == 3 and dev['w_total'] == 3
    print('[runtime-preflight] CTC train/dev=PASS validation_only=true optimizer_updates=0', flush=True)


if __name__ == '__main__':
    check()
