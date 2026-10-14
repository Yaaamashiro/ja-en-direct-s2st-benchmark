"""Execute pinned CTC source with real torch/editdistance, without old Hydra imports."""
import ast
import __future__
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
import editdistance
import torch
import torch.nn.functional as F


def load_class(path, name, namespace, methods=None):
    node = next(n for n in ast.parse((ROOT/path).read_text(encoding='utf-8')).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    node.decorator_list = []
    if methods:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def probe():
    assert editdistance.eval([1, 2], [1, 3]) == 1
    assert editdistance.eval(['hello', 'world'], ['hello']) == 1
    class Dictionary:
        symbols = ['<s>', '<pad>', '</s>', '<unk>', '▁hello', '▁world']
        def pad(self): return 1
        def eos(self): return 2
        def __len__(self): return len(self.symbols)
        def string(self, values): return ' '.join(self.symbols[int(v)] for v in values)
    dictionary = Dictionary()
    namespace = dict(torch=torch, nn=torch.nn, F=F, math=math, Namespace=NS,
                     utils=NS(item=lambda v: v.item()), _Loss=torch.nn.Module,
                     BaseFairseqModel=torch.nn.Module,
                     check_type=lambda obj, cls: isinstance(obj, cls))
    load_class('third_party/fairseq/fairseq/criterions/fairseq_criterion.py', 'FairseqCriterion', namespace, {'__init__'})
    load_class('third_party/fairseq/fairseq/models/fairseq_encoder.py', 'FairseqEncoder', namespace, {'__init__'})
    decoder = load_class('third_party/fairseq/fairseq/models/speech_to_speech/modules/ctc_decoder.py', 'CTCDecoder', namespace)
    wrapper = load_class('third_party/fairseq/fairseq/models/fairseq_model.py', 'FairseqEncoderModel', namespace)
    source = ROOT/'third_party/fairseq/fairseq/data/data_utils.py'
    post = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'post_process')
    exec(compile(ast.Module(body=[post], type_ignores=[]), str(source), 'exec'), namespace)
    criterion_cls = load_class('third_party/fairseq/fairseq/criterions/ctc.py', 'CtcCriterion', namespace)
    cfg = NS(post_process='sentencepiece', sentence_avg=False, zero_infinity=False,
             wer_args=None, wer_kenlm_model=None)
    criterion = criterion_cls(cfg, NS(target_dictionary=dictionary))
    torch.manual_seed(1)
    model = wrapper(decoder(dictionary, 4))
    batch = dict(id=torch.arange(2), target=torch.tensor([[4, 5], [4, 1]]),
                 target_lengths=torch.tensor([2, 1]), ntokens=3,
                 net_input=dict(src_tokens=torch.randn(6, 2, 4), src_lengths=torch.tensor([6, 5])))
    loss, size, train = criterion(model.train(), batch)
    assert torch.isfinite(loss) and size == 3 and 'c_errors' not in train
    loss.backward()
    assert torch.isfinite(model.encoder.proj.weight.grad).all()
    with torch.no_grad():
        loss, size, dev = criterion(model.eval(), batch)
    assert torch.isfinite(loss) and size == 3 and dev['c_total'] == dev['w_total'] == 3
    print('PASS actual pinned CTC train/backward/dev with native editdistance: '+str(editdistance.__file__))


if __name__ == '__main__':
    probe()
