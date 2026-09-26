"""Scoped, idempotent source overlay on the pinned fairseq runtime COPY only."""
from pathlib import Path
import re


def replace(path, before, after):
    text = path.read_text(encoding='utf-8')
    if before in text:
        path.write_text(text.replace(before, after), encoding='utf-8')
    elif after not in text:
        raise ValueError(f'incompatible fairseq source: {path}: {before}')


def apply(root):
    root = Path(root).resolve()
    if (root / '.git').exists() or root.name != 'fairseq':
        raise ValueError('compatibility overlay requires an unversioned fairseq copy')
    replace(root/'setup.py', 'hydra-core>=1.0.7,<1.1', 'hydra-core==1.3.2')
    replace(root/'setup.py', 'omegaconf<2.1', 'omegaconf==2.3.0')
    # Python 3.11+ rejects unhashable dataclass instances as field defaults.
    for relative in ('fairseq/dataclass/configs.py', 'fairseq/models/transformer/transformer_config.py'):
        path = root / relative
        text = path.read_text(encoding='utf-8')
        text = re.sub(r'(: \w+Config = )(\w+Config)\(\)', r'\1field(default_factory=\2)', text)
        text = text.replace('field(default=QuantNoiseConfig())', 'field(default_factory=QuantNoiseConfig)')
        path.write_text(text, encoding='utf-8')
    replace(root/'fairseq/dataclass/initialize.py',
        'v = FairseqConfig.__dataclass_fields__[k].default',
        'v = getattr(FairseqConfig(), k)')
    for relative, before, after in (
        ('fairseq/data/data_utils.py', 'np.int,', 'int,'),
        ('fairseq/modules/dynamic_crf_layer.py', 'np.float("inf")', 'float("inf")'),
        ('fairseq/model_parallel/megatron/gpt2_data_loader.py', 'dtype=np.int)', 'dtype=int)'),
    ):
        replace(root/relative, before, after)


if __name__ == '__main__':
    import sys
    apply(sys.argv[1])
