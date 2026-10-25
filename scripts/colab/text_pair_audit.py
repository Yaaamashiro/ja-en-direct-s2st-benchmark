"""Cell 8D bootstrap. No training setup, model dependencies, Hugging Face or audio."""
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

TEXT_AUDIT_BACKEND_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]


def require_widgets():
    # Standard Colab kernels supply these. Check before any Drive output writes;
    # never install an unreviewed dependency closure into a training environment.
    try:
        import ipywidgets
        import IPython.display
        for name in ('Button', 'Textarea', 'Accordion', 'Checkbox', 'Dropdown', 'Output', 'VBox', 'HBox', 'HTML', 'Layout'):
            if not hasattr(ipywidgets, name):
                raise ImportError('incomplete widget API')
        if int(ipywidgets.__version__.split('.')[0]) not in (7, 8):
            raise ImportError('ipywidgets 7/8 required')
    except ImportError as error:
        raise RuntimeError('8D requires ipywidgets 7/8 and IPython in the notebook kernel. '
                           'Use a standard Colab Python runtime (no Cell 3 needed). '
                           'No packages or experiment files were changed.') from error


def load_backend(root=ROOT):
    # A separate package namespace avoids reusing cached modules from a different
    # training revision. Do not clear/import the user's direct_s2st model modules.
    root = Path(root).resolve()
    name = '_s2st_8d_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root/'src/direct_s2st/__init__.py',
                                                     submodule_search_locations=[str(root/'src/direct_s2st')])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name+'.diagnostics.text_pair_audit'), importlib.import_module(name+'.diagnostics.text_pair_audit_ui')


def launch(experiment, *, corpus_root=None, source_manifest=None, per_stratum=100, seed=20261025):
    require_widgets()
    sys.dont_write_bytecode = True
    try:
        from google.colab import output
        output.enable_custom_widget_manager()
    except ImportError:
        pass  # Regular Jupyter and local CPU widget tests need no Colab manager.
    core, interface = load_backend()
    store = core.prepare(experiment, corpus_root=corpus_root, source_manifest=source_manifest,
                         per_stratum=per_stratum, seed=seed)
    audit = core.read(store.root/'sampling-audit.json')
    print('=== 8D: KFTT/JESC blinded text-pair audit ===\n\nDiagnostic only.\n'
          'No TTS / ASR / MT / GPU inference.\n'
          'Corpus identity is hidden until annotation is complete.\n'
          'Do not inspect the internal identity/queue/sampling files before finishing.\n\nPopulation:')
    for group in audit['strata']:
        print('{population}/{corpus}: requested={requested} available={available} selected={selected}'.format(**group))
    print('Excluded for missing text/provenance:', len(audit['excluded']), '(reasons saved internally)')
    print(f"\nTotal blinded pairs: {len(store.queue)}\nAlready annotated: {len(store.annotations)}\n"
          f"Remaining: {len(store.queue)-len(store.annotations)}\n\nSaved:\n{store.root}")
    return interface.show(store)
