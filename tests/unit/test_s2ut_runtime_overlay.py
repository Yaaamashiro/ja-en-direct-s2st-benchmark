import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
from s2ut_runtime_overlay import (AUDIT, PERFORMANCE, RUNTIME, SESSION, CHECKPOINT, PREFLIGHT,
                                 audit_span, metadata_span, prepare, replacements)

PIN = '2486164f0dbc0851ea035feb783ce44ca9a3d1ea'


def baseline(name, revision=PIN):
    return subprocess.check_output(['git', '-c', 'safe.directory='+ROOT.as_posix(),
                                   '-C', str(ROOT), 'show', revision+':'+name])


def execution(tmp_path, revision=PIN):
    root = tmp_path/'execution'
    for name in (PERFORMANCE, AUDIT, RUNTIME, SESSION, CHECKPOINT, PREFLIGHT):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(baseline(name, revision))
    return root


@pytest.mark.parametrize('revision', [PIN, 'e56ca1c749a72476f07e8d496e5634b43e12d47c', '39b41c3'])
def test_repair_only_changes_reviewed_runtime_checkpoint_and_audit_scope(tmp_path, revision):
    root = execution(tmp_path, revision)
    changed = replacements(root, ROOT)
    assert changed.get(PERFORMANCE, baseline(PERFORMANCE, revision)) == (ROOT/PERFORMANCE).read_bytes()
    def without_audit(value):
        lines = value.decode().splitlines(keepends=True)
        for span in sorted((audit_span, metadata_span), key=lambda f: f(value)[0], reverse=True):
            a, b = span(value)
            lines[a:b] = ['<reviewed-audit-or-recipe-binding>\n']
        return ''.join(lines)
    repaired_audit = changed.get(AUDIT, baseline(AUDIT, revision))
    assert without_audit(repaired_audit) == without_audit(baseline(AUDIT, revision))
    def without_freeze(value):
        tree = ast.parse(value)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'checkpoint_saved')
        a, b = min([node.lineno]+[d.lineno for d in node.decorator_list])-1, node.end_lineno
        lines = value.decode().splitlines(keepends=True)
        lines[a:b] = ['<freeze-only>\n']
        return ''.join(lines)
    assert without_freeze(changed.get(RUNTIME, baseline(RUNTIME, revision))) == without_freeze(baseline(RUNTIME, revision))
    module = {'__name__': 'direct_s2st.s2ut.reviewed_fixture', '__package__': 'direct_s2st.s2ut'}
    exec(compile(repaired_audit, 'reviewed-audit', 'exec'), module)
    audit = module['TrainingAudit']()
    state = audit.attempt_state()
    audit.loss_count = 100
    audit.restore_attempt(state)
    assert audit.loss_count == 0
    assert 'check_audio' in module['prepare_research_metadata'].__wrapped__.__kwdefaults__
    for source in changed.values(): ast.parse(source)
    (root/PERFORMANCE).write_bytes(baseline(PERFORMANCE)+b'\n# unreviewed change\n')
    with pytest.raises(ValueError, match='reviewed pinned runtime'):
        replacements(root, ROOT)
    assert replacements(ROOT, ROOT) == {}


def test_overlay_resumes_and_preserves_originals_configs_and_other_source(tmp_path):
    root = execution(tmp_path)
    config, model = root/'configs/s2ut/train.yaml', root/'src/direct_s2st/s2ut/model-fixture.py'
    config.parent.mkdir(parents=True)
    config.write_text('original optimizer batch warmup FP16 recipe')
    model.write_text('unchanged model')
    overlay = prepare(root, ROOT, tmp_path/'persistent', 's2ut-run')
    assert (root/PERFORMANCE).read_bytes() == baseline(PERFORMANCE)
    assert (root/AUDIT).read_bytes() == baseline(AUDIT)
    assert (overlay/config.relative_to(root)).read_bytes() == config.read_bytes()
    assert (overlay/model.relative_to(root)).read_bytes() == model.read_bytes()
    assert prepare(root, ROOT, tmp_path/'persistent', 's2ut-run') == overlay
    assert len(list((tmp_path/'persistent/execution-overlays').glob('*.json'))) == 1
    (overlay/PERFORMANCE).write_text('unexpected modification')
    with pytest.raises(FileExistsError):
        prepare(root, ROOT, tmp_path/'persistent', 's2ut-run')
    assert (root/PERFORMANCE).read_bytes() == baseline(PERFORMANCE)
