"""Generate the independent, CPU-only Baseline report import notebook."""
import json
from pathlib import Path


def build():
    cells = []
    def cell(kind, source):
        value = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        cells.append(value)
    cell('markdown', '''# Import measured S2UT Baseline metrics into W&B

Use a **CPU runtime**. This notebook reads existing reports and snapshot receipts;
it never loads a checkpoint/model or recalculates loss/BLEU. The four report files
and all Baseline artifacts stay read-only. Reporter state goes into a sibling
directory. The run is named `baseline` in the existing `alignment-transfer-s2ut`
project (or your explicitly selected existing project). Specify the same entity/team used by the transfer runs.

First inspect counts, then enable synchronization. Add a `WANDB_API_KEY` **Read &
Write** API key to Colab Secrets (grant notebook access); never paste it into a
cell or a command. Rerun the last cell after a disconnect/network failure.
''')
    cell('code', '''#@title 1. Import settings
SOURCE = '/content/drive/MyDrive/alignment-transfer-s2ut-data/experiment-001/baseline' #@param {type:"string"}
STATE = '/content/drive/MyDrive/alignment-transfer-s2ut-data/experiment-001/wandb-baseline' #@param {type:"string"}
WANDB_ENTITY = '' #@param {type:"string"}
WANDB_PROJECT = 'alignment-transfer-s2ut' #@param {type:"string"}
WANDB_GROUP = '' #@param {type:"string"}
WANDB_EXISTING_RUN_ID = '' #@param {type:"string"}
BASELINE_RUN_KEY = '' #@param {type:"string"}
# Empty attempts to discover the published registration for the standard source.
REGISTRATION_SNAPSHOT = '' #@param {type:"string"}
AUDIT_FAILURE_JOURNALS = False #@param {type:"boolean"}
SYNC_TO_WANDB = False #@param {type:"boolean"}
# Optional full commit SHA. Empty selects main ONCE per local checkout.
CODE_REVISION = '' #@param {type:"string"}
''')
    cell('code', '''#@title 2. Mount Drive and obtain reporter code (no training setup)
from google.colab import drive
drive.mount('/content/drive')
import json, os, re, subprocess, sys, threading, time
from pathlib import Path
CODE = Path('/content/s2st-baseline-wandb-code')
assert not CODE_REVISION or re.fullmatch('[0-9a-f]{40}', CODE_REVISION), 'Use an immutable full code SHA'

def command(args, *, env=None, capture=False):
    # Local logs only. Do not append to the original Baseline session log.
    logs = Path('/content/s2st-wandb-command-logs')
    logs.mkdir(exist_ok=True)
    label = str(time.time_ns())
    process = subprocess.Popen(list(map(str, args)), env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE if capture else subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace')
    (logs/(label+'.pid')).write_text(str(process.pid))
    errors = []
    def drain_errors():
        for line in process.stderr:
            errors.append(line)
            print(line, end='', flush=True)
    reader = threading.Thread(target=drain_errors, daemon=True) if capture else None
    if reader:
        reader.start()
    try:
        lines = []
        for line in process.stdout:
            lines.append(line)
            if not capture:
                print(line, end='', flush=True)
        process.wait()
        if reader:
            reader.join()
        out, err = ''.join(lines), ''.join(errors)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    (logs/(label+'.log')).write_text(out + (err or ''), encoding='utf-8')
    (logs/(label+'.status')).write_text(str(process.returncode))
    if process.returncode:
        raise RuntimeError('Reporter command failed; source preserved. Local log: '+str(logs/(label+'.log')))
    return out

if not CODE.exists():
    command(['git', 'clone', '--no-checkout', 'https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark.git', CODE])
    command(['git', '-C', CODE, 'fetch', 'origin', CODE_REVISION or 'main'])
    command(['git', '-C', CODE, 'checkout', '--detach', 'FETCH_HEAD'])
elif CODE_REVISION and command(['git', '-C', CODE, 'rev-parse', 'HEAD'], capture=True).strip() != CODE_REVISION:
    raise ValueError('Local reporter revision differs. Restart the CPU runtime to select another revision.')
print('Reporter revision:', command(['git', '-C', CODE, 'rev-parse', 'HEAD'], capture=True).strip())
HELPER = CODE/'scripts/colab/baseline_wandb.py'
assert HELPER.is_file(), 'Select a revision containing the Baseline W&B integration'

def evidence_args():
    args = []
    registration = REGISTRATION_SNAPSHOT.strip()
    if not registration:
        directory = Path(SOURCE).parent/'stages/register-baseline'
        candidates = list(directory.glob('*/snapshot.json'))
        if len(candidates) == 1:
            registration = str(candidates[0])
    if registration:
        args += ['--registration', registration]
    if AUDIT_FAILURE_JOURNALS:
        args.append('--audit-journals')
    return args
''')
    cell('code', '''#@title 3. Inspect measured, checkpoint-confirmed metrics (no W&B dependency)
report = json.loads(command([sys.executable, HELPER, 'inspect', '--source', SOURCE, *evidence_args()], capture=True))
print('Confirmed optimizer update:', report['durable_update'])
print('Verified training fingerprint:', report['training_fingerprint'])
print('Saved summary:', json.dumps(report['summary'], indent=2))
for metric, count in report['counts'].items():
    print(f'{metric}: {count}')
print('Source SHA-256:', json.dumps(report['sources'], indent=2))
''')
    cell('code', '''#@title 4. Synchronize / resume the same baseline run
if SYNC_TO_WANDB:
    assert WANDB_ENTITY.strip(), 'Set WANDB_ENTITY to the same account/team as the transfer runs'
    assert WANDB_PROJECT.strip(), 'Set the same existing project as the transfer runs'
    print('Destination:', WANDB_ENTITY.strip()+'/'+WANDB_PROJECT.strip(), 'group:', WANDB_GROUP.strip(), 'run: baseline')
    import runpy
    settings = runpy.run_path(str(CODE/'scripts/colab/wandb_runtime.py'))['prepare'](CODE)
    if settings is None:
        raise RuntimeError('Reporter dependency/auth setup failed. Source data were not modified.')
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['WANDB_API_KEY'] = settings['key']
    try:
        result = json.loads(command([settings['python'], HELPER, 'sync', '--source', SOURCE,
            '--state', STATE, '--entity', WANDB_ENTITY.strip(), '--project', WANDB_PROJECT.strip(),
            '--group', WANDB_GROUP.strip(), '--existing-run-id', WANDB_EXISTING_RUN_ID.strip(),
            '--baseline-run-key', BASELINE_RUN_KEY, *evidence_args()],
            env=env, capture=True))
        print(json.dumps(result, indent=2))
        print('W&B:', f"https://forge.coreweave.com/wandb/{result['entity']}/{result['project']}/runs/{result['run_id']}")
    finally:
        env.pop('WANDB_API_KEY', None)
        settings = None
else:
    print('Inspection only. Set SYNC_TO_WANDB=True in cell 1, then rerun this cell to upload.')
''')
    return dict(cells=cells, metadata=dict(colab=dict(name='baseline_wandb_import.ipynb'),
                kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                language_info=dict(name='python')), nbformat=4, nbformat_minor=0)


if __name__ == '__main__':
    destination = Path(__file__).resolve().parents[2]/'notebooks/baseline_wandb_import.ipynb'
    destination.write_text(json.dumps(build(), ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
