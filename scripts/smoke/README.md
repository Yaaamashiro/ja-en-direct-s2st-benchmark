`run.py` plans a Docker S2UT smoke by default. Set `--env-file` and optionally
`--docker-context`; add `--execute` only on the chosen execution environment.
It accepts 1..100 pairs per split and 1..10 training updates (defaults: 5 and 2).
It does not download corpus speech or create fake checkpoints. Read
`docs/REPRODUCTION.md` for image/artifact prerequisites and the result contract.

`suite.py` is the three-system driver (23 stages when fitting both vocoders):

```sh
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --execute
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --execute --resume
```

Planning is default. --vocoder-mode pretrained uses the configured checkpoints
instead of fitting. --device controls direct models/vocoders; Cascade/evaluation
retain their configured accelerator behavior. Read docs/PIPELINE_COMPLETION.md
for environment, resume and scientific limits. Two updates do not guarantee EOS
or intelligible speech; acceptance must pass without forced predictions.
