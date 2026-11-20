# Contributing

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]' ruff
tensi doctor            # should end with "ok ready"
pytest -q               # rewriter, mapping, solver, CLI and LSP acceptance tests (about 2 minutes)
ruff check tensi tests && ruff format --check tensi tests
```

The VS Code extension lives in `editors/vscode`:

```bash
cd editors/vscode
npm ci
npm run typecheck
npm run package         # builds tensi-<version>.vsix
code --install-extension tensi-*.vsix
```

## Layout

| Path | What it does |
| --- | --- |
| `tensi/rewrite.py` | turns a file into its *shadow*: the same code with the inferred dims spelled out in Pyrefly's shape types; edits never add lines |
| `tensi/seeds.py` | the solver: asks Pyrefly what each module call receives and feeds it into the callee, round by round |
| `tensi/mapping.py` | maps positions and names between your text and the shadow |
| `tensi/server.py` | the LSP proxy in front of `pyrefly lsp` |
| `tensi/stubs.py` | tensi's patch set over `pyrefly-torch-stubs` |
| `tensi/cli.py`, `peek.py`, `doctor.py` | the `tensi` command |
| `scripts/sweep.sh` | runs `peek` and `check` over a folder of model code and prints totals |

## Measuring a change

Coverage changes are judged on real code, not on one model. Clone a few model repositories (the README
lists the two corpora) and compare before and after:

```bash
scripts/sweep.sh ~/corpus before.txt
# make the change
scripts/sweep.sh ~/corpus after.txt
diff before.txt after.txt
```

A change should not lower any file's "fully named" count or add a `check` error that isn't a real bug.

## Releasing

See [RELEASING.md](RELEASING.md): the one-time setup for a first release, and the three steps for
every release after it.
