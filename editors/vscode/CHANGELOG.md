# Changelog

## 0.3.1

* Prefers the project's own `.venv` over a global interpreter the Python extension happens to have
  selected (a Homebrew Python, say), which left tensi without torch.
* A `tensi.pythonPath` that no longer exists is reported once and skipped instead of breaking tensi.
* One **Install tensi** button installs into the project (`uv add --dev` in a uv project), and works on
  Python 3.10 projects.
* A status bar item shows when tensi is working out shapes and when it is ready; click it to check the
  setup.
* Quieter output: the environment is probed once per restart, not on every file change.

## 0.3.0

First Marketplace release.

* Follows each project's interpreter (the Python extension's choice, else `.venv`) and picks up
  `uv sync`, a new `.venv` or an interpreter change by itself.
* Works with tensi installed in the project (`uv add --dev tensi-shapes`) or once for all projects
  (`uv tool install tensi-shapes`), and offers either when it's missing.
* New command **tensi: Check setup** (`tensi doctor`), and a tensi output channel.
