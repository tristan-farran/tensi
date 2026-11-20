# Changelog

## 0.3.0

First Marketplace release.

* Follows each project's interpreter (the Python extension's choice, else `.venv`) and picks up
  `uv sync`, a new `.venv` or an interpreter change by itself.
* Works with tensi installed in the project (`uv add --dev tensi-shapes`) or once for all projects
  (`uv tool install tensi-shapes`), and offers either when it's missing.
* New command **tensi: Check setup** (`tensi doctor`), and a tensi output channel.
