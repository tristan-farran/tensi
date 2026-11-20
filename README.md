<p align="center">
  <img src="https://raw.githubusercontent.com/tristan-farran/tensi/main/brand/tensi-logo.svg" alt="tensi" width="440">
</p>

<p align="center">
  <b>tensi sees your tensors, and now you do too!</b><br>
  Hover any PyTorch tensor and see its shape, with _named dimensions_.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/tristan-farran/tensi/main/docs/demo.gif" alt="tensi showing named tensor shapes in nanoGPT">
</p>

## Install


<p align="center">
  <a href="https://marketplace.visualstudio.com/items?itemName=TristanFarran.tensi">VS Code</a> ·
  <a href="https://pypi.org/project/tensi-shapes/">PyPI</a>
</p>

**Command line and CI:** `uv add --dev tensi-shapes` (or `pip install tensi-shapes`).

```bash
tensi peek model.py   # every tensor's shape
tensi check src/      # shape errors; exits 1 if any
```

<sub>Built on Pyrefly :)</sub>
