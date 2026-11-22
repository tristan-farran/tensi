import os

# Tests never ask PyPI for the latest tensi; test_update.py exercises the check with a stub.
os.environ.setdefault("TENSI_NO_UPDATE_CHECK", "1")
