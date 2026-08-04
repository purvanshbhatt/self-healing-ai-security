# Placeholder conftest.py. Pytest auto-discovers this file.
import os
import sys
from pathlib import Path

# Make `src/` importable as top-level package in test runs.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Use example .env if a real one is missing, so config can read OPENAI_API_KEY.
env_example = ROOT / ".env.example"
env_real = ROOT / ".env"
if not env_real.exists() and env_example.exists():
    for line in env_example.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ.setdefault("OPENAI_API_KEY", line.split("=", 1)[1])
