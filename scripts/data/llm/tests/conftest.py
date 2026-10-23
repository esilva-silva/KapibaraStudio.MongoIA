import sys
from pathlib import Path

LLM = Path(__file__).resolve().parents[1]
DATA = LLM.parent
for p in (str(LLM), str(DATA)):
    if p not in sys.path:
        sys.path.insert(0, p)
