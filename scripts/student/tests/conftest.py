import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDENT = HERE.parent
ROOT = STUDENT.parents[1]
for p in (ROOT / "scripts" / "distill", STUDENT):
    sys.path.insert(0, str(p))
import _distill_common  # noqa: E402,F401  (acrescenta scripts/data, scripts/teacher e bakeoff ao sys.path)

# numpy/tokenizers/yaml moram no venv ROCm: acrescenta no FIM do sys.path (o env/.venv do pytest nao os tem)
site = os.environ.get("KAPI_TOKENIZERS_SITE")
if site and Path(site).exists() and site not in sys.path:
    sys.path.append(site)
