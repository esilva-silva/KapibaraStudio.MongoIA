import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DISTILL = HERE.parent
sys.path.insert(0, str(DISTILL))
import _distill_common  # noqa: E402,F401  (acrescenta scripts/data, scripts/teacher e bakeoff ao sys.path)

# numpy/tokenizers moram so no venv ROCm: acrescenta no FIM do sys.path (o env/.venv nao os tem)
import os  # noqa: E402

site = os.environ.get("KAPI_TOKENIZERS_SITE")
if site and Path(site).exists() and site not in sys.path:
    sys.path.append(site)
