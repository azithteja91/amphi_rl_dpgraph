import sys
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    cur = Path.cwd().resolve()
    ROOT = cur
    for candidate in [cur, *cur.parents]:
        if (candidate / "src").exists():
            ROOT = candidate
            break

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
