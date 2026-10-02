"""Run the ablation of ablation.py with only the configs all_off and +memory_opt, each at every --dtypes value. Every other flag and the output format are ablation.py's; --configs and --out get their defaults here when not passed.
"""

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation  # noqa: E402

ablation.ENDS = ("all_off", "+memory_opt")


def passed(flag):
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


if not passed("--configs"):
    sys.argv += ["--configs", *ablation.ENDS]
if not passed("--out"):
    sys.argv += ["--out", f"zoo/ablation/{datetime.date.today():%Y-%m-%d}-ablation-memory-results.md"]

ablation.main()
