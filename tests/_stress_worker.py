"""Worker for test_store.py's multi-process stress test (not collected by pytest).

usage: python _stress_worker.py DATA_DIR TAG N
Creates N decisions and fully scores each one (5 writes per decision).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_decision_lab.core import DecisionLab  # noqa: E402

data_dir, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
lab = DecisionLab(data_dir, lock_timeout=60)
for i in range(n):
    did = lab.start_decision(
        f"{tag} decision {i}",
        ["A", "B"],
        [{"name": "x", "weight": 1}, {"name": "y", "weight": 1}],
    )["decision_id"]
    for opt in ("A", "B"):
        for crit in ("x", "y"):
            lab.score_option(did, opt, crit, 5, f"{tag} scored {opt}/{crit}")
