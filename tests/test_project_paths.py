import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_inspect_csv_runs_from_source_directory():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "inspect_csv.py")],
        cwd=str(REPO_ROOT / "src"),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "File:" in result.stdout
