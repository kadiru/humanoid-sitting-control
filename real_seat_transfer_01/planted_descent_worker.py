"""Add planted-foot retargeting after the existing alignment and rest controls."""
import runpy
import sys
from pathlib import Path

from planted_descent import attach
import sitting_load_effort


if __name__ == '__main__':
    entry = Path(__file__).with_name('sitting_load_effort_worker.py')
    previous = sitting_load_effort.attach
    sitting_load_effort.attach = lambda config, inherited: attach(config, lambda: previous(config, inherited))
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name='__main__')
