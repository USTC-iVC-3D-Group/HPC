"""Prefer HPC's locally built CUDA extensions over unrelated installed builds."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1] / 'submodules'
for _name in ('gridencoder', 'simple-knn', 'diff-gaussian-rasterization'):
    _path = _root / _name
    if _path.is_dir():
        sys.path.insert(0, str(_path))
