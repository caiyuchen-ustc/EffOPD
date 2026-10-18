"""Check the selected verl tree without starting Ray or importing CUDA libraries."""
import importlib.util
import os
from pathlib import Path


def check_imports():
    root = Path(os.environ['EFFOPD_VERL_ROOT']).resolve()
    spec = importlib.util.find_spec('verl')
    if spec is None or spec.origin is None:
        raise RuntimeError('verl is not importable')
    actual = Path(spec.origin).resolve()
    expected = root / 'verl' / '__init__.py'
    if actual != expected:
        raise RuntimeError(f'Wrong verl: {actual}; expected {expected}')
    print(f'[EffOPD import check] {actual}', flush=True)


if __name__ == '__main__':
    check_imports()
