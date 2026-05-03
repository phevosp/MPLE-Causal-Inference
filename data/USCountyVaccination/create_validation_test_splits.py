"""Thin shim: delegates to repo-root create_validation_test_splits."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from create_validation_test_splits import (  # noqa: F401, E402
    DEFAULT_INNER_NUM_FOLDS,
    DEFAULT_OUTER_NUM_FOLDS,
    DEFAULT_TEST_FOLD_ID,
    build_validation_test_splits_for_experiment,
    create_validation_test_splits,
    main,
    parse_args,
)

if __name__ == "__main__":
    main(sys.argv[1:])
