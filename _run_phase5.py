"""
One-shot executor for Phase 5 cells (33-42) of the dissertation notebook.

Strategy
--------
* Open the notebook in-place with `nbformat`.
* Spin up a single IPython kernel via `nbclient.NotebookClient`.
* Execute *only* cells 33-42 inside that single kernel session, in order,
  preserving stdout / display_data / mime-bundle outputs back onto the
  notebook cells (so the professor sees the figures and tables when
  opening the .ipynb).
* Save the notebook back to disk.

Notes
-----
* Phase 5 cells were written to be `_safe_load`-self-contained, so they
  load X_test / y_test / results / best_name / best_model from the
  Phase-4 .pkl artifacts without re-training.
* We DO NOT execute Phases 0-4 (they take hours and the artifacts are
  already persisted).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB_PATH    = Path("Predicting_Extubation_Failure.ipynb")
START, END = 33, 42                # inclusive Phase-5 cell range
TIMEOUT_S  = 900                   # 15-min cap per cell (SHAP can be slow)


def main() -> int:
    nb = nbformat.read(NB_PATH, as_version=4)
    n_cells = len(nb.cells)
    print(f"[boot] notebook has {n_cells} cells; targeting [{START}..{END}]")

    client = NotebookClient(
        nb,
        timeout       = TIMEOUT_S,
        kernel_name   = "python3",
        allow_errors  = False,        # surface failures immediately
        record_timing = True,
    )

    failures: list[tuple[int, str]] = []
    t0 = time.time()
    with client.setup_kernel():
        for idx in range(START, min(END + 1, n_cells)):
            cell = nb.cells[idx]
            if cell.cell_type != "code":
                print(f"[skip] cell {idx}: {cell.cell_type}")
                continue
            preview = "".join(cell.source).splitlines()[0][:70]
            print(f"[run ] cell {idx}: {preview}")
            t_cell = time.time()
            try:
                client.execute_cell(cell, idx)
                dt = time.time() - t_cell
                print(f"[ok  ] cell {idx}  ({dt:6.1f}s)")
            except CellExecutionError as e:
                dt = time.time() - t_cell
                failures.append((idx, str(e).splitlines()[-1]))
                print(f"[FAIL] cell {idx}  ({dt:6.1f}s): "
                      f"{str(e).splitlines()[-1][:200]}")
                # keep going so we collect outputs from later cells too
                # (NotebookClient already wrote the error output onto the cell)

    nbformat.write(nb, NB_PATH)
    total = time.time() - t0
    print(f"\n[done] total wall-time: {total/60:.1f} min")
    print(f"[done] notebook saved with new outputs -> {NB_PATH}")

    if failures:
        print("\n[!!!] failed cells:")
        for idx, msg in failures:
            print(f"  - cell {idx}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
