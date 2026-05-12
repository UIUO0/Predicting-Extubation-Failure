"""Re-run a specific set of cells in the notebook and persist outputs."""
from __future__ import annotations
import sys, time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB = "Predicting_Extubation_Failure.ipynb"
# Re-run Section 5.3 (cell 36) — it was the only failing cell — and then
# Section 5.9 summary (cell 42) because it reads the 5.3 CSV that must exist.
TARGETS = [36, 42]

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name="python3",
                        allow_errors=False, record_timing=True)
t0 = time.time()
with client.setup_kernel():
    for idx in TARGETS:
        cell = nb.cells[idx]
        if cell.cell_type != "code":
            continue
        preview = "".join(cell["source"]).splitlines()[0][:60]
        print(f"[run ] cell {idx}: {preview}")
        try:
            t = time.time()
            client.execute_cell(cell, idx)
            print(f"[ok  ] cell {idx} ({time.time()-t:.1f}s)")
        except CellExecutionError as e:
            msg = str(e).splitlines()[-1]
            print(f"[FAIL] cell {idx}: {msg[:250]}")
nbformat.write(nb, NB)
print(f"\n[done] {time.time()-t0:.1f}s total, notebook saved.")
