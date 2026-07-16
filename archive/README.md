# Archive

This directory contains historical code that is useful for tracing earlier
research ideas but is not part of the supported runtime.

- `legacy/dp/soft_dp.py` is a Graph2Vid-era differentiable-DP prototype. The
  current Runner uses `dp/graph_utils.py`; the archived module has unresolved
  dependencies and should not be imported as production code.

Git history remains the source for removed `.bak` files and pre-cleanup config
copies.
