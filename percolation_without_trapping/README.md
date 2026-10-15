# Equilibrium percolation without trapping

This directory contains an unchanged copy of the four Python source files from
[AIAdebimpe/percolation_without_trapping](https://github.com/AIAdebimpe/percolation_without_trapping)
at commit `1a777b62abfa952ffdfc7b138e187e62d76a980f`, inspected on 15 October 2026.
It is the independent quasi-static model identified by the author for
Adebimpe et al. (2024),
[Physical Review E 110, 035105](https://doi.org/10.1103/PhysRevE.110.035105).
The current source snapshot has not been verified to reproduce the paper.

## What this directory contains

- `main.py`: input loading, single-phase initialization, and drainage/imbibition cycles.
- `twoPhaseD.py`: primary and secondary drainage wrappers around pnflowPy.
- `twoPhaseImb.py`: primary and secondary imbibition wrappers around pnflowPy.
- `__init__.py`: upstream package marker.
- `UPSTREAM_README.md`: the author's original README, unchanged.
- `UPSTREAM.json`: the exact revision, file mapping, blob IDs and SHA-256 hashes.

Tracked upstream `__pycache__/*.pyc` files are intentionally excluded. No source
algorithm, input parameters or existing OstRipening solver files were changed.

The imbibition wrapper explicitly passes `includeTrapping=False` and computes
neighboring entry pressures with `trapping=False`. The drainage implementation
still checks wetting-phase trapping. The model name should not be interpreted
as a claim that all trapping checks have been removed everywhere.

## Dependency and entry point

This is an extension of pnflowPy, not a standalone implementation. Follow the
[parent README](../README.md) to put `OstRipening/` and `pnflowPy/` side by side.
The pnflowPy revision inspected during import was
`f90dd008d14688665d11633e27a9dc6eedf4357f`; compatibility has not been executed
or numerically validated. Its `environment.yml` is the environment source to
inspect. Direct imports in this package require `numpy`, `sortedcontainers`,
`dill` and `joblib`, in addition to pnflowPy's own dependencies.

From the directory containing both repositories, in a POSIX shell on the
research machine:

```sh
PYTHONPATH=OstRipening python -m percolation_without_trapping.main pnflowPy/data/input_pnflow.dat
```

`PYTHONPATH` makes this directory importable under the original top-level
package name, while `-m` resolves its relative imports. Do not prefix this
module name with `OstRipening`: the inspected dependency's `compat.py` switches
to the root ripening `Cluster` when `OstRipening` is in `sys.modules`. The
standalone entry point retains the route to `pnflowPy.cluster.Cluster`.
The `python -m pnflowPy.main ...` command in `UPSTREAM_README.md` selects the
standard model instead. The original README also refers to a local
`environment.yml` and local example data that are absent from that upstream
snapshot; use the actual dependency's files and the instructions above.

The input path above is a generic example distributed with pnflowPy. To use
another input, keep it with the corresponding network files and specify its
path. The network basename is read from `NETWORK`; files are read relative to
the input's directory. `SAT_CONTROL` defines the successive drainage/imbibition
targets. This driver does not call the root-level time-dependent ripening solver.

Results are written under `percolation_without_trapping_results/`, relative to
the process working directory. Start each model in a fresh process because the
wrappers replace `pnflowPy.tPhaseD.popUpdateOilInj` and
`pnflowPy.tPhaseImb.popUpdateWaterInj` in memory.

## Reproduction status

The author's email refers to an attached input file for the first paper.
That attachment was not supplied with this import and does not occur in the
upstream tree. Neither the bundled pnflowPy example nor the existing
OstRipening input should be described as that paper's reproduction input.

Import checks cover file identity, the repository diff, relative module paths
and source inspection. No dependency environment was created, package imported,
simulation run or paper result reproduced. A future reproduction needs the
author's input file, matching network data, documented dependency versions and
comparison to the paper's results on an appropriate research machine.

## Attribution and license statement

Author/source: Ademola Adebimpe, repository linked above. The original README
states MIT, but that snapshot contains no standalone license text or copyright
notice. The original statement is retained in `UPSTREAM_README.md`. pnflowPy's
separate license remains in its own repository. See `UPSTREAM.json` for details.
