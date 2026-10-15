# OstRipening: pore-network models for Ostwald ripening

This repository contains the existing Ostwald-ripening model and an independent
copy of Ademola Adebimpe's quasi-static percolation-without-trapping model.
Both build on [pnflowPy](https://github.com/ImperialCollegeLondon/pnflowPy).

## Choose the model

| Model | Location and entry point | Paper / purpose |
| --- | --- | --- |
| Standard Valvatne–Blunt network model | Separate [pnflowPy repository](https://github.com/ImperialCollegeLondon/pnflowPy); `pnflowPy.main` | Conventional drainage and imbibition with trapping. |
| Equilibrium percolation without trapping | [`percolation_without_trapping/`](percolation_without_trapping/); `percolation_without_trapping.main` with the module path below | Adebimpe et al. (2024), [Physical Review E 110, 035105](https://doi.org/10.1103/PhysRevE.110.035105). Imported from the author's separate repository. |
| Existing Ostwald-ripening model | Root-level `main.py`, `timeDependency.py`, `equilibrium.py`; `OstRipening.main` | Time-dependent model described in Adebimpe et al. (2026), [Physical Review E 114, 035104](https://doi.org/10.1103/kbb7-wbln). The existing driver also selects its own equilibrium calculation when `mode` is false. |

The imported percolation model has a separate driver and drainage/imbibition
wrappers. It is not selected by setting the existing root driver's `RIPENING`
mode to false. The root solver files and existing data are unchanged by this
import. The paper links identify the associated models; the imported revision
has not been validated against either paper's numerical results.

## Source and setup

Clone the two repositories into the same parent directory, keeping these names:

```sh
git clone https://github.com/ImperialCollegeLondon/OstRipening.git
git clone https://github.com/ImperialCollegeLondon/pnflowPy.git
```

The expected layout is:

```text
parent/
  OstRipening/
    main.py
    percolation_without_trapping/
      main.py
  pnflowPy/
    network.py
    environment.yml
    data/
```

For a Conda setup, the environment file is in **pnflowPy**, not in this repository:

```sh
conda env create -f pnflowPy/environment.yml
conda activate flowenv
```

The drivers also import `joblib`. Check that it is available in the chosen
environment. Review the selected model's imports and pnflowPy's dependencies
when setting up the environment. This import does not add a new environment
definition or claim that a dependency combination has been tested.
See the imported model's [setup and validation notes](percolation_without_trapping/README.md).

## Run the selected model

Run commands from `parent/`, with both repositories importable. Start a fresh
Python process for each model; the imported wrappers replace functions in
pnflowPy modules during a run.

The entry points are:

```sh
# Standard model, using the example distributed by pnflowPy
python -m pnflowPy.main pnflowPy/data/input_pnflow.dat

# Existing Ostwald-ripening driver, using its own input format
python -m OstRipening.main OstRipening/data/input_pnflow.dat
```

For the imported percolation model, add the clone's directory to the module
search path and retain the author's top-level package name. In a POSIX shell
on the research machine, from the same `parent/` directory:

```sh
PYTHONPATH=OstRipening python -m percolation_without_trapping.main pnflowPy/data/input_pnflow.dat
```

Do not launch this model as `OstRipening.percolation_without_trapping.main`.
The inspected pnflowPy `compat.py` selects `OstRipening.cluster.Cluster` when
`OstRipening` is already imported. The top-level package name above preserves
the standalone route to `pnflowPy.cluster.Cluster`. This is a static import
check, not evidence of numerical equivalence.

These are entry-point instructions derived from the source layout, not a record
of successful runs. In particular, the pnflowPy example is a generic input;
it is **not** the paper-reproduction input mentioned in the author's email.
For the existing ripening driver, review `RES_DIR` and all `RIPENING` parameters
in `OstRipening/data/input_pnflow.dat` before running.

The network reader resolves the four network files relative to the input file's
directory. Keep the input alongside the matching `<name>_link1.dat`,
`<name>_link2.dat`, `<name>_node1.dat` and `<name>_node2.dat` files, and set
`NETWORK` accordingly. Read [`doc/input_pnflow.pdf`](doc/input_pnflow.pdf) and
the [pnflow guide](https://github.com/ImperialCollegeLondon/pnflow) for the
underlying input conventions. Only implemented keywords should be used.
Networks may be generated using
[pnextract](https://github.com/ImperialCollegeLondon/pnextract) or
[poreXtractor](https://github.com/ImperialCollegeLondon/poreOccupancyAnalysis).

## Provenance and validation

The imported model's four Python source files are copied without modification
from upstream commit
[`1a777b62abfa952ffdfc7b138e187e62d76a980f`](https://github.com/AIAdebimpe/percolation_without_trapping/tree/1a777b62abfa952ffdfc7b138e187e62d76a980f).
Its original README is retained as
[`UPSTREAM_README.md`](percolation_without_trapping/UPSTREAM_README.md).
[`UPSTREAM.json`](percolation_without_trapping/UPSTREAM.json) records source
paths, blob IDs, SHA-256 hashes, the dependency revision inspected and omitted
Python bytecode. The dependency revision is an inspection reference, not a
validated compatibility pin.

The 2024 paper links to this repository in reference 40. The 2026 paper links
to the [porescale index](https://github.com/ImperialCollegeLondon/porescale) in
reference 34. The author's paper-reproduction input file was not supplied with
this import and is not present in the imported upstream revision. Add it with
its provenance when available. No simulation or numerical reproduction was
performed as part of this repository import.

## License and attribution

The previous root README and the imported upstream README both state that the
code is distributed under the MIT License. Neither inspected revision supplies
a standalone MIT license text. The upstream statement is preserved verbatim in
`UPSTREAM_README.md`; this import does not invent a copyright holder or license
file. pnflowPy is a separate dependency with its own `LICENSE`, which must be
consulted separately.

## Contributions and contact

Please use a pull request for code or documentation contributions. For model
questions, contact Ademola Adebimpe at a.adebimpe21@imperial.ac.uk or
aisaac.adebimpe@gmail.com. See also the
[Imperial pore-scale modelling group](https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling).
