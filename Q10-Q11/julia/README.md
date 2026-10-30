# Julia environment

Julia 1.12 + JupyterLab, run in the project container (`opt` service in `code/docker-compose.yaml`).

## Docker (primary)

```
cd code
docker compose up --build          # JupyterLab at http://localhost:8896 (no token)
```

In JupyterLab pick the **Julia 1.12 (project)** kernel (`julia-project`). It starts Julia with
`--project=/home/repo-intro/scripts/julia`, i.e. this folder's `Project.toml`/`Manifest.toml`,
so packages are reproducible and versioned in git.

- Add a package from a notebook cell: `using Pkg; Pkg.add("DataFrames")`; commit the updated
  `Project.toml` and `Manifest.toml`. On a fresh clone run `using Pkg; Pkg.instantiate()` once.
- Notebooks and `.jl` scripts go in `code/scripts/` (mounted at `/home/repo-intro/scripts`).
- Julia REPL / script from the host: `docker compose run --rm opt julia-1.12 --project=scripts/julia scripts/<file>.jl`.
- Kernels in the image: `julia-project` (use this), `julia-1.12` (no project), `julia-1.9` (legacy, shipped by the base image).

How it is built (`code/Dockerfile`): the official Julia 1.12.5 tarball is unpacked in `/opt`
(arch-aware for Apple Silicon and x86), the depot lives in `/opt/julia_depot` (owned by the
notebook user), and IJulia registers the `julia-project` kernel. Change `JULIA_VERSION` to upgrade.

## Reproducibility

`Project.toml` lists the direct dependencies (DataFrames, Distributions, FastGaussQuadrature, FiniteDiff,
LaTeXStrings, Optim, Plots, ReadStatTables, StatsBase) and `Manifest.toml` pins every version (Julia 1.12.5).
Always run against this environment, never the global one, and commit both files after any `Pkg.add`/`Pkg.update`.
Recreate it anywhere with `julia --project=code/scripts/julia -e 'using Pkg; Pkg.instantiate()'`.
Keep tools (IJulia, Revise) in the global environment, not here.

## Native (alternative, no Docker)

Julia is managed with [juliaup](https://github.com/JuliaLang/juliaup) (`juliaup status`; 1.12.x is installed on the author's Mac).

```
julia --project=code/scripts/julia -e 'using Pkg; Pkg.instantiate()'
```

Best native interface: **VS Code + the Julia extension** (run `.jl` files and `# %%` cells with the
integrated REPL, plots and variable viewer), opening the repo and selecting `code/scripts/julia` as the
environment. Native notebook kernel: **Julia (problem-set)** (`julia-problem-set`, created with `IJulia.installkernel("Julia (problem-set)", "--project=<repo>/code/scripts/julia"; specname="julia-problem-set")`; select it in VS Code). Check with `Base.active_project()`. Otherwise, `using Pkg; Pkg.add("IJulia")` in the global env, then
`julia --project=code/scripts/julia -e 'using IJulia; notebook(detached=true)'` (or pick the Julia kernel
in VS Code / Jupyter). Native and Docker use the same `Manifest.toml`; keep the Julia minor version
(1.12) aligned or the manifest will be re-resolved.
