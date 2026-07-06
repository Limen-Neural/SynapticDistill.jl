# SynapticDistill.jl

Modular online training for spiking neural networks in Julia — E-prop, OTTT, and more.
Works with pure SNNs or hybrid SNN+LLM systems.

## Project overview

- **Language:** Julia (1.8+)
- **Package name:** SynapticDistill
- **License:** Dual MIT / Apache-2.0
- **Key dependencies:** Zygote (AD), MLUtils, LinearAlgebra

## Dev environment tips

```bash
# Install and instantiate
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# REPL workflow
julia --project=.
] activate .
using SynapticDistill
```

## Build & test commands

```bash
# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Quick smoke test
julia --project=. -e 'using SynapticDistill; println("OK")'
```

## Code style

- SPDX license header required on all source files: `# SPDX-License-Identifier: MIT OR Apache-2.0`
- Module name: `SynapticDistill` (not `SpikenautDistill`)
- Julia naming conventions: lowercase_snake for functions, CamelCase for types
- Keep functions small and focused; prefer pure functions where possible

## Testing instructions

- Tests live in `test/runtests.jl`
- Test file should verify module loads and exported symbols
- Add tests for any new functions or exports
- Run `julia --project=. -e 'using Pkg; Pkg.test()'` before committing
- Tests must pass before merging

## PR instructions

- Title format: `<type>: <description>` (e.g. `feat:`, `fix:`, `chore:`, `docs:`)
- New files must include the SPDX license header
- Add or update tests for any code changes
- Run tests locally before pushing
