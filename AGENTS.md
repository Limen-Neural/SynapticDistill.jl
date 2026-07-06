# SynapticDistill.jl

Modular online training for spiking neural networks (SNNs) in Julia — E-prop, OTTT, and more.
Works with pure SNNs or hybrid SNN+LLM systems.

## Project overview

- **Language:** Julia (1.8+)
- **Package name:** SynapticDistill
- **License:** Dual MIT / Apache-2.0
- **Key dependencies:** Zygote (automatic differentiation), MLUtils, LinearAlgebra

## Scope boundaries

**SynapticDistill owns:**
- Differentiable/online distillation and teacher-student transfer
- E-prop, OTTT, and surrogate gradient training rules
- Gradient computation for spiking neurons
- Training loop utilities and callbacks

**SynapticDistill does NOT own:**
- Reward-modulated STDP or Hebbian learning (see `plasticity-lab`)
- IPC wire protocol types (see `corpus-ipc`)
- Domain-specific model architectures (trading, mining, etc.)
- Hardware-specific optimizations (unless generic)

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

- SPDX license identifier required on all source files: `# SPDX-License-Identifier: MIT OR Apache-2.0`
- Module name: `SynapticDistill` (not `SpikenautDistill`)
- Julia naming conventions: lowercase_snake for functions, CamelCase for types
- Keep functions small and focused; prefer pure functions where possible

## Testing instructions

- Tests live in `test/runtests.jl`
- Test file verifies module loads and exported symbols
- Add tests for any new functions or exports
- Run `julia --project=. -e 'using Pkg; Pkg.test()'` before committing
- Tests should pass before merging (CI will verify automatically)

## PR instructions

- Title format: `<type>: <description>` (e.g. `feat:`, `fix:`, `chore:`, `docs:`)
- New files should include the SPDX license header
- Add or update tests for any code changes
- Run tests locally before pushing
- Resolve all review threads before merge
