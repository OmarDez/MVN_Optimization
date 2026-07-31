# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

The Python package `pymvn/` sits at the repository root alongside `pyproject.toml`,
so every command below runs from the root. `.github/workflows/` must stay at the
root too — GitHub Actions only picks up workflows from there.

`PLAN.md` is the authoritative design document: the mathematics (§2), the
interface contracts (§4), the experiment definitions E1–E7 (§5), and the
already-measured result that throughput is *not* the story (§7). Read it before
changing anything structural.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ".[train]" for torch/torchvision

bash kernels/build.sh            # -> kernels/libangular.so (NEON on aarch64, scalar elsewhere)
pytest tests/ -v                 # expect 81 passed, +5 skipped off-Arm

pytest tests/test_algebra.py::test_group_isomorphism -v          # one test
pytest tests/test_parity.py -k "neon" -v                         # by pattern

python bench/run_bench.py --synthetic --repeat 3 --batches 64 --bits 4 --out results/smoke.json
python bench/run_bench.py --synthetic --repeat 30 --out results/local.json
python bench/report.py results/*.json --out docs/                # -> docs/RESULTS.md

python scripts/train_zoo.py --model mlmvn_fft --dataset mnist    # -> models/*.npz
```

On this host there is no `.venv` yet and no `python` on PATH (only `python3`),
so create and activate the venv before using the commands above verbatim.

`kernels/build.sh` must be re-run after editing `kernels/angular_neon.c`; the
`.so` is gitignored, so a fresh clone has no kernel and the `neon` backend
raises until it is built. `PYMVN_THREADS=1` (the default assumption) pins the
ONNX Runtime session to one thread — benchmarks are meaningless without it.

## Architecture

```
backbone (any) → features ℝ^d → lift → T^d → AngularHead → logits
```

The claim the whole repository exists to test: with unit-modulus weights,
quantizing phase to `b` bits confines weights to the `2^b`-th roots of unity,
where complex multiplication *is* integer addition mod `L = 2^b`. A complex MAC
becomes one integer add plus a `L`-entry cos/sin lookup. At `b ≤ 4` that table
is ≤ 16 bytes = one NEON register = one `vqtbl1q_u8`.

- `pymvn/algebra.py` — the isomorphism and its primitives (`group_mul` = add,
  `group_div` = sub, `phase_to_index`, `cos_lut_i8`). Everything else is an
  implementation or measurement of what is defined here. `LUT_SCALE_I8 = 127`
  is the fixed-point convention shared by NumPy and the C kernel.
- `pymvn/angular.py` — `AngularHead`, six interchangeable backends behind one
  interface (`logits`/`predict`/`accuracy`/`mac_report`).
- `pymvn/encode.py` — `encode_fft` (phase-only spectrum, for the pure MLMVN)
  and the four lifts `minmax`/`tanh`/`cdf`/`learned` (ℝ^d → T^d for hybrid
  backbones). Which lift wins is an open question (E2), hence four.
- `pymvn/core.py` — training only: `MVNHead` (one-vs-all error correction) and
  `MLMVN` (two layers; hidden error via a truncated SVD pseudo-inverse of W2,
  refreshed every `svd_every` samples). Training is float64 on any host;
  only inference is benchmarked on Arm.
- `pymvn/quant.py` — PolarQuant + `ablate_bits`, the bit-width sweep (E3).
- `pymvn/io.py` — `.npz` checkpoint contract.
- `kernels/angular_neon.c` — `angular_gemm`, called via ctypes. Dispatches to
  the `vqtbl1q_u8` path only when `L <= 16`; scalar otherwise and on x86.

### Invariants worth knowing before editing

- **Column 0 of `W` is the bias**, matching a leading constant-1 input.
  `AngularHead._with_bias` prepends it when `X.shape[1] != d1`, so callers may
  pass either shape.
- **`phase_only=True` is what makes the datapath multiplier-free.** Keeping the
  modulus (`phase_only=False`, "full-polar") reintroduces a real multiply, and
  `mac_report()` must keep saying so.
- **The `onnx` backend is the baseline for every speedup number**, never
  `complex128`. No mainstream edge runtime supports `complex64`, so four real
  GEMMs is what actually ships.
- **`angular_naive` is the reference for the angular backends.** It is the
  literal transcription of the algebra; `angular_tiled` and `neon` must match
  it essentially exactly. Drift there is a kernel bug, not quantization.
- **Tile size is a performance knob only** — `test_tiling_is_schedule_invariant`
  enforces that it never changes results.
- Checkpoints must load with `allow_pickle=False`; no Python objects in `.npz`.
- Lifts are fitted on training features only.

### Backends

| Backend | Representation | Role |
|---|---|---|
| `complex128` | full complex | ground truth |
| `complex64` | full complex, single precision | cheap reference |
| `onnx` | real/imag split, 4 real GEMMs | **the baseline** |
| `angular_naive` | integer indices, float LUT | correctness reference |
| `angular_tiled` | uint8 indices, int8 LUT, tiling | portable optimum |
| `neon` | uint8 + `vqtbl1q_u8` via C | Arm-specific optimum |

## Testing

`pytest tests/` green is the merge gate (PLAN.md §4, Contract 4). The three
files are not interchangeable:

- `test_algebra.py` — the central claim, checked exhaustively over the whole
  group for b ∈ {1…8}. If `test_group_isomorphism` fails, nothing downstream
  means anything.
- `test_parity.py` — cross-backend agreement. Distinguishes *exact* parity
  (same arithmetic, must match to fp tolerance) from *quantized* parity (angular
  backends compute a different, lower-precision quantity). The prediction-
  agreement floors are deliberately loose because they are calibrated on random
  weights, the worst case; PLAN.md says to tighten them once real checkpoints
  land in `models/`.
- `test_contracts.py` — checkpoint format and uniform backend interface.

Tests skip rather than fail when `onnxruntime` is absent or the kernel is not
built for the host.

## Benchmark hygiene (non-negotiable, PLAN.md §6)

Median and p95 over ≥ 30 repetitions after 5 warm-ups; single thread; accuracy
(`agreement_vs_fp`) recorded in the same record as latency — a fast kernel that
changed the predictions is a regression. `.github/workflows/bench.yml` runs
tests then benchmarks on `ubuntu-24.04` and `ubuntu-24.04-arm` from the same
commit and uploads `results/` (including Arm `sysreport` output) as artifacts.

## What is deliberately not claimed

The angular kernel does **not** beat a tuned BLAS GEMM on throughput; §7 of
PLAN.md records the measurements that settled this. The primary claims are the
8–32× smaller head weights, zero real multiplies in the head, and the
machine-checked algebraic structure. Do not reintroduce throughput-superiority
language into README or docs.
