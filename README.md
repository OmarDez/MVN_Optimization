# pymvn-arm

**Phase-native inference for Multi-Valued Neuron models on Arm64.**

A weight on the unit circle times an activation on the unit circle is a
rotation — and a rotation is an *addition of angles*. Quantize those angles to
`b` bits and the complex multiply-accumulate becomes **one modular integer add
plus a 2^b-entry table lookup**. At `b = 4` that table is 16 bytes: exactly one
AArch64 NEON register, exactly one `vqtbl1q_u8` instruction.

This repository turns that observation into executable, measured, falsifiable
code.

> Full technical plan, mathematics and experiment design: **[PLAN.md](PLAN.md)**

---

## Why this is interesting

Multi-Valued Neurons (Aizenberg's school) are complex-valued neurons whose
activation depends only on **phase**: `P(z) = z/|z|`. They learn by error
correction rather than backpropagation, and a single MVN solves XOR and Parity-n
— problems not linearly separable over the reals.

They are also nearly impossible to deploy. No mainstream edge runtime — ONNX
Runtime, ExecuTorch, LiteRT — supports `complex64`. Shipping a complex layer
today means decomposing it into four real GEMMs.

This project takes the opposite route. Instead of emulating complex arithmetic
with real arithmetic, it exploits the group structure to remove the arithmetic
altogether.

---

## The mathematics in one screen

Unit-modulus complex numbers form a group under multiplication. Restricting the
phase to `b` bits confines weights to the cyclic subgroup of `L = 2^b`-th roots
of unity, and that subgroup is isomorphic to the integers mod L:

```
μ_L = { e^{i·2πj/L} }  ≅  ℤ_L        via   φ(e^{i·2πj/L}) = j
```

Because φ is a group homomorphism, **multiplication becomes addition**:

```
e^{i·2πk_w/L} · e^{i·2πk_x/L} = e^{i·2π((k_w + k_x) mod L)/L}

        k_out = (k_w + k_x) mod L
```

| per MAC | complex multiply | angular |
|---|---|---|
| real multiplies | 4 | **0** |
| real adds | 2 | 0 |
| integer adds | 0 | **1** |

`L` is a power of two, so the modulo is a bitwise AND. This is verified
**exhaustively over the entire group** in `tests/test_algebra.py`.

**Where it stops.** The homomorphism covers the product only — accumulation
leaves the group, since `|Σ wᵢxᵢ| ≠ 1`. Reconstruction needs a cosine/sine
lookup per term. That lookup is the real cost, and it is why `b` matters:
at `b ≤ 4` the table fits one NEON register.

---

## Install

Requires Python ≥ 3.10 and a C compiler. No GPU. **No Arm hardware needed** —
CI provides it.

```bash
git clone https://github.com/OmarDez/MVN_Optimization.git
cd MVN_Optimization

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

bash kernels/build.sh      # NEON path on aarch64; portable scalar C elsewhere
pytest tests/ -v           # expect 81 passed (+5 skipped off-Arm)
```

## Run

```bash
# Benchmark all backends, no checkpoint required
python bench/run_bench.py --synthetic --repeat 30 --out results/local.json

# Train and freeze a model, then benchmark it
python scripts/train_zoo.py --model mlmvn_fft --dataset mnist
python bench/run_bench.py --models models/*.npz --out results/zoo.json

# Generate result tables
python bench/report.py results/*.json --out docs/
```

## Verify the central claim yourself

```bash
python - <<'PY'
import numpy as np
from pymvn import algebra as alg

b, L = 4, 16
ka, kb = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
lhs = alg.index_to_unit(ka.ravel(), b) * alg.index_to_unit(kb.ravel(), b)   # complex multiply
rhs = alg.index_to_unit(alg.group_mul(ka.ravel(), kb.ravel(), b), b)        # integer add

print("max error over all", L*L, "pairs:", np.abs(lhs - rhs).max())
print("int8 cosine LUT:", alg.cos_lut_i8(b).nbytes, "bytes")
PY
```

---

## Usage

```python
from pymvn import AngularHead, load_checkpoint, lift

ckpt = load_checkpoint("models/resnet18_mnist.npz")
X    = lift(features, ckpt.lift)          # ℝ^d -> T^d

baseline = AngularHead(ckpt.W, b=4, backend="onnx")    # 4 real GEMMs
angular  = AngularHead(ckpt.W, b=4, backend="neon")    # integer adds + TBL

assert (baseline.predict(X) == angular.predict(X)).mean() > 0.99
print(angular.mac_report())
# {'head_macs': 5130, 'real_mults_per_mac': 0, 'int_adds_per_mac': 1,
#  'lut_bytes': 16, 'multiplier_free': True}
```

### Backends

| Backend | Representation | Role |
|---|---|---|
| `complex128` | full complex arithmetic | ground truth |
| `complex64` | full complex, single precision | cheap reference |
| `onnx` | real/imag split, 4 real GEMMs | **honest baseline** |
| `angular_naive` | integer indices, float LUT | proof of concept |
| `angular_tiled` | uint8 indices, int8 LUT, tiling | portable reference |
| `neon` | uint8 + `vqtbl1q_u8` | Arm-specific optimum |

Speedups are reported against `onnx`, never against `complex128`. The ONNX
four-GEMM path is what anyone would actually deploy; beating a deliberately slow
reference would prove nothing.

### Any backbone plugs in

```
backbone (any) ──► features ℝ^d ──► lift ──► T^d ──► AngularHead ──► logits
```

Four lift maps are provided (`minmax`, `tanh`, `cdf`, `learned`) because which
one is best is an open empirical question — see PLAN.md, Experiment E2.

---

## Reproducing on Arm without Arm hardware

GitHub-hosted `ubuntu-24.04-arm` runners are free for public repositories and
run on Cobalt 100 (Arm Neoverse N2, 4 vCPU, Armv9-A with SVE2).

`.github/workflows/bench.yml` runs the test suite and the full benchmark on both
`x86_64` and `aarch64` **from the same commit**, captures the host configuration
with Arm's own `sysreport`, and uploads everything as artifacts. Push, or
dispatch the workflow manually.

---

## Honest scope

This project does **not** claim to beat a tuned BLAS GEMM on raw throughput. A
naive angular kernel loses to decades of GEMM optimization, and NEON's 16-wide
`TBL` buys roughly parity rather than a win. PLAN.md §7 documents the measured
numbers that led to this conclusion.

What is claimed, and measured:

- **8–32× smaller head weights** — phase indices instead of float pairs
- **Zero real multiplies** in the head — architecturally true, verified in code
- **A machine-checked algebraic structure** connecting encoding, model,
  quantization and instruction set

---

## License

Apache-2.0. See [LICENSE](LICENSE).

## References

- Aizenberg, I. — *Complex-Valued Neural Networks with Multi-Valued Neurons*, Springer, 2011
- Oppenheim, A. & Lim, J. — *The importance of phase in signals*, Proc. IEEE, 1981
- Arm — [`ubuntu-24.04-arm` GitHub-hosted runners](https://learn.arm.com/learning-paths/cross-platform/github-arm-runners/)
- Arm — [`sysreport`](https://github.com/ArmDeveloperEcosystem/sysreport)
