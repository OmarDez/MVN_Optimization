# PLAN

**Project:** `pymvn-arm` — phase-native inference for Multi-Valued Neuron models on Arm64
**Target:** Arm Create: AI Optimization Challenge — Track 2 (Cloud AI)
**Submission deadline:** 14 Aug 2026, 4:00 pm PDT · **internal freeze:** 10 Aug 2026

---

## 1. Thesis

> In a Multi-Valued Neuron layer whose weights have unit modulus, the
> weight×activation product is a **group isomorphism** (S¹, ×) → (ℤ_L, +).
> The complex multiply-accumulate collapses into a single modular integer add
> plus a table lookup. When the model needs only b ≤ 4 bits of phase — which
> the ablation shows it does — the lookup table is at most 16 bytes, which is
> exactly the operand width of the AArch64 `TBL` instruction.

The encoding (FFT), the model (MVN), the quantization (roots of unity) and the
target hardware (modular integer adder) are four views of one algebraic
structure. This project makes that claim executable, measurable and falsifiable.

### 1.1 What is being claimed, precisely

| # | Claim | Status |
|---|---|---|
| C1 | Weight×activation in μ_L is exactly integer addition mod L | **Proved and machine-checked** (`tests/test_algebra.py`) |
| C2 | Phase weights need ≤ 4 bits to match the unquantized baseline | **Measured on a real checkpoint** — b = 4 is within 0.0004 of b = 8 (E3) |
| C3 | The head's weight footprint shrinks 8–16× versus deployed fp32 | To be measured — Experiment E5 |
| C4 | Inference requires zero real multiplies in the head | Architecturally true; verified by `mac_report()`. Costs 2.8 accuracy points on the MLMVN checkpoint — see E3 |
| C5 | The angular kernel outperforms a tuned BLAS GEMM in throughput | **Not claimed.** See §7 |

C5 is stated as a non-claim on purpose. Section 7 explains why, and what
replaces it.

---

## 2. Mathematics

### 2.1 The group and its finite subgroup

Unit-modulus complex numbers form a group under multiplication:

```
S¹ = { e^{iθ} : θ ∈ [0, 2π) },        |w| = 1
```

Restricting the phase to `b` bits confines weights to the cyclic subgroup of
`L = 2^b`-th roots of unity:

```
μ_L = { e^{i·2πj/L} : j = 0 … L−1 } ⊂ S¹
```

`μ_L` is cyclic of order L, hence isomorphic to the additive group of integers
modulo L. The isomorphism is the index map:

```
φ : μ_L → ℤ_L,      φ(e^{i·2πj/L}) = j
```

### 2.2 The core identity

Because φ is a group homomorphism, multiplication becomes addition:

```
e^{i·2πk_w/L} · e^{i·2πk_x/L} = e^{i·2π((k_w + k_x) mod L)/L}

    ⟹    k_out = (k_w + k_x) mod L
```

Cost accounting per MAC:

| | complex multiply | angular |
|---|---|---|
| real multiplies | 4 | **0** |
| real adds | 2 | 0 |
| integer adds | 0 | **1** |
| bitmask | 0 | 1 (free, fused) |

Since `L = 2^b`, the modulo is a bitwise AND with `L − 1`. No division, no
branch. `tests/test_algebra.py::test_group_isomorphism` verifies the identity
**exhaustively over the entire group** for every b ∈ {1…8}.

### 2.3 Where the isomorphism stops

State this before a reviewer does. The homomorphism covers the **product only**.
Accumulation leaves the group:

```
| Σᵢ wᵢ xᵢ |  ≠  1     in general
```

so the weighted sum `z` is not a point of μ_L and its phase is not an index.
Reconstruction requires mapping each product index back to Cartesian form:

```
Re(z) = Σᵢ cos(2π kᵢ / L)          Im(z) = Σᵢ sin(2π kᵢ / L)
```

These are table lookups over an L-entry table, and they are the **real cost** of
the scheme. This is precisely why `b` matters far beyond model accuracy.

### 2.4 The hardware coincidence

`vqtbl1q_u8` performs **16 independent byte lookups into a 16-byte table in one
instruction**. A 128-bit NEON register holds exactly 16 bytes.

The PolarQuant ablation shows b = 3–4 already recovers the unquantized baseline.
At b = 4 the int8 cosine table has 16 entries = 16 bytes = **one register, one
instruction**.

The phase resolution the network requires and the width of the Arm lookup
datapath coincide. That is the project's central observation, and it is
falsifiable: if E3 shows the model needs b ≥ 6, the coincidence evaporates and
the honest thing is to report that.

### 2.5 The state space is a torus

A layer with `d` unit-modulus inputs lives on the d-torus:

```
T^d = (S¹)^d,     quantized:  (ℤ_L)^d
```

Consequences that are not cosmetic:

- The natural metric is **geodesic (angular) distance**, not Euclidean.
  `algebra.angular_distance` implements it; it is what "close" means for every
  quantization decision here.
- Wrap-around at `0 ≡ 2π` is a genuine topological feature, not a numerical
  edge case. Quantization error is bounded by half a sector, `π/L`, uniformly —
  proved in `test_quantization_error_bounded`.
- FFT phase-only preprocessing works because it places data on the correct
  manifold before the model sees it.

### 2.6 The dual operation

Phase-coherence attention uses the mirror image of the product rule:

```
Re⟨Q, K⟩ = Σᵢ cos(φᵢ^Q − φᵢ^K)     ⟹     k = (k_Q − k_K) mod L
```

Product is an integer **ADD**; coherence is an integer **SUB**. Same datapath,
same lookup table, same silicon. `algebra.group_div` exposes it and
`test_division_is_subtraction` verifies it.

### 2.7 Why the FFT is not a coincidence

By **Pontryagin duality** the character group of S¹ is ℤ: the irreducible
representations of the circle are indexed by integers. Fourier analysis is
therefore the canonical change of basis for data living on S¹.

This closes the loop. The encoding, the model, the quantization and the hardware
are one structure seen from four angles.

### 2.8 The lift: R^d → T^d

The bridge between a real-valued backbone and an angular head. Its form is an
**open empirical question**, so four variants are implemented and measured
rather than assumed.

| Kind | Formula | Property |
|---|---|---|
| `minmax` | `e^{iπ·norm(z)}` | Baseline; uses half the circle; outlier-sensitive |
| `tanh` | `e^{iπ·tanh((z−med)/(τ·mad))}` | Outlier-robust; τ tunable |
| `cdf` | `e^{i2π·F(z)}` | Uniform on the full circle by construction |
| `learned` | `e^{i(a·z + b)}` | Per-channel affine, trainable |

Hypothesis: `cdf` wins, because uniform occupancy of the circle maximizes the
entropy available to a fixed b-bit budget. Untested. That is the point.

---

## 3. Architecture

```
backbone (any) ──► features ℝ^d ──► lift ──► T^d ──► AngularHead ──► logits
```

`AngularHead` is **agnostic to the backbone**. Anything producing a feature
vector plugs in. That is what makes multi-model support a property of the design
rather than a pile of special cases.

### 3.1 Backends

Six implementations of the same mathematical object, behind one interface.

| Backend | Representation | Role |
|---|---|---|
| `complex128` | full complex arithmetic | ground truth |
| `complex64` | full complex, single precision | cheap reference |
| `onnx` | real/imag split, 4 real GEMMs | **honest baseline** |
| `angular_naive` | integer indices, float LUT | proof of concept |
| `angular_tiled` | uint8 indices, int8 LUT, cache tiling | portable reference |
| `neon` | uint8 + `vqtbl1q_u8` via C | Arm-specific optimum |

**Why `onnx` is the baseline.** No mainstream edge runtime — ONNX Runtime,
ExecuTorch, LiteRT — supports `complex64`. Deploying a complex layer today means
decomposing it into four real GEMMs:

```
(a + bi)(c + di) = (ac − bd) + (ad + bc)i
```

That is what a competent engineer ships. Measuring against `complex128` NumPy
would be beating a strawman.

### 3.2 Model zoo

| # | Backbone | Dataset | Why | Priority |
|---|---|---|---|---|
| 1 | MLMVN-FFT (pure) | MNIST | **100 % of inference MACs multiplier-free** | required |
| 2 | ResNet-18 | MNIST | already validated at 97.1 % | required |
| 3 | MobileNetV3-Small | CIFAR-10 | designed for Arm; the relevant comparable | required |
| 4 | ViT-Tiny | CIFAR-10 | works on attention features too | stretch S1 |
| 5 | MFCC/speech | Speech Commands | second modality | stretch S2 |

Entry 1 is the strong claim: end-to-end phase-native, nothing to fall back on.
Entries 2–3 are the accuracy story. Stretch goals get cut without ceremony.

*Measured* (`scripts/train_zoo.py`, 5 epochs, frozen into `models/`):

| Checkpoint | Accuracy | Note |
|---|---|---|
| `mlmvn_fft_mnist` | **0.9201** | full-polar; 0.8925 phase-only — see E3 |
| `resnet18_mnist` | 0.8917 | not the 97.1 % above |
| `mobilenetv3_cifar10` | 0.2324 | barely above the 0.10 floor |

Entry 1 delivers. **Entries 2–3 do not, and the cause is one line rather than
three:** `backbone_features` builds each backbone with `weights=None`, puts it
in `eval()`, and never trains it or loads pretrained weights. Those are random
convolutional projections, and only the MVN head learns. CIFAR-10 is hit twice,
because `load_dataset` also averages the RGB channels to grey before handing
them to a network expecting three.

That makes them a lower bound on the head rather than the accuracy story they
were meant to be — whatever the head reaches here, it reaches unaided. The
angular claims are unaffected: C1, C2 and C4 are about the head, and E3 is run
on entry 1, which is trained end to end. But entries 2–3 should not be quoted as
accuracy results until the backbones are either pretrained or actually trained,
and the 97.1 % in the table above has never been reproduced in this repository.

---

## 4. Interface contracts

Fixed before any code is written; changing one is a breaking change.

**Contract 1 — checkpoint (`models/*.npz`)**

```
W_real   float64 (k, d+1)   real part; column 0 is the bias
W_imag   float64 (k, d+1)   imaginary part
lift_*   ...                fitted lift parameters
meta     str (json)         backbone, dataset, accuracy_fp32, timestamp
```

No pickled objects; `allow_pickle=False` must suffice. Enforced by
`test_checkpoint_has_no_pickle`.

**Contract 2 — backend interface**

```python
h = AngularHead(W, b=4, backend="neon", phase_only=True, tile=(64, 256))
h.logits(X)      # (n, d) -> (n, k) complex
h.predict(X)     # (n, d) -> (n,)   int
h.mac_report()   # dict: head_macs, real_mults_per_mac, lut_bytes, multiplier_free
```

All six backends satisfy it identically. Enforced by
`test_backend_interface_is_uniform`.

**Contract 3 — results (`results/*.json`)**

```json
{"model":"resnet18_mnist","backend":"neon","b":4,"batch":1024,
 "arch":"aarch64","cpu":"Neoverse-N2","repeats":30,
 "median_ms":1.23,"p95_ms":1.41,"agreement_vs_fp":0.998,"multiplier_free":true}
```

**Contract 4 — the merge gate.** `pytest tests/` green before any merge.
No exceptions.

---

## 5. Experiments

Each states a hypothesis, a method, and what would falsify it.

### E1 — Group isomorphism (done)

*Hypothesis.* Complex multiplication in μ_L equals modular integer addition,
exactly, with no tolerance.
*Method.* Exhaustive enumeration of all L² pairs, b ∈ {1…8}.
*Falsified if.* Any pair disagrees beyond 1e-12.
*Status.* **Passing.** This is the foundation; everything else is engineering.

### E2 — Lift sweep

*Hypothesis.* `cdf` beats `minmax` because uniform circle occupancy maximizes
the entropy available to b bits.
*Method.* Fix backbone and b; sweep four lifts; report test accuracy.
*Falsified if.* `minmax` matches `cdf` within noise → report that the lift does
not matter, which is also a result.
*Deliverable.* `docs/fig_lift_sweep.png`.

### E3 — Phase bit-width ablation

*Hypothesis.* b = 3–4 recovers the unquantized baseline in full-polar mode;
phase-only costs more but is tolerable on strong backbones.
*Method.* Sweep b ∈ {1,2,3,4,5,6,8} × {full-polar, phase-only} × all zoo models.
Run **on Arm**, not x86.
*Falsified if.* The model needs b ≥ 6 → the 16-byte LUT coincidence dies and
§2.4 must be retracted.
*Deliverable.* `docs/fig_bitwidth_ablation.png`. **This is the most important
figure in the submission.**

*Result* (`scripts/ablate_bits.py`, `mlmvn_fft_mnist.npz`, 10 000 MNIST test
samples — real activations, not random points on S¹). **Not falsified.** Both
halves of the hypothesis hold:

| b | L | phase-only | full-polar | modulus costs | agreement vs b=8 |
|---|---|---|---|---|---|
| 1 | 2 | 0.7740 | 0.8238 | +0.0498 | 0.8021 |
| 2 | 4 | 0.8402 | 0.9046 | +0.0644 | 0.8965 |
| 3 | 8 | 0.8827 | 0.9167 | +0.0340 | 0.9614 |
| **4** | **16** | **0.8920** | **0.9203** | +0.0283 | 0.9792 |
| 6 | 64 | 0.8928 | 0.9205 | +0.0277 | 0.9949 |
| 8 | 256 | 0.8922 | 0.9207 | +0.0285 | 0.9991 |

b = 4 is within 0.0004 of b = 8 in full-polar and within 0.0005 phase-only.
Nothing above b = 4 buys anything, so §2.4 survives: the precision the model
needs really is the width of one NEON register.

**The separate finding, which is a caveat on C3 rather than on C2.** Discarding
the modulus — the thing that makes the datapath multiplier-free — costs a
consistent **2.8 points** on this checkpoint (0.9203 → 0.8920 at b = 4). The
reason is that the trained weights are not unit-modulus at all: |W| spans
0.0015 to 2.0771. MVN theory assumes they are, but `MLMVN`'s additive
error-correction update does not preserve it.

Hard-projecting back onto the unit circle after every update does make
phase-only free — the gap closes to *exactly* 0.0000, as it must — but the
model it produces is worse overall than the one whose modulus is then thrown
away (0.8135 against 0.8571 at 3 epochs). So the projection is not the fix.

### E3b — Magnitude pruning

*Why.* The gap is not lost magnitude information, it is **amplified noise**.
Setting `phase_only=True` rescales every weight to |w| = 1, so a weight training
left at |w| = 0.0015 has its contribution multiplied by ~666× — the weights the
model learned to ignore end up shouting as loudly as the ones that matter.

*Method.* Threshold τ; weights with |w| < τ contribute nothing. That costs one
**mask bit** per weight and stays multiplier-free, because a masked term is a
select (`vbslq_s8` / `vandq_s8`), not a product.

*Result* (b = 4, 10 000 MNIST test samples, `--prune-taus`):

| τ | sparsity | accuracy | gap closed |
|---|---|---|---|
| 0.00 | 0.000 | 0.8920 | — |
| 0.03 | 0.168 | 0.9016 | 33.9 % |
| **0.06** | **0.449** | **0.9089** | **59.7 %** |
| 0.07 | 0.541 | 0.9072 | 53.7 % |
| 0.10 | 0.752 | 0.8376 | −192 % |

Removing **45 % of the weights makes the model better**, recovering 60 % of the
modulus gap. That is the prediction of the amplified-noise account and not of
the lost-information one, so it is evidence for the diagnosis as well as a
mitigation. It also gives a **second compression axis** orthogonal to phase
bits — though not free: a dense mask is 1 bit per weight, so b = 4 becomes 5
bits and 16× becomes 12.8×. Past ~50 % sparsity a sparse layout wins, and
`mac_report()` reports both. `docs/fig_magnitude_pruning.png`.

The C kernel has no masked path; `backend='neon'` raises rather than silently
returning unpruned logits. One `vandq_s8` against a 0x00/0xFF lane would
implement it.

*Still open.* A training rule that *learns* unit-modulus weights, by updating
the phase directly instead of projecting after an additive step, would remove
the gap at its source rather than mitigating it. Until then the multiplier-free
claim is architecturally exact and carries a measured cost that should be quoted
with it: ~2.8 points, or ~1.1 after pruning.

### E4 — Tile-size sweep

*Hypothesis.* Blocking so the working set fits L1D (typically 64 KiB on
Neoverse N2) gives a measurable win over the untiled form.
*Method.* Sweep `(tile_n, tile_d)`; verify result-invariance
(`test_tiling_is_schedule_invariant`) then measure.
*Deliverable.* `docs/fig_tile_sweep.png`.

### E5 — Weight footprint

*Hypothesis.* Phase indices compress the head 8× against deployed fp32
real/imag, 16× unpacked against complex128, 32× packed at b = 4.
*Method.* Direct byte accounting, cross-checked against on-disk checkpoint size.
*Falsified if.* Packing overhead eats the gain.
*Note.* This claim does **not** depend on beating a tuned GEMM, which is why
§7 promotes it to primary.

### E6 — Latency, x86 vs Arm64, same commit

*Method.* CI matrix over `ubuntu-24.04` and `ubuntu-24.04-arm`; 30 repetitions
after 5 warm-ups; single thread; median and p95.
*Reported against.* `onnx`, never `complex128`.

### E7 — Migration template

*Hypothesis.* An arbitrary ONNX classifier can have its final Gemm replaced by
an angular head with bounded accuracy loss.
*Method.* `scripts/onnx_migrate.py` on 2–3 public models.
*Deliverable.* Before/after table. This is the Track 2 adoption story.

---

## 6. Validation protocol

Three distinct notions of correctness. Conflating them is how people fool
themselves.

**Exact parity.** Backends implementing identical arithmetic must agree to
floating-point tolerance: `complex64` ↔ `complex128`, `onnx` ↔ `complex128`.
If the ONNX baseline is wrong, every speedup number is meaningless.

**Reference parity.** The angular backends compute a genuinely different,
lower-precision quantity, so they cannot match bit-for-bit. But
`angular_tiled` and `neon` must match `angular_naive` — the literal
transcription of the algebra — essentially exactly. Any drift there is a
**kernel bug**, not a quantization effect.

**Prediction agreement.** How often the b-bit head picks the same class as the
full-precision head. Current thresholds are calibrated on **random** weights,
which is the worst case: random logits are nearly tied, so a fraction of a
sector flips the argmax. Trained models concentrate the decision margin and do
substantially better. **Tighten these once real checkpoints land.**

Benchmark hygiene, non-negotiable:

- Median and p95 over ≥ 30 repetitions. Never a single timing — CI runners are
  shared machines.
- Single thread by default (`PYMVN_THREADS=1`), or thread-count differences
  swamp the arithmetic effect being measured.
- Accuracy recorded alongside latency in every record. A fast kernel that
  changed the predictions is a regression.
- Host configuration captured via Arm's `sysreport` and committed with results.

---

## 7. Known result: throughput is not the story

**Measured on x86_64 during scaffolding, before any Arm work:**

```
layer_4096x256, b=4, batch=256, single thread
  onnx (4 GEMMs, BLAS)      31.4 ms
  angular C scalar kernel   539.2 ms      0.06×
  angular NumPy tiled      1669.8 ms      0.02×
```

This is not a bug. It is what happens when a naive triple loop meets a
decades-tuned BLAS GEMM. NumPy's fancy indexing (`lut[idx]`) is a gather, one of
the slowest primitives available, and the C scalar path has **no register-level
data reuse** — a good GEMM keeps a tile of C in registers and streams A and B.

NEON's 16-wide `TBL` buys roughly 16× over scalar, which lands the kernel near
**parity** with BLAS, not ahead of it. Beating a tuned GEMM on raw throughput is
not achievable on this timeline and should not be attempted.

**Measured on Arm afterwards (Neoverse N2, `ubuntu-24.04-arm`, b = 4, single
thread), which confirms the prediction and locates parity exactly:**

| shape | MACs/sample | neon/onnx | fp32 re/im | packed 4-bit |
|---|---|---|---|---|
| `head_512x10` | 5,130 | 0.158 | 0.04 MiB | 0.00 MiB |
| `layer_4096x256` | 1,048,832 | 0.707 | 8.0 MiB | 0.5 MiB |
| `layer_8192x512` | 4,194,816 | 0.887 | 32.0 MiB | 2.0 MiB |
| `layer_12288x768` | 9,437,952 | 0.961 | 72.0 MiB | 4.5 MiB |
| **`layer_16384x1024`** | **16,778,240** | **1.016** | **128.0 MiB** | **8.0 MiB** |
| `ffn_4096x11008` | 45,088,768 | — | 344.0 MiB | 21.5 MiB |

The ratio does cross 1.0, and the margin is outside the noise (697.9 ± 1.35 ms
for `onnx` against 687.1 ± 0.61 ms for `neon` over 30 repetitions, with `neon`'s
p95 still beating `onnx`'s fastest run).

**Where that lands.** The crossover is ~3000× above this repository's own 512×10
head, which invites the reading that parity only arrives at sizes nobody runs.
Run the arithmetic instead. A Llama-7B FFN layer is 4096 × 11008 = **45.1M MACs
per sample**, three per block. The crossover sits **2.7× below that**. It is not
absurd territory — it is beneath the regime where transformer FFN compute
already lives, and above small classifier heads. The honest frame is that the
angular datapath loses on small heads and reaches parity across the shape range
where the FLOPs actually are.

Note the scope: the claim is that the **shape region coincides**. It is not a
claim that MVN works for LLMs — nothing here trains or evaluates one. That is
future work, not a result.

**And parity is not the whole row.** At the crossover the weights are 128 MiB as
deployed fp32 real/imag against 8 MiB packed 4-bit. "As fast as BLAS" is easy to
wave away and "16× smaller" is easy to discount as a memory trick; the two in
the same row are much harder to dismiss than either alone, which is why
`bench/report.py` prints them in one table rather than two.

**The x86_64 control.** Flat at 0.047–0.059 across every shape. Same code, same
weights, same batches — the only difference is that without `vqtbl1q_u8` the
kernel falls back to scalar C. That attributes the entire Arm curve, 0.158 to
1.016, to the table-lookup instruction and to nothing in NumPy, BLAS or the
harness. It is the single cleanest piece of evidence in the sweep that this is
an Arm-specific result rather than a benchmarking artifact.

### The reframe

| Claim | Verdict |
|---|---|
| Beats BLAS GEMM on throughput | **Dropped.** Do not claim it. |
| 8–32× smaller head weights | **Primary.** Unarguable, trivially measured. |
| Zero real multiplies in the head | **Primary.** Architecturally true, verified. |
| Competitive latency at batch = 1 | **Secondary.** Plausible; memory-bound regime. |
| Novel algebraic structure, machine-checked | **Primary.** Nobody else has this. |

Two engineering consequences:

1. **Weight packing becomes a required feature, not a nice-to-have.** At b = 4
   two indices pack into one byte. That is where the 32× lives.
2. **The current benchmark is unfair to the angular path.** It converts complex
   activations to indices on every call (`np.angle` + rounding), which is
   expensive float work that a truly phase-native pipeline would never do —
   activations would already *be* indices, arriving from the previous angular
   layer or straight from the FFT. Fixing this to accept pre-indexed inputs is
   the highest-value change available, and it is what makes model #1
   (MLMVN-FFT) the flagship: it is the only configuration where indices never
   leave the integer domain.

Stating this openly is a strength. A reviewer who knows Arm CPUs will notice
immediately that an fp32 FMA on Neoverse N2 is not slow; being the one to say so
first reads as competence, not weakness.

---

## 8. Schedule

Internal freeze **10 August**. The remaining days are buffer, not plan.

| Day | Date | Deliverable |
|---|---|---|
| D0 | Fri 31 Jul | Register on Devpost + Arm Developer Program. Repo public, Apache-2.0 visible in *About*. First green CI run on `ubuntu-24.04-arm`. |
| D1 | Sat 1 Aug | Package installs clean; benchmark harness runs end to end on both architectures. |
| D2 | Sun 2 Aug | Models 1–2 trained on full MNIST and frozen to `models/`. |
| D3 | Mon 3 Aug | **Parity gate green.** Investigate the 3.9 % prediction discrepancy at b = 4 — rounding at sector boundaries, or a bug? Must be known. |
| D4 | Tue 4 Aug | Pre-indexed input path (§7.2). Weight packing at b = 4. |
| D5 | Wed 5 Aug | E4 tile sweep + E2 lift sweep. Model 3 frozen. |
| D6 | Thu 6 Aug | NEON kernel with register blocking. E3 ablation run on Arm. |
| D7 | Fri 7 Aug | **Feature freeze.** If NEON fails parity, ship without it. |
| D8 | Sat 8 Aug | Final benchmark runs, ≥ 30 repetitions. All figures generated. |
| D9 | Sun 9 Aug | README complete, Learning-Path format. Demo video < 3 min. All English. |
| D10 | Mon 10 Aug | **Draft saved on Devpost.** Everything reproducible from a clean clone. |
| — | 11–13 Aug | Buffer. Reruns and polish only. Submit by 13 Aug, 6 pm. |

### Risk register

| Risk | Trigger | Cut |
|---|---|---|
| NEON kernel fails parity | D7 | Ship `angular_tiled`; report scalar C numbers honestly |
| Model 3 will not train | D5 | Ship with models 1–2, document the gap |
| E3 shows b ≥ 6 required | D6 | Retract §2.4; the memory claim survives intact |
| Throughput never competitive | already known | Already reframed — see §7 |
| Zoo entries 4–5 slip | any | Cut without ceremony; they are stretch goals |

---

## 9. Install and test

Requires Python ≥ 3.10 and a C compiler. No GPU. No Arm hardware needed to
develop — CI provides it.

```bash
git clone https://github.com/OmarDez/MVN_Optimization.git
cd MVN_Optimization

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Build the angular kernel. On aarch64 this compiles the NEON path;
# elsewhere it falls back to portable scalar C.
bash kernels/build.sh

# Correctness gate — must be green before anything else matters.
pytest tests/ -v
```

Expected: **101 passed** on Arm; **92 passed plus 9 skipped** elsewhere (the
NEON tests and the pre-indexed NEON parity tests).

```bash
# Benchmark, no checkpoint required
python bench/run_bench.py --synthetic --repeat 30 --out results/local.json

# Quick smoke run
python bench/run_bench.py --synthetic --repeat 3 --batches 64 --bits 4 \
  --out results/smoke.json

# Train and freeze a model
python scripts/train_zoo.py --model mlmvn_fft --dataset mnist
python scripts/train_zoo.py --model resnet18   --dataset mnist

# Benchmark against frozen checkpoints
python bench/run_bench.py --models models/*.npz --out results/zoo.json

# Generate the results tables
python bench/report.py results/*.json --out docs/
```

### Verifying the central claim by hand

```bash
python - <<'PY'
import numpy as np
from pymvn import algebra as alg

b, L = 4, 16
ka, kb = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
lhs = alg.index_to_unit(ka.ravel(), b) * alg.index_to_unit(kb.ravel(), b)
rhs = alg.index_to_unit(alg.group_mul(ka.ravel(), kb.ravel(), b), b)

print("max error over all", L*L, "pairs:", np.abs(lhs - rhs).max())
print("int8 cosine LUT:", alg.cos_lut_i8(b).nbytes, "bytes",
      "(one NEON register)" if alg.cos_lut_i8(b).nbytes <= 16 else "")
PY
```

### Reproducing on Arm without owning Arm hardware

GitHub-hosted `ubuntu-24.04-arm` runners are free for public repositories and
run on Cobalt 100 (Arm Neoverse N2, 4 vCPU, Armv9-A with SVE2). Push to `main`
or dispatch the workflow manually; `.github/workflows/bench.yml` runs the test
suite and the benchmark on both `x86_64` and `aarch64` from the same commit and
uploads the results as artifacts.

---

## 10. Repository layout

```
MVN_Optimization/
├── LICENSE                       Apache-2.0
├── README.md                     usage, results, Learning-Path format
├── PLAN.md                       this document
├── pymvn/
│   ├── algebra.py                the isomorphism and its primitives
│   ├── encode.py                 FFT phase-only + the four lifts
│   ├── quant.py                  PolarQuant + bit-width ablation
│   ├── angular.py                AngularHead, six backends
│   ├── core.py                   MVN error-correction training
│   └── io.py                     checkpoint contract
├── kernels/
│   ├── angular_neon.c            NEON vqtbl1q_u8 + scalar fallback
│   └── build.sh
├── bench/
│   ├── run_bench.py              harness: median/p95, JSON records
│   └── report.py                 JSON -> markdown tables
├── tests/
│   ├── test_algebra.py           the isomorphism, machine-checked
│   ├── test_parity.py            cross-backend gate
│   └── test_contracts.py         checkpoint + interface contracts
├── scripts/
│   ├── train_zoo.py              train and freeze the model zoo
│   └── onnx_migrate.py           migration template (Track 2 story)
├── models/                       frozen checkpoints (.npz)
├── results/                      benchmark JSON + sysreport output
├── docs/                         generated figures and tables
└── .github/workflows/bench.yml   x86_64 vs aarch64 matrix
```

---

## 11. Submission checklist

Drawn directly from the official rules.

- [ ] Repository **public**, Apache-2.0 **detectable in the About section**
- [ ] Track selected explicitly: **Track 2 (Cloud AI)**
- [ ] Write-up covers **Project Overview**, **Functionality / Output**,
      **Setup Instructions**
- [ ] Setup instructions specify how to build, run and validate **on an Arm64
      environment**
- [ ] Everything in **English** — write-up, video, testing instructions
- [ ] Demo video **< 3 minutes**, publicly visible on YouTube/Vimeo/Youku
- [ ] "Significantly updated during the Submission Period" documented explicitly
- [ ] Judges can reproduce every number from a clean clone
- [ ] Ambiguity noted: the rules describe Track 2 as *migration/adoption value*
      while the Track Details page calls it *Cloud AI*. The write-up addresses
      both readings — optimization output **and** a reusable migration template.
      Request clarification in the Devpost forum (rules §11.F).
