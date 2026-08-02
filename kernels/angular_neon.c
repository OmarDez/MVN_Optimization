/*
 * angular_gemm -- multiplier-free MVN layer for AArch64.
 *
 * Computes, for every sample i and class j:
 *
 *     acc_re[i][j] = sum_t  LUT_COS[ (kx[i][t] + kw[j][t]) mod L ]
 *     acc_im[i][j] = sum_t  LUT_SIN[ (kx[i][t] + kw[j][t]) mod L ]
 *
 * where kx and kw are b-bit phase indices and L = 2^b.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS FAST ON ARM
 * ---------------------------------------------------------------------------
 * The inner operation is NOT a multiply-accumulate. It is:
 *
 *     add -> mask -> table lookup -> widening add
 *
 * All four steps have single-instruction NEON forms, and the table lookup is
 * the interesting one. `vqtbl1q_u8` performs 16 independent byte lookups into
 * a 16-byte table in ONE instruction.
 *
 * That matters because of a fact about the model rather than the hardware:
 * the PolarQuant ablation shows b = 3..4 already recovers the unquantized
 * baseline. With b = 4 the cosine table has 16 entries -- exactly one NEON
 * register, exactly one vqtbl1q_u8 operand. The precision the network needs
 * and the width of the Arm lookup datapath coincide.
 *
 * For b > 4 the table exceeds 16 bytes and the scalar path is used instead.
 * That is a deliberate boundary, not an omission: it is the measurable cost of
 * asking for more phase resolution than the model requires.
 *
 * ---------------------------------------------------------------------------
 * NUMERICS
 * ---------------------------------------------------------------------------
 * LUT entries are int8, cos/sin scaled by 127. Accumulation is int32.
 * Worst case |acc| <= 127 * d, so d may reach ~16.9M before int32 overflows.
 * Callers divide by 127.0 to recover the float value.
 *
 * Build:  bash kernels/build.sh
 * License: Apache-2.0
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#  include <arm_neon.h>
#  define ANGULAR_NEON 1
#else
#  define ANGULAR_NEON 0
#endif

int angular_has_neon(void) { return ANGULAR_NEON; }

/* ------------------------------------------------------------------------ */
/* Portable scalar reference. Correct everywhere; used for b > 4 and on x86. */
/* ------------------------------------------------------------------------ */
/* Unrolled by two to match angular_gemm_packed's loop structure exactly. That
 * is not cosmetic: the packed kernel consumes two indices per byte and so
 * processes two features per iteration by construction. Comparing it against a
 * non-unrolled baseline would credit packing with a loop-overhead win it did
 * not earn. Same shape here, so the difference measures packing alone. */
static void angular_gemm_scalar(const uint8_t *kx, const uint8_t *kw,
                                const int8_t *lut_cos, const int8_t *lut_sin,
                                int32_t *acc_re, int32_t *acc_im,
                                int n, int k, int d, int L)
{
    const uint8_t mask = (uint8_t)(L - 1);
    const int d2 = d & ~1;
    for (int i = 0; i < n; ++i) {
        const uint8_t *xrow = kx + (size_t)i * d;
        for (int j = 0; j < k; ++j) {
            const uint8_t *wrow = kw + (size_t)j * d;
            int32_t sre = 0, sim = 0;
            for (int t = 0; t < d2; t += 2) {
                uint8_t i0 = (uint8_t)((xrow[t]     + wrow[t])     & mask);
                uint8_t i1 = (uint8_t)((xrow[t + 1] + wrow[t + 1]) & mask);
                sre += lut_cos[i0] + lut_cos[i1];
                sim += lut_sin[i0] + lut_sin[i1];
            }
            if (d2 != d) {                              /* odd tail */
                uint8_t idx = (uint8_t)((xrow[d - 1] + wrow[d - 1]) & mask);
                sre += lut_cos[idx];
                sim += lut_sin[idx];
            }
            acc_re[(size_t)i * k + j] = sre;
            acc_im[(size_t)i * k + j] = sim;
        }
    }
}

#if ANGULAR_NEON
/* ------------------------------------------------------------------------ */
/* NEON path, L <= 16: one vqtbl1q_u8 resolves 16 lookups.                  */
/* ------------------------------------------------------------------------ */
static void angular_gemm_neon16(const uint8_t *kx, const uint8_t *kw,
                                const int8_t *lut_cos, const int8_t *lut_sin,
                                int32_t *acc_re, int32_t *acc_im,
                                int n, int k, int d, int L)
{
    /* Replicate the L-entry table across all 16 lanes so that a masked index
     * always lands on a valid entry. */
    int8_t cbuf[16], sbuf[16];
    for (int t = 0; t < 16; ++t) {
        cbuf[t] = lut_cos[t % L];
        sbuf[t] = lut_sin[t % L];
    }
    const int8x16_t tc = vld1q_s8(cbuf);
    const int8x16_t ts = vld1q_s8(sbuf);
    const uint8x16_t vmask = vdupq_n_u8((uint8_t)(L - 1));

    const int dv = d & ~15;   /* vectorized span */
    const uint8_t mask = (uint8_t)(L - 1);

    for (int i = 0; i < n; ++i) {
        const uint8_t *xrow = kx + (size_t)i * d;
        for (int j = 0; j < k; ++j) {
            const uint8_t *wrow = kw + (size_t)j * d;

            int32x4_t vre = vdupq_n_s32(0);
            int32x4_t vim = vdupq_n_s32(0);

            for (int t = 0; t < dv; t += 16) {
                uint8x16_t vx = vld1q_u8(xrow + t);
                uint8x16_t vw = vld1q_u8(wrow + t);

                /* group_mul: multiplication in mu_L == addition in Z_L */
                uint8x16_t vi = vandq_u8(vaddq_u8(vx, vw), vmask);

                /* the whole point: 16 table lookups, one instruction */
                int8x16_t c = vqtbl1q_s8(tc, vi);
                int8x16_t s = vqtbl1q_s8(ts, vi);

                /* widen 8 -> 16 -> 32 and accumulate */
                int16x8_t c16 = vaddl_s8(vget_low_s8(c), vget_high_s8(c));
                int16x8_t s16 = vaddl_s8(vget_low_s8(s), vget_high_s8(s));
                vre = vaddq_s32(vre, vaddl_s16(vget_low_s16(c16), vget_high_s16(c16)));
                vim = vaddq_s32(vim, vaddl_s16(vget_low_s16(s16), vget_high_s16(s16)));
            }

            int32_t sre = vaddvq_s32(vre);
            int32_t sim = vaddvq_s32(vim);

            for (int t = dv; t < d; ++t) {          /* tail */
                uint8_t idx = (uint8_t)((xrow[t] + wrow[t]) & mask);
                sre += lut_cos[idx];
                sim += lut_sin[idx];
            }

            acc_re[(size_t)i * k + j] = sre;
            acc_im[(size_t)i * k + j] = sim;
        }
    }
}
#endif /* ANGULAR_NEON */

/* ------------------------------------------------------------------------ */
/* Packed weights: two 4-bit indices per byte (b <= 4 only).                 */
/*                                                                           */
/* Halves the weight footprint, which is the claim that does NOT depend on   */
/* beating a tuned GEMM. kw_packed holds d/2 bytes per class row; the low    */
/* nibble is the even feature, the high nibble the odd one. Unpacking costs  */
/* one AND and one shift, both cheap next to the table lookup -- but the     */
/* point is to MEASURE that, not to assume it.                               */
/*                                                                           */
/* d must be even; the caller pads. Only the scalar form is provided: the    */
/* NEON path would need a de-interleave to keep lanes aligned with the       */
/* unpacked kx, and that is a separate optimization with its own parity risk.*/
/* ------------------------------------------------------------------------ */
void angular_gemm_packed(const uint8_t *kx, const uint8_t *kw_packed,
                         const int8_t *lut_cos, const int8_t *lut_sin,
                         int32_t *acc_re, int32_t *acc_im,
                         int n, int k, int d, int L)
{
    const uint8_t mask = (uint8_t)(L - 1);
    const int dp = d / 2;
    for (int i = 0; i < n; ++i) {
        const uint8_t *xrow = kx + (size_t)i * d;
        for (int j = 0; j < k; ++j) {
            const uint8_t *wrow = kw_packed + (size_t)j * dp;
            int32_t sre = 0, sim = 0;
            for (int t = 0; t < dp; ++t) {
                const uint8_t packed = wrow[t];
                const uint8_t w_lo = packed & 0x0Fu;
                const uint8_t w_hi = (uint8_t)(packed >> 4);

                uint8_t i0 = (uint8_t)((xrow[2 * t]     + w_lo) & mask);
                uint8_t i1 = (uint8_t)((xrow[2 * t + 1] + w_hi) & mask);
                sre += lut_cos[i0] + lut_cos[i1];
                sim += lut_sin[i0] + lut_sin[i1];
            }
            acc_re[(size_t)i * k + j] = sre;
            acc_im[(size_t)i * k + j] = sim;
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Complex activations -> phase indices, without ever forming an angle.      */
/*                                                                           */
/* This is the OTHER half of the multiplier-free story. The kernel above     */
/* never multiplies, but getting into its index domain used to cost an       */
/* atan2 per activation -- and at small k that dominated: for a 512x10 head  */
/* the conversion was 48% of total time, because conversion is O(n*d) while  */
/* the layer is only O(n*k*d).                                               */
/*                                                                           */
/* No angle is needed. The sector index is a partition of the plane, and its */
/* boundaries ARE these inequalities, so the result is exact rather than     */
/* approximate:                                                              */
/*                                                                           */
/*   rotate by pi/L    so wedge boundaries land on round-to-nearest cuts     */
/*   sign(y)           -> half        (1 comparison)                         */
/*   sign(x)           -> quadrant    (1 comparison)                         */
/*   y vs tan(a)*x     -> subdivide   (1 comparison per remaining bit)       */
/*                                                                           */
/* b comparisons and b-2 rotations, all branch-free and register-resident,   */
/* against one libm atan2. The same tree in NumPy is 2.25x SLOWER than       */
/* atan2, because there each step is a separate pass over memory -- the win  */
/* only exists once the point stays in registers, which is here.             */
/*                                                                           */
/* z is interleaved re/im, i.e. the memory layout of complex128.             */
/* ------------------------------------------------------------------------ */
void angular_phase_index(const double *z, uint8_t *out, int n, int b)
{
    const int L = 1 << b;
    const double half = 3.14159265358979323846 / (double)L;
    const double ch = cos(half), sh = sin(half);

    /* Per-level rotation constants, levels j = 3..b. */
    double ta[9], ca[9], sa[9];
    for (int j = 3; j <= b && j <= 8; ++j) {
        double a = 3.14159265358979323846 / (double)(1 << (j - 1));
        ta[j] = tan(a); ca[j] = cos(a); sa[j] = sin(a);
    }

    /* Branchless throughout. The comparisons are data-dependent on essentially
     * random phases, so every branch here would be a coin flip -- and a
     * mispredict costs more than the whole tree. Selecting arithmetically keeps
     * the pipeline full and is what lets this beat atan2 at all. */
    for (int i = 0; i < n; ++i) {
        const double re = z[2 * i], im = z[2 * i + 1];
        double x = re * ch - im * sh;
        double y = re * sh + im * ch;
        int k = 0;

        if (b >= 1) {                       /* half: negate if below the axis */
            const double m = (y < 0.0) ? -1.0 : 1.0;
            k |= (y < 0.0) ? (L >> 1) : 0;
            x *= m; y *= m;
        }
        if (b >= 2) {                       /* quadrant: rotate by -pi/2 */
            const int c = (x < 0.0);
            const double nx = c ? y : x, ny = c ? -x : y;
            k |= c ? (L >> 2) : 0;
            x = nx; y = ny;
        }
        for (int j = 3; j <= b; ++j) {      /* subdivide: rotate by -pi/2^(j-1) */
            const int c = (y > ta[j] * x);
            const double nx = x * ca[j] + y * sa[j];
            const double ny = -x * sa[j] + y * ca[j];
            k |= c ? (L >> j) : 0;
            x = c ? nx : x; y = c ? ny : y;
        }
        out[i] = (uint8_t)(k & (L - 1));
    }
}

/* ------------------------------------------------------------------------ */
/* Dispatch                                                                  */
/* ------------------------------------------------------------------------ */
void angular_gemm(const uint8_t *kx, const uint8_t *kw,
                  const int8_t *lut_cos, const int8_t *lut_sin,
                  int32_t *acc_re, int32_t *acc_im,
                  int n, int k, int d, int L)
{
#if ANGULAR_NEON
    if (L <= 16) {
        angular_gemm_neon16(kx, kw, lut_cos, lut_sin, acc_re, acc_im, n, k, d, L);
        return;
    }
#endif
    angular_gemm_scalar(kx, kw, lut_cos, lut_sin, acc_re, acc_im, n, k, d, L);
}
