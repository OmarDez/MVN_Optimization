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
static void angular_gemm_scalar(const uint8_t *kx, const uint8_t *kw,
                                const int8_t *lut_cos, const int8_t *lut_sin,
                                int32_t *acc_re, int32_t *acc_im,
                                int n, int k, int d, int L)
{
    const uint8_t mask = (uint8_t)(L - 1);
    for (int i = 0; i < n; ++i) {
        const uint8_t *xrow = kx + (size_t)i * d;
        for (int j = 0; j < k; ++j) {
            const uint8_t *wrow = kw + (size_t)j * d;
            int32_t sre = 0, sim = 0;
            for (int t = 0; t < d; ++t) {
                uint8_t idx = (uint8_t)((xrow[t] + wrow[t]) & mask);
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
