#include "repack-gcn.cuh"
#include "convert.cuh"
#include "quantize.cuh"

#include "ggml-backend-impl.h"
#include "mmid.cuh"
#include "unary.cuh"

#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------
// layout helpers
// ---------------------------------------------------------------------

// Sub-blocks (32 weights) per repacked row, padded by one when the
// natural count is a power of two (a power-of-two row stride aliases
// every row onto the same HBM channel: ~3x matvec penalty). Shared by
// all repacked types.
static __host__ __device__ inline int64_t repack_q4k_nsp(const int64_t ne0) {
    const int64_t n_sub = ne0 / 32;
    return (n_sub & (n_sub - 1)) == 0 ? n_sub + 1 : n_sub;
}

// Plane bytes per type:
//   Q3_K: 8 lo2 + 4 hi1 + 2 signed-scale-pair per sub-block, 2 (d) per superblock
//   Q4_K: 16 nib + 2 sc|m per sub-block, 4 (d|dmin fp16) per superblock
//   Q5_K: 16 nib + 4 qh + 2 sc|m per sub-block, 4 per superblock
//   Q6_K: 16 nib + 8 h2 + 2 signed-scale-pair per sub-block, 2 (d) per superblock
//   Q8_0: 32 qs + 2 (d fp16) per sub-block
//   Q5_1: 16 nib + 4 qh + 4 (d, m fp16) per sub-block
static inline size_t repack_gcn_nbytes(const ggml_type type, const int64_t ne0, const int64_t ne1) {
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const int64_t n_blocks = ne0 / 256;
    switch (type) {
        case GGML_TYPE_Q3_K: return (size_t) ne1 * (nsp * 14 + n_blocks * 2);
        case GGML_TYPE_Q4_K: return (size_t) ne1 * (nsp * 18 + n_blocks * 4);
        case GGML_TYPE_Q5_K: return (size_t) ne1 * (nsp * 22 + n_blocks * 4);
        case GGML_TYPE_Q6_K: return (size_t) ne1 * (nsp * 26 + n_blocks * 2);
        case GGML_TYPE_Q8_0: return (size_t) ne1 * nsp * 34;
        case GGML_TYPE_Q5_1: return (size_t) ne1 * nsp * 24;
        default:             GGML_ABORT("unsupported repack type");
    }
}

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t) {
    // 2D weights (MUL_MAT) or 3D per-expert stacks (MUL_MAT_ID)
    if ((ggml_n_dims(t) != 2 && ggml_n_dims(t) != 3) || !ggml_is_contiguous(t)) {
        return false;
    }
    switch (t->type) {
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return t->ne[0] % 256 == 0;
        case GGML_TYPE_Q8_0: {
            // Q8_0 repack is its own opt-in: the repacked MMQ wins
            // prefill big (+43% on a pure-Q8_0 0.8B) but the repacked
            // matvec loses ~6% decode to the canonical mmvq (it was
            // tuned on MoE-expert shapes in reinstinct; on-disk Q8_0 is
            // already nearly contiguous so repack buys less). Re-tune
            // before considering it for default-on.
            static const bool q8 = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q8_0");
                return e != nullptr && e[0] != '0';
            }();
            return q8 && t->ne[0] % 32 == 0;
        }
        case GGML_TYPE_Q5_1: {
            // opt-in while being validated; K only needs to be a multiple of 32 (qwen4exp ffn_down_exps has K=640)
            static const bool q51 = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q5_1");
                return e != nullptr && e[0] != '0';
            }();
            return q51 && t->ne[0] % 32 == 0;
        }
        default:             return false;
    }
}

// ---------------------------------------------------------------------
// host-side repack (one-shot at weight upload)
// ---------------------------------------------------------------------

// ggml-quants.c's get_scale_min_k4: unpack sub-block j's 6-bit (sc, m)
// from the 12-byte packed scales array.
static inline void repack_get_scale_min_k4(const int j, const uint8_t * q, uint8_t * sc, uint8_t * m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >>   4) | ((q[j    ] >> 6) << 4);
    }
}

static void repack_q4k_host(const block_q4_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    // The padding sub-block (when nsp != ne0/32) must read as zero
    // weights with zero scales so the kernel can include it harmlessly.
    memset(dst, 0, nib_len + sm_len + (size_t) ne1 * n_blocks * 4);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q4_K * b = &blocks[row * n_blocks + blk];

            // superblock plane: raw fp16 d, dmin per 256 weights
            // (block_q4_K starts with d at byte 0, dmin at byte 2 —
            // guaranteed by the ggml-common.h size/layout asserts)
            uint8_t * dd = dst + nib_len + sm_len + (size_t)(row * n_blocks + blk) * 4;
            memcpy(dd, b, 4);

            for (int s = 0; s < 8; s++) {
                const int64_t gsb = blk * 8 + s; // sub-block index within the row

                // this sub-block's 32 nibble weights: qs bytes (s/2)*32..+32,
                // even sub-blocks take low nibbles, odd take high
                const uint8_t * qs = b->qs + (s >> 1) * 32;
                uint8_t w[32];
                if ((s & 1) == 0) {
                    for (int k = 0; k < 32; k++) { w[k] = qs[k] & 0x0F; }
                } else {
                    for (int k = 0; k < 32; k++) { w[k] = qs[k] >> 4; }
                }

                // nibble plane: byte 4j+b = w[4j+b] | (w[16+4j+b] << 4),
                // so uint32 j feeds dp4a with weights 4j..4j+3 / 16+4j..+3
                uint8_t * nib = dst + (size_t)(row * nsp + gsb) * 16;
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        nib[j * 4 + bb] = w[4 * j + bb] | (w[16 + 4 * j + bb] << 4);
                    }
                }

                // scale plane: 6-bit sc then m as two u8
                uint8_t sc, m;
                repack_get_scale_min_k4(s, b->scales, &sc, &m);
                uint8_t * sm = dst + nib_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = sc;
                sm[1] = m;
            }
        }
    }
}

// Q5_K: like Q4_K plus a qh plane — per sub-block one u32 whose bit
// 4g+b is the 5th bit of weight b of dp4a group g.
static void repack_q5k_host(const block_q5_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  qh_len   = (size_t) ne1 * nsp * 4;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    memset(dst, 0, nib_len + qh_len + sm_len + (size_t) ne1 * n_blocks * 4);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q5_K * b = &blocks[row * n_blocks + blk];

            uint8_t * dd = dst + nib_len + qh_len + sm_len + (size_t)(row * n_blocks + blk) * 4;
            memcpy(dd, b, 4); // fp16 d, dmin lead the block

            for (int s = 0; s < 8; s++) {
                const int64_t gsb = blk * 8 + s;
                const uint8_t * qs = b->qs + (s >> 1) * 32;

                uint8_t w[32], hb[32];
                for (int k = 0; k < 32; k++) {
                    w[k]  = ((s & 1) == 0) ? (qs[k] & 0x0F) : (qs[k] >> 4);
                    hb[k] = (b->qh[k] >> s) & 1;
                }

                uint8_t * nib = dst + (size_t)(row * nsp + gsb) * 16;
                uint32_t qh_packed = 0;
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        nib[j * 4 + bb] = w[4 * j + bb] | (w[16 + 4 * j + bb] << 4);
                        // dp4a group 2j holds weights 4j+bb, group 2j+1 holds 16+4j+bb
                        qh_packed |= (uint32_t) hb[4 * j + bb]      << (4 * (2 * j)     + bb);
                        qh_packed |= (uint32_t) hb[16 + 4 * j + bb] << (4 * (2 * j + 1) + bb);
                    }
                }
                memcpy(dst + nib_len + (size_t)(row * nsp + gsb) * 4, &qh_packed, 4);

                uint8_t sc, m;
                repack_get_scale_min_k4(s, b->scales, &sc, &m);
                uint8_t * sm = dst + nib_len + qh_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = sc;
                sm[1] = m;
            }
        }
    }
}

// Q6_K: nibble plane + 8-byte h2 plane (the 6-bit quant's high pair per
// weight, 2 bits at position 2b of byte g) + signed per-16-weight scale
// pairs + d-only superblock plane.
static void repack_q6k_host(const block_q6_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  h2_len   = (size_t) ne1 * nsp * 8;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    memset(dst, 0, nib_len + h2_len + sm_len + (size_t) ne1 * n_blocks * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q6_K * b = &blocks[row * n_blocks + blk];

            memcpy(dst + nib_len + h2_len + sm_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);

            for (int s = 0; s < 8; s++) {
                const int64_t gsb  = blk * 8 + s;
                const int     chunk = s / 4;
                const int     quad  = s % 4;
                const int     ql_off = chunk * 64;
                const int     qh_off = chunk * 32;

                uint8_t lo[32], hi[32];
                for (int k = 0; k < 32; k++) {
                    const uint8_t qh = b->qh[qh_off + k];
                    switch (quad) {
                        case 0:  lo[k] = b->ql[ql_off + k]      & 0x0F; hi[k] =  qh       & 3; break;
                        case 1:  lo[k] = b->ql[ql_off + k + 32] & 0x0F; hi[k] = (qh >> 2) & 3; break;
                        case 2:  lo[k] = b->ql[ql_off + k]      >> 4;   hi[k] = (qh >> 4) & 3; break;
                        default: lo[k] = b->ql[ql_off + k + 32] >> 4;   hi[k] = (qh >> 6) & 3; break;
                    }
                }

                uint8_t * nib = dst + (size_t)(row * nsp + gsb) * 16;
                uint8_t h2p[8] = {};
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        nib[j * 4 + bb] = lo[4 * j + bb] | (lo[16 + 4 * j + bb] << 4);
                        h2p[2 * j]     |= hi[4 * j + bb]      << (2 * bb);
                        h2p[2 * j + 1] |= hi[16 + 4 * j + bb] << (2 * bb);
                    }
                }
                memcpy(dst + nib_len + (size_t)(row * nsp + gsb) * 8, h2p, 8);

                uint8_t * sm = dst + nib_len + h2_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = (uint8_t) b->scales[chunk * 8 + quad * 2];
                sm[1] = (uint8_t) b->scales[chunk * 8 + quad * 2 + 1];
            }
        }
    }
}

// Q3_K: the 3-bit quant stays sub-nibble to fit VRAM. A lo2 plane (low 2
// bits, packed like Q6_K's h2) and a hi1 plane (high bit, packed like
// Q5_K's qh) reconstruct q3 = lo2 | (hbit << 2) at compute. Symmetric
// like Q6_K (bias 4) with a signed per-16-weight scale pair (unpacked
// 6-bit scale minus 32) and a d-only superblock plane.
static void repack_q3k_host(const block_q3_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  lo2_len  = (size_t) ne1 * nsp * 8;
    const size_t  hi1_len  = (size_t) ne1 * nsp * 4;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    memset(dst, 0, lo2_len + hi1_len + sm_len + (size_t) ne1 * n_blocks * 2);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q3_K * b = &blocks[row * n_blocks + blk];

            memcpy(dst + lo2_len + hi1_len + sm_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);

            // ggml-quants.c dequantize_row_q3_K: unpack the 16 6-bit scales
            uint32_t aux[4];
            memcpy(aux, b->scales, 12);
            const uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = ( aux[0]       & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = ( aux[1]       & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const uint8_t * sc6 = (const uint8_t *) aux;

            for (int s = 0; s < 8; s++) {
                const int64_t gsb   = blk * 8 + s;
                const int     n     = s >> 2;
                const int     shift = 2 * (s & 3);
                const uint8_t * qs  = b->qs + n * 32;

                uint8_t  lo2[32], hb[32];
                for (int k = 0; k < 32; k++) {
                    lo2[k] = (qs[k] >> shift) & 3;
                    hb[k]  = (b->hmask[k] >> s) & 1;
                }

                // lo2 plane: byte 2j holds the low-half group j (weights
                // 4j..), byte 2j+1 the high-half (weights 16+4j..), weight
                // b at bits 2b..2b+1 -- the Q6_K h2 packing.
                uint8_t lo2p[8] = {};
                uint32_t hi1p = 0;
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        lo2p[2 * j]     |= lo2[4 * j + bb]      << (2 * bb);
                        lo2p[2 * j + 1] |= lo2[16 + 4 * j + bb] << (2 * bb);
                        hi1p |= (uint32_t) hb[4 * j + bb]      << (8 * j + bb);
                        hi1p |= (uint32_t) hb[16 + 4 * j + bb] << (8 * j + 4 + bb);
                    }
                }
                memcpy(dst + (size_t)(row * nsp + gsb) * 8, lo2p, 8);
                memcpy(dst + lo2_len + (size_t)(row * nsp + gsb) * 4, &hi1p, 4);

                uint8_t * sm = dst + lo2_len + hi1_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = (uint8_t) (int8_t) ((int) sc6[2 * s]     - 32);
                sm[1] = (uint8_t) (int8_t) ((int) sc6[2 * s + 1] - 32);
            }
        }
    }
}

// Q8_0: two planes — 32 aligned qs bytes per sub-block, then the fp16
// d-scales as their own stream. Same bytes as on-disk modulo padding;
// the win is alignment (one i32 load per sdot4 instead of two
// uint16 loads OR-shifted around the on-disk 2-byte offset).
static void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  qs_len   = (size_t) ne1 * nsp * 32;

    memset(dst, 0, qs_len + (size_t) ne1 * nsp * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t)(row * nsp + blk) * 32, b->qs, 32);
            memcpy(dst + qs_len + (size_t)(row * nsp + blk) * 2, &b->d, 2);
        }
    }
}

// Q5_1: nibble plane (qs as-is), qh plane in the Q5_K dp4a-group bit order, (d, m) plane.
static void repack_q5_1_host(const block_q5_1 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  qh_len   = (size_t) ne1 * nsp * 4;

    memset(dst, 0, nib_len + qh_len + (size_t) ne1 * nsp * 4);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q5_1 * b = &blocks[row * n_blocks + blk];
            const size_t idx = (size_t)(row * nsp + blk);

            memcpy(dst + idx * 16, b->qs, 16);

            uint32_t qh_raw;
            memcpy(&qh_raw, b->qh, 4);
            uint32_t qh_packed = 0;
            for (int j = 0; j < 4; j++) {
                for (int bb = 0; bb < 4; bb++) {
                    // dp4a group 2j holds weights 4j+bb, group 2j+1 holds 16+4j+bb
                    qh_packed |= ((qh_raw >> (4 * j + bb))      & 1u) << (4 * (2 * j)     + bb);
                    qh_packed |= ((qh_raw >> (16 + 4 * j + bb)) & 1u) << (4 * (2 * j + 1) + bb);
                }
            }
            memcpy(dst + nib_len + idx * 4, &qh_packed, 4);
            memcpy(dst + nib_len + qh_len + idx * 4, &b->dm, 4);
        }
    }
}

// ---------------------------------------------------------------------
// kernels (GCN only — guarded so non-HIP / non-GCN builds still compile)
// ---------------------------------------------------------------------

// --- MUL_MAT_ID support -----------------------------------------------
// Expert routing comes compacted from ggml_cuda_launch_mm_ids_helper:
// assignment index a in [0, n_assign) is expert-sorted; expert_bounds
// gives each expert's [start, end) range; ids_src1[a] is the flat
// column index into the naturally-ordered activation buffer; ids_dst[a]
// is the flat destination column. Weights for expert e live at
// wbase + e * expert_stride (per-expert repacked slabs, identical
// layout to the 2D case).

// tile_off[e] = exclusive prefix sum of per-expert token-tile counts
// (BN-sized tiles), tile_off[n_expert] = total; tile_expert[tile] = the
// expert owning that tile. One 1024-thread block, Hillis-Steele scan
// over chunks of 1024 experts with a running carry.
template <int BN>
static __global__ void __launch_bounds__(1024, 1) repack_tile_map(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_off,
        int32_t * __restrict__ tile_expert, const int n_expert) {
    __shared__ int s[1024];
    const int t = threadIdx.x;
    int carry = 0;
    for (int e0 = 0; e0 < n_expert; e0 += 1024) {
        const int e   = e0 + t;
        const int cnt = e < n_expert ? (expert_bounds[e + 1] - expert_bounds[e] + BN - 1) / BN : 0;
        s[t] = cnt;
        __syncthreads();
        for (int off = 1; off < 1024; off <<= 1) {
            const int v = t >= off ? s[t - off] : 0;
            __syncthreads();
            s[t] += v;
            __syncthreads();
        }
        const int excl = carry + s[t] - cnt;
        if (e < n_expert) {
            tile_off[e] = excl;
            for (int k = 0; k < cnt; k++) {
                tile_expert[excl + k] = e;
            }
        }
        carry += s[1023];
        __syncthreads();
    }
    if (t == 0) {
        tile_off[n_expert] = carry;
    }
}

// Repacked Q4_K matvec. Block = 256 threads = 4 wave64s; each wave
// computes ROWS=2 output rows; lane l streams sub-block l, l+64, ... —
// consecutive lanes read consecutive 16-byte chunks, a fully-coalesced
// sweep of the nibble plane.
//
//   dot = sum_sub [ (d*sc) * dx * <nibbles . q8> - (dmin*m) * sx ]
//
// with (dx, sx) = block_q8_1.ds — sx is dx * sum(q8), which is exactly
// the dequantized sub-block sum the min-term needs (same contract as
// vec_dot_q4_K_q8_1).
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q4k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);
    const uint32_t n_super = n_sub >> 3;

    float acc[ROWS] = {0.0f, 0.0f};

    // ID path (experts) uses 16-weight half-sub-block units: expert
    // tensors are small-K (down-proj K=768 -> 24 sub-blocks for 64
    // lanes) and full units leave most of the wave idle. The -deff*sx
    // min term is applied by the even half only.
    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub;
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }

            const uint4    q  = nib[(size_t) row * nsp + sb];
            const uint16_t sm = smp[(size_t) row * nsp + sb];
            const uint32_t dd = ddp[(size_t) row * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))
                               * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                               * (float)(sm >> 8);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
            int idot = 0;
#pragma unroll
            for (int j = j0; j < j1; j++) {
                idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
            }
            acc[r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Spread 4 bits (b0..b3) to bit 4 of bytes 0..3 — positions the Q5_K
// high bit above the dp4a nibble lanes.
static __device__ __forceinline__ uint32_t repack_spread4(const uint32_t h) {
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

// Spread four 2-bit fields (weight b at bits 2b..2b+1) to bits 4..5 of
// bytes 0..3 — the Q6_K quant's high pair.
static __device__ __forceinline__ uint32_t repack_spread2(const uint32_t h) {
    return ((h & 0x03u) << 4) | ((h & 0x0Cu) << 10)
         | ((h & 0x30u) << 16) | ((h & 0xC0u) << 22);
}

// Q3_K reconstruction: four 2-bit fields to bits 0..1 of bytes 0..3 (the
// low pair) and four 1-bit fields to bit 2 of bytes 0..3 (the high bit).
static __device__ __forceinline__ uint32_t repack_spread2_lo(const uint32_t h) {
    return (h & 0x03u) | ((h & 0x0Cu) << 6)
         | ((h & 0x30u) << 12) | ((h & 0xC0u) << 18);
}
static __device__ __forceinline__ uint32_t repack_spread1_hi(const uint32_t h) {
    return ((h & 1u) << 2) | ((h & 2u) << 9) | ((h & 4u) << 16) | ((h & 8u) << 23);
}

// Q5_K repacked matvec — Q4_K's shape plus the qh plane OR-ed onto the
// nibbles before each dp4a.
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q5k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // see Q4_K note
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx = (size_t) row * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qh  = qhp[idx];
            const uint16_t sm  = smp[idx];
            const uint32_t dd  = ddp[(size_t) row * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))
                               * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                               * (float)(sm >> 8);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
            int idot = 0;
#pragma unroll
            for (int j = j0; j < j1; j++) {
                const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu)
                    | repack_spread4((qh >> (8 * j))     & 0xFu);
                const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu)
                    | repack_spread4((qh >> (8 * j + 4)) & 0xFu);
                idot = ggml_cuda_dp4a((int) lo, xq32[j],     idot);
                idot = ggml_cuda_dp4a((int) hi, xq32[j + 4], idot);
            }
            acc[r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q5_1 repacked matvec: Q5_K's shape with a per-sub-block (d, m); the min term adds (x = d*q + m).
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q5_1_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(wbase + (size_t) ne1 * nsp * 16);
    const half2    * dmp = reinterpret_cast<const half2 *>(wbase + (size_t) ne1 * nsp * 20);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // half-sub-block units for the per-slot case, see Q4_K
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx = (size_t) row * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qh  = qhp[idx];
            const float2   dm  = __half22float2(dmp[idx]);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
            int idot = 0;
#pragma unroll
            for (int j = j0; j < j1; j++) {
                const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j))     & 0xFu);
                const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j + 4)) & 0xFu);
                idot = ggml_cuda_dp4a((int) lo, xq32[j],     idot);
                idot = ggml_cuda_dp4a((int) hi, xq32[j + 4], idot);
            }
            acc[r] += dm.x * dx * (float) idot + (half == 0 ? dm.y * sx : 0.0f);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q5_1 repacked matvec for short rows (ne0 <= 1024, e.g. qwen4exp ffn_down_exps with K = 640): one lane
// per sub-block, SEG lanes per row. The row-per-wave kernel above uses half-sub-block units that each
// load the whole 16-byte nibble chunk (twice the weight traffic) and leaves lanes idle at this K.
template <int SEG, bool HAS_IDS>
static __global__ void __launch_bounds__(256) mul_mat_vec_q5_1_repacked_seg(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
    } else {
        GGML_UNUSED_VARS(ids_src1, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS_PER_BLOCK = 256 / SEG;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(wbase + (size_t) ne1 * nsp * 16);
    const half2    * dmp = reinterpret_cast<const half2 *>(wbase + (size_t) ne1 * nsp * 20);

    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / SEG;
    const uint32_t sb  = threadIdx.x % SEG;

    float acc = 0.0f;
    if (row < ne1 && sb < n_sub) {
        const size_t   idx = (size_t) row * nsp + sb;
        const uint4    q   = nib[idx];
        const uint32_t qh  = qhp[idx];
        const float2   dm  = __half22float2(dmp[idx]);
        const block_q8_1 * xb = xq + sb;
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);
        const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
        int idot = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j))     & 0xFu);
            const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j + 4)) & 0xFu);
            idot = ggml_cuda_dp4a((int) lo, xq32[j],     idot);
            idot = ggml_cuda_dp4a((int) hi, xq32[j + 4], idot);
        }
        acc = dm.x * __low2float(xb->ds) * (float) idot + dm.y * __high2float(xb->ds);
    }
    acc = warp_reduce_sum<SEG>(acc);
    if (sb == 0 && row < ne1) {
        y[row] = acc;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

template <bool HAS_IDS>
static void launch_mul_mat_vec_q5_1_repacked_seg(
        const uint8_t * w, const block_q8_1 * xq, float * y, const int64_t ne00, const int64_t ne01,
        const int64_t n_slots, const int32_t * ids, const size_t expert_stride, const uint32_t xs_id,
        const uint32_t dst_s1, cudaStream_t stream) {
    const int64_t n_sub = ne00 / 32;
    auto launch = [&](auto seg_c) {
        constexpr int SEG = decltype(seg_c)::value;
        const dim3 grid((ne01 + 256 / SEG - 1) / (256 / SEG), n_slots, 1);
        mul_mat_vec_q5_1_repacked_seg<SEG, HAS_IDS><<<grid, 256, 0, stream>>>(
            w, xq, y, (uint32_t) ne00, (uint32_t) ne01, ids, expert_stride, xs_id, dst_s1);
    };
    if (n_sub <= 8) {
        launch(std::integral_constant<int, 8>{});
    } else if (n_sub <= 16) {
        launch(std::integral_constant<int, 16>{});
    } else {
        launch(std::integral_constant<int, 32>{});
    }
}

// Q6_K repacked matvec. Symmetric quant (value = q-32); the offset is
// folded out via activation half-sums: sum (q-32)x = sum qx - 32 sum x.
// Two signed scales per sub-block, one per 16 weights.
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q6k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * h2p = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 2);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // see Q4_K note
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const int hj0 = HAS_IDS ? (int)(half * 2) : 0;
        const int hj1 = HAS_IDS ? hj0 + 2         : 4;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

        // per-half activation sums: the -32 fold splits with them
        int xis0 = 0, xis1 = 0;
#pragma unroll
        for (int j = hj0; j < hj1; j++) {
            xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
            xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
        }

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx  = (size_t) row * nsp + sb;
            const uint4    q    = nib[idx];
            const uint32_t h2lo = h2p[idx * 2];
            const uint32_t h2hi = h2p[idx * 2 + 1];
            const uint16_t sm     = smp[idx];
            const uint16_t d_bits = ddp[(size_t) row * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            const float dsc_lo = d * (float)(int)(int8_t)(sm & 0xFFu);
            const float dsc_hi = d * (float)(int)(int8_t)(sm >> 8);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            int idot0 = 0, idot1 = 0;
#pragma unroll
            for (int j = hj0; j < hj1; j++) {
                const uint32_t ge = 2 * j;
                const uint32_t go = 2 * j + 1;
                const uint32_t he = ((ge < 4 ? h2lo : h2hi) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t ho = ((go < 4 ? h2lo : h2hi) >> (8 * (go & 3))) & 0xFFu;
                const uint32_t q6lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread2(he);
                const uint32_t q6hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread2(ho);
                idot0 = ggml_cuda_dp4a((int) q6lo, xq32[j],     idot0);
                idot1 = ggml_cuda_dp4a((int) q6hi, xq32[j + 4], idot1);
            }
            acc[r] += dsc_lo * dx * (float)(idot0 - 32 * xis0)
                    + dsc_hi * dx * (float)(idot1 - 32 * xis1);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q3_K repacked matvec. Like Q6_K but the quant is 2-bit lo + 1-bit hi
// (no 4-bit nibble plane); reconstruct q3 = lo2 | (hbit << 2) per group.
// Symmetric with bias 4.
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q3k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint2    * lo2p = reinterpret_cast<const uint2 *>(wbase);
    const uint32_t * hi1p = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 8);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // see Q4_K note
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const int hj0 = HAS_IDS ? (int)(half * 2) : 0;
        const int hj1 = HAS_IDS ? hj0 + 2         : 4;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

        int xis0 = 0, xis1 = 0;
#pragma unroll
        for (int j = hj0; j < hj1; j++) {
            xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
            xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
        }

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx    = (size_t) row * nsp + sb;
            const uint2    lo2v   = lo2p[idx];
            const uint32_t qh     = hi1p[idx];
            const uint16_t sm     = smp[idx];
            const uint16_t d_bits = ddp[(size_t) row * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            const float dsc_lo = d * (float)(int)(int8_t)(sm & 0xFFu);
            const float dsc_hi = d * (float)(int)(int8_t)(sm >> 8);

            const uint32_t lo2lo = lo2v.x, lo2hi = lo2v.y;
            int idot0 = 0, idot1 = 0;
#pragma unroll
            for (int j = hj0; j < hj1; j++) {
                const uint32_t ge = 2 * j;
                const uint32_t go = 2 * j + 1;
                const uint32_t lb = ((ge < 4 ? lo2lo : lo2hi) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t hb = ((go < 4 ? lo2lo : lo2hi) >> (8 * (go & 3))) & 0xFFu;
                const uint32_t q3lo = repack_spread2_lo(lb) | repack_spread1_hi((qh >> (8 * j))     & 0xFu);
                const uint32_t q3hi = repack_spread2_lo(hb) | repack_spread1_hi((qh >> (8 * j + 4)) & 0xFu);
                idot0 = ggml_cuda_dp4a((int) q3lo, xq32[j],     idot0);
                idot1 = ggml_cuda_dp4a((int) q3hi, xq32[j + 4], idot1);
            }
            acc[r] += dsc_lo * dx * (float)(idot0 - 4 * xis0)
                    + dsc_hi * dx * (float)(idot1 - 4 * xis1);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q8_0 repacked matvec — NWAVES wave64s per block, ROWS output rows
// per wave. The original reinstinct tuning used single-wave blocks
// (NWAVES=1) for its MoE-expert shapes; small dense models want the
// 4-wave shape the K-quant matvecs use (NWAVES=4). ROWS=1 doubles the
// wavefront count and wins at out_dim >= 4096 where ROWS=2 leaves too
// few wavefront generations in flight to sustain HBM bandwidth.
// Repacked Q8_0 matvec for short rows (ne0 <= 1024): one lane per 32-weight sub-block and SEG lanes
// per row, so a wave covers 64/SEG rows. The one-row-per-wave kernel below leaves most lanes idle
// when a row has few sub-blocks (qwen4exp hc up-projections, K = 320: 10 of 64 lanes busy, ~90 GB/s).
template <int SEG, bool HAS_IDS>
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_seg(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
    } else {
        GGML_UNUSED_VARS(ids_src1, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS_PER_BLOCK = 256 / SEG;
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;

    const int4     * qs4     = reinterpret_cast<const int4 *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);

    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / SEG;
    const uint32_t sb  = threadIdx.x % SEG;

    float acc = 0.0f;
    if (row < ne1 && sb < n_blocks) {
        const block_q8_1 * xb = xq + sb;
        const int4 * x4 = reinterpret_cast<const int4 *>(xb->qs);
        const int4 w0 = qs4[((size_t) row * nsp + sb) * 2 + 0];
        const int4 w1 = qs4[((size_t) row * nsp + sb) * 2 + 1];
        const int4 a0 = x4[0];
        const int4 a1 = x4[1];
        int idot = 0;
        idot = ggml_cuda_dp4a(w0.x, a0.x, idot);
        idot = ggml_cuda_dp4a(w0.y, a0.y, idot);
        idot = ggml_cuda_dp4a(w0.z, a0.z, idot);
        idot = ggml_cuda_dp4a(w0.w, a0.w, idot);
        idot = ggml_cuda_dp4a(w1.x, a1.x, idot);
        idot = ggml_cuda_dp4a(w1.y, a1.y, idot);
        idot = ggml_cuda_dp4a(w1.z, a1.z, idot);
        idot = ggml_cuda_dp4a(w1.w, a1.w, idot);
        const uint16_t db = d_plane[(size_t) row * nsp + sb];
        acc = __half2float(*reinterpret_cast<const __half *>(&db)) * __low2float(xb->ds) * (float) idot;
    }
    acc = warp_reduce_sum<SEG>(acc);
    if (sb == 0 && row < ne1) {
        y[row] = acc;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Repacked Q8_0 matvec for few long rows (non-ID): one 256-thread workgroup per row splits K over
// all four waves and reduces through LDS. The row-per-wave kernels launch too few waves to load the
// GPU when there are only a few hundred rows (qwen4exp hc down-projections, 320 rows x K = 10240).
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_splitk(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);

    const uint32_t row = blockIdx.x;
    float acc = 0.0f;
    // work unit: a 16-weight half sub-block, as in the row-per-wave kernel
    for (uint32_t hb = threadIdx.x; hb < n_blocks * 2; hb += 256) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const block_q8_1 * xb = xq + sb;
        const int * xq32  = reinterpret_cast<const int *>(xb->qs) + half * 4;
        const int * w_int = qs_int + ((size_t) row * nsp + sb) * 8 + half * 4;
        int idot = 0;
#pragma unroll
        for (int g = 0; g < 4; g++) {
            idot = ggml_cuda_dp4a(w_int[g], xq32[g], idot);
        }
        const uint16_t db = d_plane[(size_t) row * nsp + sb];
        acc += __half2float(*reinterpret_cast<const __half *>(&db)) * __low2float(xb->ds) * (float) idot;
    }
    acc = warp_reduce_sum<64>(acc);
    __shared__ float part[4];
    if ((threadIdx.x & 63) == 0) {
        part[threadIdx.x >> 6] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        y[row] = part[0] + part[1] + part[2] + part[3];
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

template <bool HAS_IDS>
static void launch_mul_mat_vec_q8_0_repacked_seg(
        const uint8_t * w, const block_q8_1 * xq, float * y, const int64_t ne00, const int64_t ne01,
        const int64_t n_slots, const int32_t * ids, const size_t expert_stride, const uint32_t xs_id,
        const uint32_t dst_s1, cudaStream_t stream) {
    const int64_t n_blocks = ne00 / 32;
    auto launch = [&](auto seg_c) {
        constexpr int SEG = decltype(seg_c)::value;
        const dim3 grid((ne01 + 256 / SEG - 1) / (256 / SEG), n_slots, 1);
        mul_mat_vec_q8_0_repacked_seg<SEG, HAS_IDS><<<grid, 256, 0, stream>>>(
            w, xq, y, (uint32_t) ne00, (uint32_t) ne01, ids, expert_stride, xs_id, dst_s1);
    };
    if (n_blocks <= 8) {
        launch(std::integral_constant<int, 8>{});
    } else if (n_blocks <= 16) {
        launch(std::integral_constant<int, 16>{});
    } else {
        launch(std::integral_constant<int, 32>{});
    }
}

template <int ROWS, int NWAVES, bool HAS_IDS>
static __global__ void mul_mat_vec_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;

    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 32);

    const int wave = threadIdx.x >> 6;
    const int row0 = blockIdx.x * (ROWS * NWAVES) + wave * ROWS;
    const int lane = threadIdx.x & 63;

    float acc[ROWS] = {};

    // Work unit: a 16-weight half sub-block (4 sdot4s). At small ne0 a
    // full-sub-block unit leaves lanes idle (ne0=1024 -> 32 sub-blocks
    // for 64 lanes); halves keep the wave full down to ne0=1024.
    // (8-weight quarters were tried and regress: 4x the d-plane traffic
    // outweighs the extra balance.)
    const uint32_t n_half = n_blocks * 2;
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const int      * w_int = qs_int + ((size_t) row * nsp + sb) * 8 + half * 4;
            const uint16_t   db    = d_plane[(size_t) row * nsp + sb];
            const float      dw    = __half2float(*reinterpret_cast<const __half *>(&db));

            int idot = 0;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                idot = ggml_cuda_dp4a(w_int[g], xq32[g], idot);
            }
            acc[r] += dw * dx * (float) idot;
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}


// Fused gate+up Q4_K matvec with GLU epilogue. Walks BOTH weight slabs
// in one sub-block loop (one activation read, one launch) and writes
// y[row] = glu(gate_dot) * up_dot — replacing two matvec launches plus
// an elementwise GLU op. This is what canonical mmvq fuses too; without
// it the repacked MoE decode pays ~2x the launches (measured -7% on
// 35B-A3B). Used for both dense MUL_MAT (ids == nullptr) and
// MUL_MAT_ID decode. ID path uses half-sub-block units (small-K expert
// tensors; min term on the even half).
template <bool HAS_IDS, int ROWS = 2, int MIN_BLOCKS = 1>
static __global__ void __launch_bounds__(256, MIN_BLOCKS) mul_mat_vec_q4k_repacked_glu(
        const uint8_t * __restrict__ wup, const uint8_t * __restrict__ wgate,
        const block_q8_1 * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne0, const uint32_t ne1, const int glu_op,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y; // slot index; see direct-map note above
        const uint32_t e = (uint32_t) ids_src1[a];
        wup   += e * expert_stride;
        wgate += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    // ROWS: template parameter
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint8_t * wb[2] = { wup, wgate };
    float acc[2][ROWS] = {};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub;
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const uint4    * nib = reinterpret_cast<const uint4 *>(wb[w2]);
            const uint16_t * smp = reinterpret_cast<const uint16_t *>(
                wb[w2] + (size_t) ne1 * nsp * 16);
            const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
                wb[w2] + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row0 + r;
                if (row >= (int) ne1) {
                    continue;
                }
                const uint4    q  = nib[(size_t) row * nsp + sb];
                const uint16_t sm = smp[(size_t) row * nsp + sb];
                const uint32_t dd = ddp[(size_t) row * n_super + (sb >> 3)];
                const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
                const uint16_t dmin_bits = (uint16_t)(dd >> 16);
                const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))
                                   * (float)(sm & 0xFFu);
                const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                                   * (float)(sm >> 8);

                const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
                const int j0 = HAS_IDS ? (int)(half * 2) : 0;
                const int j1 = HAS_IDS ? j0 + 2          : 4;
                int idot = 0;
#pragma unroll
                for (int j = j0; j < j1; j++) {
                    idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                    idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
                }
                acc[w2][r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float up_v   = warp_reduce_sum<64>(acc[0][r]);
        const float gate_v = warp_reduce_sum<64>(acc[1][r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            const float g = glu_op == (int) GGML_GLU_OP_SWIGLU
                ? ggml_cuda_op_silu_single(gate_v)
                : ggml_cuda_op_gelu_single(gate_v);
            y[row0 + r] = g * up_v;
        }
    }
#else
    GGML_UNUSED_VARS(wup, wgate, xq, y, ne0, ne1, glu_op, ids_src1, ids_dst,
                     expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Dense Q8_0 gate+up matvec with GLU epilogue (the shared-expert FFN):
// mul_mat_vec_q8_0_repacked<ROWS, 4> walking both weight slabs per unit.
template <int ROWS>
static __global__ void mul_mat_vec_q8_0_repacked_glu(
        const uint8_t * __restrict__ wup, const uint8_t * __restrict__ wgate,
        const block_q8_1 * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne0, const uint32_t ne1, const int glu_op) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;

    const uint8_t * wb[2] = { wup, wgate };

    const int wave = threadIdx.x >> 6;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const int lane = threadIdx.x & 63;

    float acc[2][ROWS] = {};

    const uint32_t n_half = n_blocks * 2;
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;

#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const int      * qs_int  = reinterpret_cast<const int *>(wb[w2]);
            const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wb[w2] + (size_t) ne1 * nsp * 32);
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row0 + r;
                if (row >= (int) ne1) {
                    continue;
                }
                const int      * w_int = qs_int + ((size_t) row * nsp + sb) * 8 + half * 4;
                const uint16_t   db    = d_plane[(size_t) row * nsp + sb];
                const float      dw    = __half2float(*reinterpret_cast<const __half *>(&db));

                int idot = 0;
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    idot = ggml_cuda_dp4a(w_int[g], xq32[g], idot);
                }
                acc[w2][r] += dw * dx * (float) idot;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float up_v   = warp_reduce_sum<64>(acc[0][r]);
        const float gate_v = warp_reduce_sum<64>(acc[1][r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            const float g = glu_op == (int) GGML_GLU_OP_SWIGLU
                ? ggml_cuda_op_silu_single(gate_v)
                : ggml_cuda_op_gelu_single(gate_v);
            y[row0 + r] = g * up_v;
        }
    }
#else
    GGML_UNUSED_VARS(wup, wgate, xq, y, ne0, ne1, glu_op);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// int8 MMQ tile GEMM straight from the repacked planes (prefill path).
// Y[tok, row] = Xq8[tok, :] . W[row, :] without dequantizing W.
//
// A workgroup (256 threads as a 16x16 grid) computes a BM x BN output
// tile (BM = 64 weight rows, BN = 64 tokens), walking the contraction
// in BK = 4 sub-block chunks staged through LDS. Thread (tx,ty) owns a
// strided 4x4 register micro-tile (rows ty, ty+16, ..., tokens tx,
// tx+16, ...) so a wavefront's 16 token reads land on 16 distinct LDS
// banks (block_q8_1 stride is 36 B = 9 words; gcd(9,32)=1).
//
// Tile shape carried from the production kernel in reinstinct, where a
// sweep (BK in {4,8}, TM/TN in {4,8}, occupancy 1/2) found 4x4 at
// occupancy 2 flat-optimal on gfx906.
#define MMQ_RP_BK 4
#define MMQ_RP_TM 4
#define MMQ_RP_TN 4
#define MMQ_RP_BM (16 * MMQ_RP_TM)
#define MMQ_RP_BN (16 * MMQ_RP_TN)
// MUL_MAT_ID instantiations (TN=1) are light enough for 4 waves/SIMD (64 VGPR cap)
#define MMQ_RP_OCC_ID 4

template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, HAS_IDS ? MMQ_RP_OCC_ID : 2) mmq_gemm_q4k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);

    __shared__ uint4      sW [MMQ_RP_BM][MMQ_RP_BK];     // packed nibbles
    __shared__ float2     sWs[MMQ_RP_BM][MMQ_RP_BK];     // (dsc, deff)
    __shared__ block_q8_1 sX [(16 * TN_)][MMQ_RP_BK + 1]; // int8 activations

    float acc[MMQ_RP_TM][TN_] = {};

    constexpr int LDW = MMQ_RP_BM * MMQ_RP_BK / 256; // tile elems per thread

    // activation row for this thread's sX slot, resolved once
    static_assert((16 * TN_) * MMQ_RP_BK <= 256, "sX staging assumes one slot per thread");
    const int xlr = t / MMQ_RP_BK, xlk = t % MMQ_RP_BK;
    const bool xstage = t < (16 * TN_) * MMQ_RP_BK;
    bool xval;
    uint32_t xoff; // block offset of the row in xq (32-bit: one VGPR)
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + xlr;
        xval = xstage && a < a_end;
        xoff = (xval ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        xval = xstage;
        xoff = 0; // dense: resolved per iteration (hoisting it costs VGPRs)
    }

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
#pragma unroll
        for (int i = 0; i < LDW; i++) {
            const int e  = t + i * 256;
            const int lr = e / MMQ_RP_BK, lk = e % MMQ_RP_BK;
            const uint32_t wrow = row0 + lr;
            const uint32_t sb   = sb0 + lk;
            if (wrow < ne1 && sb < n_sub) {
                sW[lr][lk] = nib[(size_t) wrow * nsp + sb];
                const uint16_t sm = smp[(size_t) wrow * nsp + sb];
                const uint32_t dd = ddp[(size_t) wrow * n_super + (sb >> 3)];
                const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
                const uint16_t dmin_bits = (uint16_t)(dd >> 16);
                sWs[lr][lk] = make_float2(
                    __half2float(*reinterpret_cast<const __half *>(&d_bits))
                        * (float)(sm & 0xFFu),
                    __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                        * (float)(sm >> 8));
            } else {
                sWs[lr][lk] = make_float2(0.0f, 0.0f);
            }
        }
        if (xstage) {
            if constexpr (!HAS_IDS) {
                xval = tok0 + xlr < n_tok;
            }
            if (xval && sb0 + xlk < n_sub) {
                if constexpr (HAS_IDS) {
                    sX[xlr][xlk] = xq[xoff + sb0 + xlk];
                } else {
                    sX[xlr][xlk] = xq[(size_t) (tok0 + xlr) * x_stride + sb0 + xlk];
                }
            } else {
                sX[xlr][xlk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint4 wq[MMQ_RP_TM];
            float dsc[MMQ_RP_TM], deff[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wq[r] = sW[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dsc[r]  = s.x;
                deff[r] = s.y;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                const float sx = __high2float(xb->ds);
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t qa[4] = { wq[r].x, wq[r].y, wq[r].z, wq[r].w };
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                        idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
                    }
                    acc[r][n] += dsc[r] * dx * (float) idot - deff[r] * sx;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// MUL_MAT_ID variant of the Q4_K tile GEMM: one wave per workgroup,
// 64 rows x 16 assignments per tile. Lane l owns rows (l&15)+16i and
// assignments (l>>4)+4j, a 4x4 register tile, so each unpacked weight
// sub-block feeds 4 tokens and each token read feeds 4 rows. The next
// K chunk is prefetched into registers while the current one runs.
// Nibbles are split to int8 once at LDS staging instead of per dp4a.
template <int BK>
static __global__ void __launch_bounds__(64) __attribute__((amdgpu_waves_per_eu(2, 2))) mmq_gemm_q4k_repacked_id_w1(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    static_assert(BK == 1 || BK == 2 || BK == 4, "BK must be a power of two <= 4");
    const int l  = threadIdx.x;
    const int rg = l & 15;
    const int tg = l >> 4;
    const uint32_t row0 = blockIdx.x * 64;
    if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
        return;
    }
    const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
    const uint32_t a_base = (uint32_t) expert_bounds[e] + (blockIdx.y - (uint32_t) tile_off[e]) * 16;
    const uint32_t a_end  = (uint32_t) expert_bounds[e + 1];
    wbase += e * expert_stride;

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 16);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);

    constexpr int NW = BK; // weight (row, kk) items per lane
    __shared__ uint4  sW [64][2 * BK + 1]; // unpacked nibbles (lo, hi) per kk, padded rows
    __shared__ float2 sWs[64][BK];         // (dsc, deff)
    __shared__ uint4  sXq[16][2 * BK + 1]; // int8 activations, padded rows
    __shared__ float2 sXd[16][BK];         // (dx, sx)

    // token staging: lane l < 16*BK owns slot (l/BK, l%BK), row resolved once
    // (upper lanes and out-of-range slots alias a valid row: no extra traffic)
    const int xl = (l / BK) & 15;
    const int xk = l % BK;
    const uint32_t xa = min(a_base + xl, a_end - 1);
    const block_q8_1 * xrow = xq + (size_t) (uint32_t) ids_src1[xa] * x_stride;

    float acc[4][4] = {};

    // Prefetch loads are unconditional (indices clamped) so the compiler
    // keeps them in flight across the compute; out-of-range weight slots
    // get zero scales at staging. Out-of-range assignments stage a valid
    // row's (finite) data and are never written back.
    uint4 pw[NW]; uint16_t psm[NW]; uint32_t pdd[NW]; int px[9];
    auto gload = [&](uint32_t sb0) {
#pragma unroll
        for (int i = 0; i < NW; i++) {
            const int it = l + 64 * i;
            const uint32_t wrow = min(row0 + it / BK, ne1 - 1);
            const uint32_t sb   = min(sb0 + it % BK, n_sub - 1);
            pw[i]  = nib[(size_t) wrow * nsp + sb];
            psm[i] = smp[(size_t) wrow * nsp + sb];
            pdd[i] = ddp[(size_t) wrow * n_super + (sb >> 3)];
        }
        const int * src = reinterpret_cast<const int *>(xrow + min(sb0 + xk, n_sub - 1));
#pragma unroll
        for (int j = 0; j < 9; j++) {
            px[j] = src[j];
        }
    };
    auto lstore = [&](uint32_t sb0) {
#pragma unroll
        for (int i = 0; i < NW; i++) {
            const int it = l + 64 * i;
            const int lr = it / BK, lk = it % BK;
            sW[lr][2 * lk]     = make_uint4( pw[i].x       & 0x0F0F0F0Fu,  pw[i].y       & 0x0F0F0F0Fu,
                                              pw[i].z       & 0x0F0F0F0Fu,  pw[i].w       & 0x0F0F0F0Fu);
            sW[lr][2 * lk + 1] = make_uint4((pw[i].x >> 4) & 0x0F0F0F0Fu, (pw[i].y >> 4) & 0x0F0F0F0Fu,
                                             (pw[i].z >> 4) & 0x0F0F0F0Fu, (pw[i].w >> 4) & 0x0F0F0F0Fu);
            const uint16_t d_bits = (uint16_t)(pdd[i] & 0xFFFF), m_bits = (uint16_t)(pdd[i] >> 16);
            const bool ok = row0 + lr < ne1 && sb0 + lk < n_sub;
            sWs[lr][lk] = ok ? make_float2(
                __half2float(*reinterpret_cast<const __half *>(&d_bits)) * (float)(psm[i] & 0xFFu),
                __half2float(*reinterpret_cast<const __half *>(&m_bits)) * (float)(psm[i] >> 8))
                : make_float2(0.0f, 0.0f);
        }
        if (l < 16 * BK) {
            sXq[xl][2 * xk]     = make_uint4(px[1], px[2], px[3], px[4]);
            sXq[xl][2 * xk + 1] = make_uint4(px[5], px[6], px[7], px[8]);
            sXd[xl][xk] = __half22float2(*reinterpret_cast<const half2 *>(&px[0]));
        }
    };

    gload(0);
    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += BK) {
        __syncthreads();
        lstore(sb0);
        __syncthreads();
        gload(sb0 + BK); // clamped on the last step
#pragma unroll 1
        for (int kk = 0; kk < BK; kk++) {
            int   xv[4][8];
            float dx[4], sx[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint4 a = sXq[tg + 4 * j][2 * kk], b = sXq[tg + 4 * j][2 * kk + 1];
                xv[j][0] = a.x; xv[j][1] = a.y; xv[j][2] = a.z; xv[j][3] = a.w;
                xv[j][4] = b.x; xv[j][5] = b.y; xv[j][6] = b.z; xv[j][7] = b.w;
                const float2 d = sXd[tg + 4 * j][kk];
                dx[j] = d.x; sx[j] = d.y;
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int lr = rg + 16 * i;
                const uint4  qa = sW[lr][2 * kk], qb = sW[lr][2 * kk + 1];
                const float2 s  = sWs[lr][kk];
                const int w8[8] = { (int) qa.x, (int) qa.y, (int) qa.z, (int) qa.w,
                                    (int) qb.x, (int) qb.y, (int) qb.z, (int) qb.w };
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    int idot = 0;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        idot = ggml_cuda_dp4a(w8[k],     xv[j][k],     idot);
                        idot = ggml_cuda_dp4a(w8[k + 4], xv[j][k + 4], idot);
                    }
                    acc[i][j] += s.x * dx[j] * (float) idot - s.y * sx[j];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t row = row0 + rg + 16 * i;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t a = a_base + tg + 4 * j;
            if (a < a_end) {
                y[(size_t) ids_dst[a] * dst_s1 + row] = acc[i][j];
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q5_K MMQ - Q4_K's tiles; the qh plane is folded into int8 at LDS staging, as in the Q5_1 kernel.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, HAS_IDS ? MMQ_RP_OCC_ID : 2) mmq_gemm_q5k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    // int8 weights per (row, sub-block), 5th bit folded in at staging (see the Q5_1 kernel)
    __shared__ int        sW8 [MMQ_RP_BM][MMQ_RP_BK][8];
    __shared__ float2     sWs [MMQ_RP_BM][MMQ_RP_BK];
    __shared__ block_q8_1 sX  [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    // activation row for this thread's sX slot, resolved once
    const bool xstage = lr < (16 * TN_);
    bool xval;
    uint32_t xoff; // block offset of the row in xq (32-bit: one VGPR)
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + lr;
        xval = xstage && a < a_end;
        xoff = (xval ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        xval = xstage;
        xoff = 0; // dense: resolved per iteration (hoisting it costs VGPRs)
    }

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            const uint4    q  = nib[(size_t) wrow * nsp + sb];
            const uint32_t qh = qhp[(size_t) wrow * nsp + sb];
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int j = 0; j < 4; j++) {
                sW8[lr][lk][j]     = (int) (( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j))     & 0xFu));
                sW8[lr][lk][4 + j] = (int) (((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j + 4)) & 0xFu));
            }
            const uint16_t sm = smp[(size_t) wrow * nsp + sb];
            const uint32_t dd = ddp[(size_t) wrow * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            sWs[lr][lk] = make_float2(
                __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm & 0xFFu),
                __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm >> 8));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                sW8[lr][lk][j] = 0;
            }
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (xstage) {
            if constexpr (!HAS_IDS) {
                xval = tok0 + lr < n_tok;
            }
            if (xval && sb < n_sub) {
                if constexpr (HAS_IDS) {
                    sX[lr][lk] = xq[xoff + sb];
                } else {
                    sX[lr][lk] = xq[(size_t) (tok0 + lr) * x_stride + sb];
                }
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                const float sx = __high2float(xb->ds);
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const int * w8 = sW8[ty + r * 16][kk];
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        idot = ggml_cuda_dp4a(w8[j], xq32[j], idot);
                    }
                    const float2 s = sWs[ty + r * 16][kk];
                    acc[r][n] += s.x * dx * (float) idot - s.y * sx;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q5_1 MMQ. The 5th bit is folded into int8 once while staging the weight tile in LDS: doing it
// per dp4a in the inner loop repeats the unpack for every token column and made the kernel
// ALU-bound, 1.5x slower than the canonical MMQ. x = d*q + m, so the min adds.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, HAS_IDS ? MMQ_RP_OCC_ID : 2) mmq_gemm_q5_1_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(wbase + (size_t) ne1 * nsp * 16);
    const half2    * dmp = reinterpret_cast<const half2 *>(wbase + (size_t) ne1 * nsp * 20);

    // int8 weights per (row, sub-block): [j] = weights 4j..4j+3, [4+j] = weights 16+4j..16+4j+3
    __shared__ int        sW8[MMQ_RP_BM][MMQ_RP_BK][8];
    __shared__ float2     sWs[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ block_q8_1 sX [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    // activation row for this thread's sX slot, resolved once
    const bool xstage = lr < (16 * TN_);
    bool xval;
    uint32_t xoff; // block offset of the row in xq (32-bit: one VGPR)
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + lr;
        xval = xstage && a < a_end;
        xoff = (xval ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        xval = xstage;
        xoff = 0; // dense: resolved per iteration (hoisting it costs VGPRs)
    }

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            const size_t   idx = (size_t) wrow * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qh  = qhp[idx];
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int j = 0; j < 4; j++) {
                sW8[lr][lk][j]     = (int) (( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j))     & 0xFu));
                sW8[lr][lk][4 + j] = (int) (((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j + 4)) & 0xFu));
            }
            sWs[lr][lk] = __half22float2(dmp[idx]);
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                sW8[lr][lk][j] = 0;
            }
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (xstage) {
            if constexpr (!HAS_IDS) {
                xval = tok0 + lr < n_tok;
            }
            if (xval && sb < n_sub) {
                if constexpr (HAS_IDS) {
                    sX[lr][lk] = xq[xoff + sb];
                } else {
                    sX[lr][lk] = xq[(size_t) (tok0 + lr) * x_stride + sb];
                }
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                const float sx = __high2float(xb->ds);
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const int * w8 = sW8[ty + r * 16][kk];
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        idot = ggml_cuda_dp4a(w8[j], xq32[j], idot);
                    }
                    const float2 dm = sWs[ty + r * 16][kk];
                    acc[r][n] += dm.x * dx * (float) idot + dm.y * sx;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q6_K MMQ — h2 plane staged as uint2, signed scale pairs, per-token
// activation half-sums for the symmetric -32 fold.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_q6k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint2    * h2p = reinterpret_cast<const uint2 *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 2);

    __shared__ uint4      sW  [MMQ_RP_BM][MMQ_RP_BK];
    __shared__ uint2      sWh2[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ float2     sWs [MMQ_RP_BM][MMQ_RP_BK];
    __shared__ block_q8_1 sX  [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    // activation row for this thread's sX slot, resolved once
    const bool xstage = lr < (16 * TN_);
    bool xval;
    uint32_t xoff; // block offset of the row in xq (32-bit: one VGPR)
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + lr;
        xval = xstage && a < a_end;
        xoff = (xval ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        xval = xstage;
        xoff = 0; // dense: resolved per iteration (hoisting it costs VGPRs)
    }

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            sW  [lr][lk] = nib[(size_t) wrow * nsp + sb];
            sWh2[lr][lk] = h2p[(size_t) wrow * nsp + sb];
            const uint16_t sm     = smp[(size_t) wrow * nsp + sb];
            const uint16_t d_bits = ddp[(size_t) wrow * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            sWs[lr][lk] = make_float2(d * (float)(int)(int8_t)(sm & 0xFFu),
                                      d * (float)(int)(int8_t)(sm >> 8));
        } else {
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (xstage) {
            if constexpr (!HAS_IDS) {
                xval = tok0 + lr < n_tok;
            }
            if (xval && sb < n_sub) {
                if constexpr (HAS_IDS) {
                    sX[lr][lk] = xq[xoff + sb];
                } else {
                    sX[lr][lk] = xq[(size_t) (tok0 + lr) * x_stride + sb];
                }
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint4 wq[MMQ_RP_TM]; uint2 wh2[MMQ_RP_TM];
            float dlo[MMQ_RP_TM], dhi[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wq[r]  = sW  [ty + r * 16][kk];
                wh2[r] = sWh2[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dlo[r] = s.x;
                dhi[r] = s.y;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                int xis0 = 0, xis1 = 0;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
                    xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
                }
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t qa[4] = { wq[r].x, wq[r].y, wq[r].z, wq[r].w };
                    const uint32_t h2lo = wh2[r].x, h2hi = wh2[r].y;
                    int idot0 = 0, idot1 = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint32_t ge = 2 * j;
                        const uint32_t go = 2 * j + 1;
                        const uint32_t he = ((ge < 4 ? h2lo : h2hi) >> (8 * (ge & 3))) & 0xFFu;
                        const uint32_t ho = ((go < 4 ? h2lo : h2hi) >> (8 * (go & 3))) & 0xFFu;
                        const uint32_t q6lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread2(he);
                        const uint32_t q6hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread2(ho);
                        idot0 = ggml_cuda_dp4a((int) q6lo, xq32[j],     idot0);
                        idot1 = ggml_cuda_dp4a((int) q6hi, xq32[j + 4], idot1);
                    }
                    acc[r][n] += dlo[r] * dx * (float)(idot0 - 32 * xis0)
                               + dhi[r] * dx * (float)(idot1 - 32 * xis1);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q3_K MMQ — Q6_K's tiles but the quant is 2-bit lo + 1-bit hi (no 4-bit
// nibble plane); reconstruct q3 = lo2 | (hbit << 2). Symmetric, bias 4.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_q3k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint2    * lo2p = reinterpret_cast<const uint2 *>(wbase);
    const uint32_t * hi1p = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 8);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    __shared__ uint2      sWl[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ uint32_t   sWh[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ float2     sWs[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ block_q8_1 sX [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    // activation row for this thread's sX slot, resolved once
    const bool xstage = lr < (16 * TN_);
    bool xval;
    uint32_t xoff; // block offset of the row in xq (32-bit: one VGPR)
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + lr;
        xval = xstage && a < a_end;
        xoff = (xval ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        xval = xstage;
        xoff = 0; // dense: resolved per iteration (hoisting it costs VGPRs)
    }

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            sWl[lr][lk] = lo2p[(size_t) wrow * nsp + sb];
            sWh[lr][lk] = hi1p[(size_t) wrow * nsp + sb];
            const uint16_t sm     = smp[(size_t) wrow * nsp + sb];
            const uint16_t d_bits = ddp[(size_t) wrow * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            sWs[lr][lk] = make_float2(d * (float)(int)(int8_t)(sm & 0xFFu),
                                      d * (float)(int)(int8_t)(sm >> 8));
        } else {
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (xstage) {
            if constexpr (!HAS_IDS) {
                xval = tok0 + lr < n_tok;
            }
            if (xval && sb < n_sub) {
                if constexpr (HAS_IDS) {
                    sX[lr][lk] = xq[xoff + sb];
                } else {
                    sX[lr][lk] = xq[(size_t) (tok0 + lr) * x_stride + sb];
                }
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint2 wl[MMQ_RP_TM]; uint32_t wh[MMQ_RP_TM];
            float dlo[MMQ_RP_TM], dhi[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wl[r] = sWl[ty + r * 16][kk];
                wh[r] = sWh[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dlo[r] = s.x;
                dhi[r] = s.y;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                int xis0 = 0, xis1 = 0;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
                    xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
                }
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t lo2lo = wl[r].x, lo2hi = wl[r].y, qh = wh[r];
                    int idot0 = 0, idot1 = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint32_t ge = 2 * j;
                        const uint32_t go = 2 * j + 1;
                        const uint32_t lb = ((ge < 4 ? lo2lo : lo2hi) >> (8 * (ge & 3))) & 0xFFu;
                        const uint32_t hb = ((go < 4 ? lo2lo : lo2hi) >> (8 * (go & 3))) & 0xFFu;
                        const uint32_t q3lo = repack_spread2_lo(lb) | repack_spread1_hi((qh >> (8 * j))     & 0xFu);
                        const uint32_t q3hi = repack_spread2_lo(hb) | repack_spread1_hi((qh >> (8 * j + 4)) & 0xFu);
                        idot0 = ggml_cuda_dp4a((int) q3lo, xq32[j],     idot0);
                        idot1 = ggml_cuda_dp4a((int) q3hi, xq32[j + 4], idot1);
                    }
                    acc[r][n] += dlo[r] * dx * (float)(idot0 - 4 * xis0)
                               + dhi[r] * dx * (float)(idot1 - 4 * xis1);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q8_0 MMQ — 32 qs bytes per sub-block staged as two uint4s; no offset
// term, so the accumulate is just dsc * dx * idot.
// The dense instantiation runs 3 blocks/CU (84 VGPR cap, 3 x 20 KB LDS):
// the next W step is prefetched into registers during the compute, and
// the kk step is split into 16-byte halves to cut live operands.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, HAS_IDS ? 2 : 3) mmq_gemm_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4    * qsp = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * dp  = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 32);

    // One raw LDS buffer with typed views. Rows of the qs planes are
    // 2*BK+1 uint4 (144 B) so b128 reads with tx-distinct rows are
    // conflict-free; slot 2*kk is qs[0..15], 2*kk+1 is qs[16..31].
    // After the K loop the buffer is reused as the output tile.
    constexpr int QS_LD  = 2 * MMQ_RP_BK + 1;
    constexpr int XR     = 16 * TN_;
    constexpr int Y_LD   = MMQ_RP_BM + 2; // 2tx+ty banks: conflict-free per half-wave
    constexpr int OFF_XQ = MMQ_RP_BM * QS_LD * 16;
    constexpr int OFF_WD = OFF_XQ + XR * QS_LD * 16;
    constexpr int OFF_XD = OFF_WD + MMQ_RP_BM * MMQ_RP_BK * 4;
    constexpr int SZ_K   = OFF_XD + XR * MMQ_RP_BK * 4;
    constexpr int SZ_Y   = HAS_IDS ? 0 : XR * Y_LD * 4;
    constexpr int SZ     = SZ_K > SZ_Y ? SZ_K : SZ_Y;
    __shared__ uint4 smem[SZ / 16];
    uint4 (*sW )[QS_LD]        = reinterpret_cast<uint4 (*)[QS_LD]>(smem);
    uint4 (*sXq)[QS_LD]        = reinterpret_cast<uint4 (*)[QS_LD]>((char *) smem + OFF_XQ);
    float (*sWd)[MMQ_RP_BK]    = reinterpret_cast<float (*)[MMQ_RP_BK]>((char *) smem + OFF_WD);
    float (*sXd)[MMQ_RP_BK]    = reinterpret_cast<float (*)[MMQ_RP_BK]>((char *) smem + OFF_XD);

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    // Staging slot of this thread: weight row lr and activation row lr,
    // sub-block lk of each BK step. Out-of-range rows and sub-blocks are
    // clamped to valid addresses so the loads issue unmasked; no zeroing
    // is needed: a clamped row only feeds outputs that are never stored,
    // and a clamped sub-block is skipped by the K-tail check. Selecting
    // on the loaded values would force an early vmcnt wait.
    const bool     xstage = XR >= 64 || lr < XR; // lr < 64 always
    // 32-bit element offsets against uniform bases keep one VGPR each
    const uint32_t w_off = min(row0 + lr, ne1 - 1) * nsp;
    uint32_t x_off;
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + lr;
        x_off = (xstage && a < a_end ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        x_off = min(tok0 + lr, n_tok - 1) * x_stride;
    }

    // W of the next BK step is prefetched into registers; X is loaded at
    // the top of its own step (prefetching it too spills at 84 VGPRs).
    // Native vectors: HIP uint4 locals here would stay in scratch.
    typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
    u32x4    pw_lo, pw_hi, px_a, px_b;
    uint32_t pd, px_c;
    auto gload_w = [&](const uint32_t sb0) {
        const uint32_t wi = w_off + min(sb0 + lk, n_sub - 1);
        pw_lo = reinterpret_cast<const u32x4 *>(qsp)[wi * 2];
        pw_hi = reinterpret_cast<const u32x4 *>(qsp)[wi * 2 + 1];
        pd    = dp[wi];
    };
    // X kept as loaded (d, qs[0..31] = 9 dwords), shuffled at the store
    auto gload_x = [&](const uint32_t sb0) {
        if (xstage) {
            const uint32_t * xi = reinterpret_cast<const uint32_t *>(xq + (x_off + min(sb0 + lk, n_sub - 1)));
            px_a = u32x4{xi[0], xi[1], xi[2], xi[3]};
            px_b = u32x4{xi[4], xi[5], xi[6], xi[7]};
            px_c = xi[8];
        }
    };
    auto lstore = [&]() {
        reinterpret_cast<u32x4 *>(sW[lr])[2 * lk]     = pw_lo;
        reinterpret_cast<u32x4 *>(sW[lr])[2 * lk + 1] = pw_hi;
        const uint16_t d_bits = (uint16_t) pd;
        sWd[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&d_bits));
        if (xstage) {
            reinterpret_cast<u32x4 *>(sXq[lr])[2 * lk]     = u32x4{px_a.y, px_a.z, px_a.w, px_b.x};
            reinterpret_cast<u32x4 *>(sXq[lr])[2 * lk + 1] = u32x4{px_b.y, px_b.z, px_b.w, px_c};
            const uint16_t xd_bits = (uint16_t) px_a.x;
            sXd[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&xd_bits));
        }
    };

    gload_w(0);
    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        gload_x(sb0);
        __syncthreads();
        lstore();
        __syncthreads();
        // unconditional: the last step reloads a clamped block, but a
        // branch here makes the compiler wait on the loads at the join
        gload_w(sb0 + MMQ_RP_BK);

        // K tail: skipped terms would add exactly +0.0f (scale 0)
        const int kk_end = min((int) MMQ_RP_BK, (int) (n_sub - sb0));
        if constexpr (!HAS_IDS) {
            // lo then hi 16-byte half; integer sums are order-free. The
            // sched barriers stop the halves' LDS reads from being hoisted
            // together, which spills at the 84 VGPR cap.
#pragma unroll 1
            for (int kk = 0; kk < kk_end; kk++) {
                int idot[MMQ_RP_TM][TN_];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    u32x4 wq[MMQ_RP_TM];
#pragma unroll
                    for (int r = 0; r < MMQ_RP_TM; r++) {
                        wq[r] = reinterpret_cast<const u32x4 *>(sW[ty + r * 16])[2 * kk + h];
                    }
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for (int n = 0; n < TN_; n++) {
                        const u32x4 xv = reinterpret_cast<const u32x4 *>(sXq[tx + n * 16])[2 * kk + h];
#pragma unroll
                        for (int r = 0; r < MMQ_RP_TM; r++) {
                            int d = h ? idot[r][n] : 0;
                            d = ggml_cuda_dp4a((int) wq[r].x, (int) xv.x, d);
                            d = ggml_cuda_dp4a((int) wq[r].y, (int) xv.y, d);
                            d = ggml_cuda_dp4a((int) wq[r].z, (int) xv.z, d);
                            d = ggml_cuda_dp4a((int) wq[r].w, (int) xv.w, d);
                            idot[r][n] = d;
                        }
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
#pragma unroll
                for (int n = 0; n < TN_; n++) {
                    const float dx = sXd[tx + n * 16][kk];
#pragma unroll
                    for (int r = 0; r < MMQ_RP_TM; r++) {
                        acc[r][n] += sWd[ty + r * 16][kk] * dx * (float) idot[r][n];
                    }
                }
            }
            continue;
        }
#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            if (kk >= kk_end) {
                break;
            }
            uint4 wq_lo[MMQ_RP_TM], wq_hi[MMQ_RP_TM];
            float dsc[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wq_lo[r] = sW [ty + r * 16][2 * kk];
                wq_hi[r] = sW [ty + r * 16][2 * kk + 1];
                dsc[r]   = sWd[ty + r * 16][kk];
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const uint4 xa = sXq[tx + n * 16][2 * kk];
                const uint4 xb = sXq[tx + n * 16][2 * kk + 1];
                const float dx = sXd[tx + n * 16][kk];
                const int xq32[8] = { (int) xa.x, (int) xa.y, (int) xa.z, (int) xa.w,
                                      (int) xb.x, (int) xb.y, (int) xb.z, (int) xb.w };
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t lo[4] = { wq_lo[r].x, wq_lo[r].y, wq_lo[r].z, wq_lo[r].w };
                    const uint32_t hi[4] = { wq_hi[r].x, wq_hi[r].y, wq_hi[r].z, wq_hi[r].w };
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = ggml_cuda_dp4a((int) lo[j], xq32[j],     idot);
                        idot = ggml_cuda_dp4a((int) hi[j], xq32[j + 4], idot);
                    }
                    acc[r][n] += dsc[r] * dx * (float) idot;
                }
            }
        }
    }
    __syncthreads();

    if constexpr (HAS_IDS) {
#pragma unroll
        for (int r = 0; r < MMQ_RP_TM; r++) {
            const uint32_t row = row0 + ty + r * 16;
            if (row >= ne1) {
                continue;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            }
        }
    } else {
        // transpose through LDS so each token row is stored contiguously
        float (*tileY)[Y_LD] = reinterpret_cast<float (*)[Y_LD]>(smem);
#pragma unroll
        for (int r = 0; r < MMQ_RP_TM; r++) {
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                tileY[tx + n * 16][ty + r * 16] = acc[r][n];
            }
        }
        __syncthreads();
#pragma unroll
        for (int idx = t; idx < MMQ_RP_BM * XR; idx += 256) {
            const int tok = idx / MMQ_RP_BM;
            const int row = idx % MMQ_RP_BM;
            if (tok0 + tok < n_tok && row0 + row < ne1) {
                y[(size_t) (tok0 + tok) * dst_s1 + row0 + row] = tileY[tok][row];
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// ---------------------------------------------------------------------
// MUL_MAT dispatch
// ---------------------------------------------------------------------

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, int64_t ne00, int64_t ne01, int64_t ne11,
        int64_t x_stride, cudaStream_t stream);

static const block_q8_1 * repack_quantize_x(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        int64_t ne10_padded, ggml_cuda_pool_alloc<char> & fallback, cudaStream_t stream);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float)); // rows may be strided; dim0 must be dense
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // M
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1]; // N
    const int64_t ne12 = src1->ne[2]; // broadcast slices (2D weight repeats)
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();
    const uint8_t * w = (const uint8_t *) src0->data;

    // Quantize the whole (possibly 3D/4D) activation once; blocks land
    // contiguously as [ne13][ne12][ne11][ne10_padded/QK8_1].
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1; // q8_1 blocks per column
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    const block_q8_1 * xq_all = repack_quantize_x(ctx, src1, ne10_padded, src1_q8_1, stream);

    for (int64_t i3 = 0; i3 < ne13; i3++) {
    for (int64_t i2 = 0; i2 < ne12; i2++) {
        const block_q8_1 * xq = xq_all + (i3 * ne12 + i2) * ne11 * x_stride;
        float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
        ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
            ne00, ne01, ne11, x_stride, stream);
    }
    }
}

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const int64_t x_stride, cudaStream_t stream) {
    if (ne11 == 1) {
        // decode: dp4a matvec straight from the planes
        switch (src0->type) {
            case GGML_TYPE_Q3_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q3k_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q4_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q4k_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q5_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q5k_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q6_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q6k_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q5_1: {
                if (ne00 <= 1024) {
                    launch_mul_mat_vec_q5_1_repacked_seg<false>(w, xq, dst_d, ne00, ne01, 1, nullptr, 0, 0, 0, stream);
                    break;
                }
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q5_1_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q8_0: {
                // short rows: several rows per wave (see mul_mat_vec_q8_0_repacked_seg)
                if (ne00 <= 1024) {
                    launch_mul_mat_vec_q8_0_repacked_seg<false>(w, xq, dst_d, ne00, ne01, 1, nullptr, 0, 0, 0, stream);
                    break;
                }
                // few long rows: split K across a whole workgroup per row
                if (ne01 <= 1024 && ne00 >= 4096) {
                    mul_mat_vec_q8_0_repacked_splitk<<<ne01, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01);
                    break;
                }
                // ne01 >= 512: single-wave ROWS=1 blocks maximize the
                // wavefront count (measured on gfx906: ROWS=2 at
                // ne01=4096 stalls ~184 GB/s, ROWS=1 ~2x it; at 512-2560
                // rows ROWS=1 is also faster in qwen4exp decode, tg +1.5%).
                // Small ne01: 4-wave ROWS=2 blocks (the K-quant matvec shape).
                if (ne01 >= 512) {
                    const dim3 grid(ne01, 1, 1);
                    mul_mat_vec_q8_0_repacked<1, 1, false><<<grid, 64, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, nullptr, nullptr, 0, 0, 0, 0);
                } else {
                    // 4-wave ROWS=2 with half-sub-block work units is the
                    // best of the swept variants (gfx906, 0.8B-Q8_0 tg128:
                    // 231.0 vs 222.7 single-wave, 222.2 full-block units,
                    // 219.9 ROWS=4, 214.2 quarter units; canonical mmvq
                    // is 238.0 — the residual ~3% is why Q8_0 stays
                    // behind its own env gate)
                    const dim3 grid((ne01 + 7) / 8, 1, 1);
                    mul_mat_vec_q8_0_repacked<2, 4, false><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, nullptr, nullptr, 0, 0, 0, 0);
                }
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    // prefill: int8 MMQ tile GEMM straight from the repacked planes
    const dim3 grid((ne01 + MMQ_RP_BM - 1) / MMQ_RP_BM,
                    (ne11 + MMQ_RP_BN - 1) / MMQ_RP_BN, 1);
    switch (src0->type) {
        case GGML_TYPE_Q3_K:
            mmq_gemm_q3k_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        case GGML_TYPE_Q4_K:
            mmq_gemm_q4k_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        case GGML_TYPE_Q5_K:
            mmq_gemm_q5k_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        case GGML_TYPE_Q6_K:
            mmq_gemm_q6k_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        case GGML_TYPE_Q5_1:
            mmq_gemm_q5_1_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        case GGML_TYPE_Q8_0:
            mmq_gemm_q8_0_repacked<false, MMQ_RP_TN><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        default: GGML_ABORT("unsupported repack type");
    }
    GGML_UNUSED(ctx);
}

// Routing cache use: bypassed on side streams and with
// GGML_CUDA_REPACK_NO_ROUTE_CACHE set. Under graph capture the hit/miss
// sequence is baked into the graph, which is fine: each replay reruns the
// misses in the same stream order, and cache buffers are never freed while
// the context lives. Growing (cudaMalloc) is not done while capturing;
// *capturing tells the caller to fall back to pool buffers instead.
static bool repack_route_cache_usable(ggml_backend_cuda_context & ctx, cudaStream_t stream, bool * capturing) {
    static const bool disabled = getenv("GGML_CUDA_REPACK_NO_ROUTE_CACHE") != nullptr;
    if (disabled || ctx.curr_stream_no != 0) {
        return false;
    }
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus st;
    CUDA_CHECK(hipStreamIsCapturing(stream, &st));
    *capturing = st != hipStreamCaptureStatusNone;
#else
    cudaStreamCaptureStatus st;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &st));
    *capturing = st != cudaStreamCaptureStatusNone;
#endif
    return true;
}

// grow-only cache buffer; false if it would have to grow during capture
static bool repack_rc_reserve(ggml_cuda_repack_route_cache & rc, void ** buf, size_t * cap, size_t need, bool capturing) {
    if (need <= *cap) {
        return true;
    }
    if (capturing) {
        return false;
    }
    if (*buf != nullptr) {
        rc.retired.push_back(*buf);
    }
    CUDA_CHECK(cudaMalloc(buf, need));
    *cap = need;
    return true;
}

void ggml_cuda_repack_xq_invalidate(ggml_backend_cuda_context & ctx, const ggml_tensor * node, bool force) {
    if (node->data == nullptr) {
        return;
    }
    const char * lo = (const char *) node->data;
    const char * hi = lo + ggml_nbytes(node);
    for (auto & e : ctx.repack_rc.xqc) {
        if (e.gen == ctx.graph_gen && (force || e.producer != node) && lo < e.hi && e.lo < hi) {
            e.gen = 0;
        }
    }
}

static const ggml_tensor * repack_view_root(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

void * ggml_cuda_repack_xq_emit_target(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const ggml_tensor * t) {
    static const bool disabled = getenv("GGML_CUDA_NO_Q8_EMIT") != nullptr;
    if (disabled || cgraph == nullptr || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
            ggml_nelements(t) % QK8_1 != 0) {
        return nullptr;
    }
    bool consumer = false;
    for (int i = 0; i < cgraph->n_nodes && !consumer; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_MUL_MAT || !n->src[0]->buffer || !ggml_backend_buft_is_cuda_repack(n->src[0]->buffer->buft)) {
            continue;
        }
        const ggml_tensor * x = n->src[1];
        consumer = repack_view_root(x) == repack_view_root(t) && x->data == t->data && ggml_is_contiguous(x) &&
            x->type == GGML_TYPE_F32 && x->ne[1] == 1 && x->ne[2] == 1 && x->ne[3] == 1 && x->ne[0] <= ggml_nelements(t);
    }
    bool capturing = false;
    if (!consumer || !repack_route_cache_usable(ctx, ctx.stream(), &capturing)) {
        return nullptr;
    }
    ggml_cuda_repack_route_cache & rc = ctx.repack_rc;
    auto & e = rc.xqc[rc.xqc_next];
    rc.xqc_next = (rc.xqc_next + 1) % ggml_cuda_repack_route_cache::N_XQ;
    e.gen = 0;
    const size_t bytes = ggml_nelements(t) / QK8_1 * sizeof(block_q8_1);
    if (!repack_rc_reserve(rc, (void **) &e.buf, &e.cap, bytes, capturing)) {
        return nullptr;
    }
    e.gen      = ctx.graph_gen;
    e.data     = t->data;
    memset(e.ne, 0, sizeof(e.ne));
    memset(e.nb, 0, sizeof(e.nb));
    e.lo       = (const char *) t->data;
    e.hi       = e.lo + ggml_nbytes(t);
    e.producer = t;
    e.flat_n   = ggml_nelements(t);
    return e.buf;
}

// Quantize src1 (F32, dim0 dense) to q8_1 as [ne3][ne2][ne1][ne10_padded/QK8_1]
// blocks. On the main stream the result is cached per graph: gate/up and the
// shared expert (or q/k/v) quantize the same activation. Entries die when any
// node writes their range (the graph loop calls ggml_cuda_repack_xq_invalidate),
// so reused allocator memory never serves stale data.
static const block_q8_1 * repack_quantize_x(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        const int64_t ne10_padded, ggml_cuda_pool_alloc<char> & fallback, cudaStream_t stream) {
    const size_t bytes = src1->ne[3] * src1->ne[2] * src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1;
    auto quantize = [&](char * out) {
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, out, GGML_TYPE_Q8_0, src1->ne[0],
            src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
            ne10_padded, src1->ne[1], src1->ne[2], src1->ne[3], stream);
    };
    bool capturing = false;
    if (!repack_route_cache_usable(ctx, stream, &capturing)) {
        char * p = fallback.alloc(bytes);
        quantize(p);
        return (const block_q8_1 *) p;
    }
    ggml_cuda_repack_route_cache & rc = ctx.repack_rc;
    const bool single_col = src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_is_contiguous(src1);
    for (auto & e : rc.xqc) {
        if (e.gen != ctx.graph_gen || e.data != src1->data) {
            continue;
        }
        if (e.flat_n > 0 ? (single_col && src1->ne[0] <= e.flat_n) :
                (memcmp(e.ne, src1->ne, sizeof(e.ne)) == 0 && memcmp(e.nb, src1->nb, sizeof(e.nb)) == 0)) {
            return (const block_q8_1 *) e.buf;
        }
    }
    auto & e = rc.xqc[rc.xqc_next];
    rc.xqc_next = (rc.xqc_next + 1) % ggml_cuda_repack_route_cache::N_XQ;
    e.gen = 0;
    if (!repack_rc_reserve(rc, (void **) &e.buf, &e.cap, bytes, capturing)) {
        char * p = fallback.alloc(bytes);
        quantize(p);
        return (const block_q8_1 *) p;
    }
    quantize(e.buf);
    e.gen  = ctx.graph_gen;
    e.data = src1->data;
    memcpy(e.ne, src1->ne, sizeof(e.ne));
    memcpy(e.nb, src1->nb, sizeof(e.nb));
    e.lo = (const char *) src1->data;
    e.hi = e.lo + ggml_nbytes(src1);
    e.producer = nullptr;
    e.flat_n   = 0;
    return (const block_q8_1 *) e.buf;
}

// MUL_MAT_ID with src0 in the repack buffer type. The mm_ids_helper
// compacts routing into expert-sorted assignment order; activations are
// quantized once in natural column order and gathered per assignment
// via ids_src1 inside the kernels; outputs scatter via ids_dst.
void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[0]  == sizeof(int32_t));
    GGML_ASSERT(src1->ne[3] == 1 && dst->ne[3] == 1);
    // column-contiguity: ids_src1/ids_dst are flat column indices
    GGML_ASSERT(src1->nb[2] == src1->nb[1] * src1->ne[1]);
    GGML_ASSERT(dst->nb[2]  == dst->nb[1]  * dst->ne[1]);
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // rows per expert
    const int64_t ne02 = src0->ne[2]; // experts
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 == ne00);
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_assign      = n_expert_used * n_tokens;

    cudaStream_t stream = ctx.stream();
    const uint8_t * w = (const uint8_t *) src0->data;
    float * dst_d = (float *) dst->data;
    const size_t expert_stride = repack_gcn_nbytes(src0->type, ne00, ne01);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    const size_t  xq_bytes    = src1->ne[2] * src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1;
    const int     si1         = ids->nb[1] / sizeof(int32_t);
    const int     sis1        = src1->nb[2] / src1->nb[1];

    // batch: grouped tile GEMM, thin 16-token tiles (MoE routing spreads
    // tokens across experts; a 64-wide tile would be mostly empty)
    constexpr int TN_ID = 1;
    // over-launch upper bound: every expert can add one partial tile
    const int64_t max_tiles = n_assign / (16 * TN_ID) + ne02;
    GGML_ASSERT(ne02 <= 4096);

    int32_t * p_ids_src1    = nullptr;
    int32_t * p_ids_dst     = nullptr;
    int32_t * p_bounds      = nullptr;
    int32_t * p_tile_off    = nullptr;
    int32_t * p_tile_expert = nullptr;
    char    * p_xq          = nullptr;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool());
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> tile_off(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> tile_expert(ctx.pool());
    ggml_cuda_pool_alloc<char>    src1_q8_1(ctx.pool());

    auto quantize_x = [&](char * xq_out) {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, xq_out,
            src0->type, ne10, s11, s12, s12 * src1->ne[2], ne10_padded,
            src1->ne[1], src1->ne[2], 1, stream);
    };
    auto route = [&](int32_t * s1, int32_t * d, int32_t * b, int32_t * to, int32_t * te) {
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, s1, d, b,
            ne02, n_tokens, n_expert_used, src1->ne[1], si1, sis1, /*write_inverse =*/ false, stream);
        CUDA_CHECK(cudaGetLastError());
        repack_tile_map<16 * TN_ID><<<1, 1024, 0, stream>>>(b, to, te, ne02);
    };

    bool capturing = false;
    const bool use_rc = n_tokens > 1 && repack_route_cache_usable(ctx, stream, &capturing);
    ggml_cuda_repack_route_cache & rc = ctx.repack_rc;

    if (n_tokens > 1 && use_rc) {
        // gate/up/down of one layer share ids: reuse the routing
        const bool route_hit = rc.gen == ctx.graph_gen && rc.ids == ids && rc.ids_data == ids->data &&
            rc.n_tokens == n_tokens && rc.n_used == n_expert_used && rc.ne02 == ne02 && rc.si1 == si1;
        // ids_src1[a] = it*sis1 + slot % ne11 (contiguous src1: sis1 == ne11),
        // ids_dst[a] = it*n_used + slot, so ne11 == n_used reuses ids_dst
        const bool src1_is_dst = src1->ne[1] == n_expert_used && sis1 == n_expert_used;
        if (route_hit && (rc.ne11 == src1->ne[1] || src1_is_dst)) {
            p_ids_src1    = rc.ne11 == src1->ne[1] ? rc.ids_src1 : rc.ids_dst;
            p_ids_dst     = rc.ids_dst;
            p_bounds      = rc.bounds;
            p_tile_off    = rc.tile_off;
            p_tile_expert = rc.tile_expert;
        } else {
            rc.gen = 0; // invalid until rewritten below
            const size_t need = (2 * n_assign + 2 * (ne02 + 1) + max_tiles) * sizeof(int32_t);
            if (repack_rc_reserve(rc, &rc.route_buf, &rc.cap, need, capturing)) {
                rc.ids_src1    = (int32_t *) rc.route_buf;
                rc.ids_dst     = rc.ids_src1 + n_assign;
                rc.bounds      = rc.ids_dst + n_assign;
                rc.tile_off    = rc.bounds + ne02 + 1;
                rc.tile_expert = rc.tile_off + ne02 + 1;
                rc.gen      = ctx.graph_gen;
                rc.ids      = ids;
                rc.ids_data = ids->data;
                rc.n_tokens = n_tokens;
                rc.n_used   = n_expert_used;
                rc.ne02     = ne02;
                rc.si1      = si1;
                rc.ne11     = src1->ne[1];
                p_ids_src1    = rc.ids_src1;
                p_ids_dst     = rc.ids_dst;
                p_bounds      = rc.bounds;
                p_tile_off    = rc.tile_off;
                p_tile_expert = rc.tile_expert;
            } else {
                p_ids_src1    = ids_src1.alloc(n_assign);
                p_ids_dst     = ids_dst.alloc(n_assign);
                p_bounds      = expert_bounds.alloc(ne02 + 1);
                p_tile_off    = tile_off.alloc(ne02 + 1);
                p_tile_expert = tile_expert.alloc(max_tiles);
            }
            route(p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert);
        }

        // gate/up share src1: reuse the quantized activations
        const bool x_hit = rc.x_gen == ctx.graph_gen && rc.x == src1 && rc.x_data == src1->data &&
            rc.x_ne0 == src1->ne[0] && rc.x_ne1 == src1->ne[1] && rc.x_ne2 == src1->ne[2];
        if (x_hit) {
            p_xq = rc.xq;
        } else {
            rc.x_gen = 0;
            if (repack_rc_reserve(rc, (void **) &rc.xq, &rc.xq_cap, xq_bytes, capturing)) {
                p_xq      = rc.xq;
                rc.x_gen  = ctx.graph_gen;
                rc.x      = src1;
                rc.x_data = src1->data;
                rc.x_ne0  = src1->ne[0];
                rc.x_ne1  = src1->ne[1];
                rc.x_ne2  = src1->ne[2];
            } else {
                p_xq = src1_q8_1.alloc(xq_bytes);
            }
            quantize_x(p_xq);
        }
    } else {
        if (n_tokens > 1) {
            p_ids_src1    = ids_src1.alloc(n_assign);
            p_ids_dst     = ids_dst.alloc(n_assign);
            p_bounds      = expert_bounds.alloc(ne02 + 1);
            p_tile_off    = tile_off.alloc(ne02 + 1);
            p_tile_expert = tile_expert.alloc(max_tiles);
            route(p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert);
        }
        // quantize all activation columns once, natural order
        p_xq = (char *) repack_quantize_x(ctx, src1, ne10_padded, src1_q8_1, stream);
    }
    const block_q8_1 * xq = (const block_q8_1 *) p_xq;

    if (n_tokens == 1) {
        // decode: one matvec per slot; experts read directly from the
        // raw ids tensor in-kernel (no compaction kernels — launch
        // parity with canonical mmvq-id). Broadcast src1 (ne[1]==1, one
        // shared activation column for all slots) uses x-stride 0.
        const uint32_t xs_eff = src1->ne[1] == 1 ? 0u : (uint32_t) x_stride;
        switch (src0->type) {
            case GGML_TYPE_Q3_K: {
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q3k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q4_K: {
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q4k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q5_K: {
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q5k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q6_K: {
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q6k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q5_1: {
                if (ne00 <= 1024) {
                    launch_mul_mat_vec_q5_1_repacked_seg<true>(w, xq, dst_d, ne00, ne01, n_assign,
                        (const int32_t *) ids->data, expert_stride, xs_eff, dst_s1, stream);
                    break;
                }
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q5_1_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q8_0: {
                if (ne00 <= 1024) {
                    launch_mul_mat_vec_q8_0_repacked_seg<true>(w, xq, dst_d, ne00, ne01, n_assign,
                        (const int32_t *) ids->data, expert_stride, xs_eff, dst_s1, stream);
                    break;
                }
                if (ne01 >= 4096) {
                    const dim3 grid(ne01, n_assign, 1);
                    mul_mat_vec_q8_0_repacked<1, 1, true><<<grid, 64, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        (const int32_t *) ids->data, nullptr, nullptr,
                        (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
                } else {
                    const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                    mul_mat_vec_q8_0_repacked<2, 4, true><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        (const int32_t *) ids->data, nullptr, nullptr,
                        (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
                }
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    const dim3 grid((ne01 + MMQ_RP_BM - 1) / MMQ_RP_BM, max_tiles, 1);

    switch (src0->type) {
        case GGML_TYPE_Q3_K:
            mmq_gemm_q3k_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q4_K:
            static_assert(TN_ID == 1, "w1 kernel tiles 16 assignments");
            mmq_gemm_q4k_repacked_id_w1<2><<<dim3((ne01 + 63) / 64, max_tiles, 1), 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q5_K:
            mmq_gemm_q5k_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q6_K:
            mmq_gemm_q6k_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q5_1:
            mmq_gemm_q5_1_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q8_0:
            mmq_gemm_q8_0_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        default: GGML_ABORT("unsupported repack type");
    }
}

// Eligibility for the fused gate+up GLU path: both weights in the
// repack buffer type, Q4_K, identical shape; decode only (one output
// column per expert slot); SWIGLU or GEGLU.
bool ggml_cuda_repack_should_fuse_glu(const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * glu) {
    const ggml_tensor * wu = up->src[0];
    const ggml_tensor * wg = gate->src[0];
    if (wu->buffer == nullptr || wg->buffer == nullptr ||
        !ggml_backend_buft_is_cuda_repack(wu->buffer->buft) ||
        !ggml_backend_buft_is_cuda_repack(wg->buffer->buft)) {
        return false;
    }
    if (wu->type != wg->type || !ggml_are_same_shape(wu, wg)) {
        return false;
    }
    // Q8_0: dense decode only (shared expert)
    if (wu->type != GGML_TYPE_Q4_K && !(wu->type == GGML_TYPE_Q8_0 && up->src[2] == nullptr)) {
        return false;
    }
    const ggml_glu_op op = ggml_get_glu_op(glu);
    if (op != GGML_GLU_OP_SWIGLU && op != GGML_GLU_OP_GEGLU) {
        return false;
    }
    if (up->src[2] != nullptr) { // MUL_MAT_ID: one token
        return up->src[1]->ne[2] == 1 && glu->ne[2] == 1;
    }
    return up->src[1]->ne[1] == 1; // dense: one column
}

void ggml_cuda_mul_mat_repacked_fused_glu(ggml_backend_cuda_context & ctx,
        const ggml_tensor * up_w, const ggml_tensor * gate_w,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const int glu_op) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    const int64_t ne00 = up_w->ne[0];
    const int64_t ne01 = up_w->ne[1];
    cudaStream_t stream = ctx.stream();
    const uint8_t * wu = (const uint8_t *) up_w->data;
    const uint8_t * wg = (const uint8_t *) gate_w->data;
    float * dst_d = (float *) dst->data;

    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;

    if (ids == nullptr) {
        // dense decode column
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        const block_q8_1 * xq = repack_quantize_x(ctx, src1, ne10_padded, src1_q8_1, stream);
        if (up_w->type == GGML_TYPE_Q8_0) {
            const dim3 grid((ne01 + 3) / 4, 1, 1);
            mul_mat_vec_q8_0_repacked_glu<1><<<grid, 256, 0, stream>>>(
                wu, wg, xq, dst_d,
                (uint32_t) ne00, (uint32_t) ne01, glu_op);
            return;
        }
        const dim3 grid((ne01 + 7) / 8, 1, 1);
        mul_mat_vec_q4k_repacked_glu<false><<<grid, 256, 0, stream>>>(
            wu, wg, xq, dst_d,
            (uint32_t) ne00, (uint32_t) ne01, glu_op,
            nullptr, nullptr, nullptr, 0, 0, 0, 0);
        return;
    }

    // MoE decode: same routing machinery as the unfused ID path
    const int64_t ne02 = up_w->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_assign = n_expert_used; // one token
    const size_t expert_stride = repack_gcn_nbytes(up_w->type, ne00, ne01);
    GGML_ASSERT(dst->nb[1] == (size_t) dst->ne[0] * sizeof(float));
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    const block_q8_1 * xq = repack_quantize_x(ctx, src1, ne10_padded, src1_q8_1, stream);
    const uint32_t xs_eff = src1->ne[1] == 1 ? 0u : (uint32_t) x_stride;
    // one row per wave: the 2-row kernel needs 51 VGPRs (4 waves/SIMD) and is
    // latency-bound at ~380 GB/s; ROWS=1 runs 41 vs 49 us per call in decode
    const dim3 grid((ne01 + 3) / 4, n_assign, 1);
    mul_mat_vec_q4k_repacked_glu<true, 1><<<grid, 256, 0, stream>>>(
        wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op,
        (const int32_t *) ids->data, nullptr, nullptr, (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
}

// ---------------------------------------------------------------------
// buffer type
// ---------------------------------------------------------------------

struct ggml_backend_cuda_repack_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_repack_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_repack_buffer_type_get_name;
}

static void ggml_backend_cuda_repack_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_cuda_repack_tensor_supported(tensor));

    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2]; // experts (1 for plain 2D weights)

    const size_t src_stride = ggml_nbytes(tensor) / ne2;
    const size_t dst_stride = repack_gcn_nbytes(tensor->type, ne0, ne1);
    std::vector<uint8_t> staged(dst_stride * ne2);
    for (int64_t e = 0; e < ne2; e++) {
        const uint8_t * src_e = (const uint8_t *) data + e * src_stride;
        uint8_t       * dst_e = staged.data() + e * dst_stride;
        switch (tensor->type) {
            case GGML_TYPE_Q3_K: repack_q3k_host ((const block_q3_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q4_K: repack_q4k_host ((const block_q4_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q5_K: repack_q5k_host ((const block_q5_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q6_K: repack_q6k_host ((const block_q6_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q8_0: repack_q8_0_host((const block_q8_0 *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q5_1: repack_q5_1_host((const block_q5_1 *) src_e, dst_e, ne0, ne1); break;
            default:             GGML_ABORT("unsupported repack type");
        }
    }

    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(tensor->data, staged.data(), staged.size(),
        cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_repack_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    GGML_ABORT("repacked tensors cannot be read back (GGML_CUDA_REPACK)");
    GGML_UNUSED_VARS(buffer, tensor, data, offset, size);
}

static ggml_backend_buffer_t ggml_backend_cuda_repack_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;

    ggml_backend_buffer_t buffer =
        ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(ctx->device), size);
    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.set_tensor  = ggml_backend_cuda_repack_buffer_set_tensor;
    buffer->iface.get_tensor  = ggml_backend_cuda_repack_buffer_get_tensor;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (ggml_cuda_repack_tensor_supported(tensor)) {
        return repack_gcn_nbytes(tensor->type, tensor->ne[0], tensor->ne[1]) * tensor->ne[2];
    }
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_repack_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_cuda_repack_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_cuda_repack_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_cuda_repack_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ ggml_backend_cuda_repack_buffer_type_get_alloc_size,
    /* .is_host        = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    // Default-on for GCN; GGML_CUDA_REPACK=0 opts out. (Repacked
    // weights cannot be read back: llama-quantize/save from a loaded
    // model needs the opt-out.)
    const char * env = getenv("GGML_CUDA_REPACK");
    if (env != nullptr && env[0] == '0') {
        return nullptr;
    }
    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }
    if (!GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[device].cc)) {
        return nullptr;
    }

    static ggml_backend_buffer_type buft_storage[GGML_CUDA_MAX_DEVICES];
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {};

    if (!initialized[device]) {
        buft_storage[device] = {
            /* .iface   = */ ggml_backend_cuda_repack_buffer_type_interface,
            /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
            /* .context = */ new ggml_backend_cuda_repack_buffer_type_context{
                                 device, GGML_CUDA_NAME + std::to_string(device) + "_Repacked"},
        };
        initialized[device] = true;
    }
    return &buft_storage[device];
}
