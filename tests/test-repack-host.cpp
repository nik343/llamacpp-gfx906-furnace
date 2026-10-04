// Host-only layout check for the GCN repack formats (repack-gcn.cu): quantize random rows with
// ggml, repack them on the host, decode the planes by the layout rules the kernels use, and
// compare against ggml's reference dequantizer. No device needed.
//
//   exact:   Q4_0, IQ4_NL, Q3_K (relabelled to Q6_K planes), Q6_K
//   folded:  IQ4_XS, IQ3_S (sub-block scale rounded to fp16: <= 2^-10 relative per weight)

#include "ggml.h"

// host-only hooks exported by repack-gcn.cu (the header pulls in HIP, so declare them here)
size_t ggml_cuda_repack_nbytes_for_test(ggml_type type, int64_t ne0, int64_t ne1);
int    ggml_cuda_repack_eff_type_for_test(ggml_type type);
void   ggml_cuda_repack_host_for_test(ggml_type type, const void * src, void * dst, int64_t ne0, int64_t ne1);

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int8_t kvalues_iq4nl_host[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

static int64_t nsp_of(int64_t ne0) {
    const int64_t n = ne0 / 32;
    return (n & (n - 1)) == 0 ? n + 1 : n;
}

static float f16(const uint8_t * p) {
    ggml_fp16_t h;
    memcpy(&h, p, 2);
    return ggml_fp16_to_fp32(h);
}

// nibble + fp16 scale planes: Q4_0 / IQ4_NL / IQ4_XS / IQ3_S
static float decode_nib(const uint8_t * dst, ggml_type type, int64_t ne0, int64_t ne1, int64_t row, int64_t k) {
    const int64_t nsp = nsp_of(ne0);
    const int64_t sb = k / 32, kk = k % 32;
    const size_t idx = (size_t) row * nsp + sb;
    const uint8_t byte = dst[idx * 16 + (kk % 16)];
    const int n = kk < 16 ? (byte & 0x0F) : (byte >> 4);
    const float d = f16(dst + (size_t) ne1 * nsp * 16 + idx * 2);
    int v;
    switch (type) {
        case GGML_TYPE_Q4_0:   v = n - 8; break;
        case GGML_TYPE_IQ3_S:  v = 2 * n - 15; break;
        default:               v = kvalues_iq4nl_host[n]; break;
    }
    return d * (float) v;
}

// Q6_K planes (also the Q3_K relabel): nib, h2, signed per-16 scale pairs, d per superblock
static float decode_q6k(const uint8_t * dst, int64_t ne0, int64_t ne1, int64_t row, int64_t k) {
    const int64_t nsp = nsp_of(ne0), n_blocks = ne0 / 256;
    const size_t nib_len = (size_t) ne1 * nsp * 16, h2_len = (size_t) ne1 * nsp * 8, sm_len = (size_t) ne1 * nsp * 2;
    const int64_t sb = k / 32, kk = k % 32;
    const size_t idx = (size_t) row * nsp + sb;
    const int j = (kk % 16) / 4, bb = kk % 4, hi_half = kk >= 16;
    const uint8_t nb = dst[idx * 16 + j * 4 + bb];
    const int lo = hi_half ? (nb >> 4) : (nb & 0x0F);
    const uint8_t h2 = dst[nib_len + idx * 8 + 2 * j + hi_half];
    const int hi = (h2 >> (2 * bb)) & 3;
    const int q6 = lo | (hi << 4);
    const int8_t sc = (int8_t) dst[nib_len + h2_len + idx * 2 + hi_half];
    const float d = f16(dst + nib_len + h2_len + sm_len + ((size_t) row * n_blocks + sb / 8) * 2);
    const float dl = d * (float) sc;
    return dl * (float) (q6 - 32);
}

static bool check(ggml_type type, int64_t ne0, int64_t ne1, double tol_rel, uint64_t seed) {
    std::vector<float> f((size_t) ne0 * ne1);
    uint64_t s = seed;
    for (auto & x : f) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        x = ((float) ((s >> 33) & 0xFFFFFF) / 8388608.0f - 1.0f) * 0.5f;
    }
    const size_t row_size = ggml_row_size(type, ne0);
    std::vector<uint8_t> q(row_size * ne1);
    ggml_quantize_init(type);
    ggml_quantize_chunk(type, f.data(), q.data(), 0, ne1, ne0, nullptr);

    std::vector<uint8_t> dst(ggml_cuda_repack_nbytes_for_test(type, ne0, ne1));
    ggml_cuda_repack_host_for_test(type, q.data(), dst.data(), ne0, ne1);

    const ggml_type eff = (ggml_type) ggml_cuda_repack_eff_type_for_test(type);
    std::vector<float> ref(ne0);
    const ggml_type_traits * tt = ggml_get_type_traits(type);
    double max_rel = 0.0; int64_t bad = 0;
    for (int64_t row = 0; row < ne1; row++) {
        tt->to_float(q.data() + row * row_size, ref.data(), ne0);
        for (int64_t k = 0; k < ne0; k++) {
            const float got = eff == GGML_TYPE_Q6_K ? decode_q6k(dst.data(), ne0, ne1, row, k)
                                                    : decode_nib(dst.data(), type, ne0, ne1, row, k);
            const double den = std::max(std::fabs((double) ref[k]), 1e-30);
            const double rel = std::fabs((double) got - ref[k]) / den;
            if (tol_rel == 0.0 ? got != ref[k] : rel > tol_rel) {
                if (bad < 3) {
                    printf("    mismatch row %ld k %ld: got %.9g ref %.9g\n", (long) row, (long) k, got, ref[k]);
                }
                bad++;
            }
            max_rel = std::max(max_rel, rel);
        }
    }
    printf("  %-7s ne0 %5ld ne1 %3ld (%s planes): %s, %ld mismatches, max rel %.3e\n", ggml_type_name(type),
        (long) ne0, (long) ne1, ggml_type_name(eff), bad == 0 ? "OK" : "FAIL", (long) bad, max_rel);
    return bad == 0;
}

int main() {
    bool ok = true;
    // ne0 = 768 (nsp = 24, no pad) and 512 / 1024 (power of two: padded sub-block)
    for (int64_t ne0 : { 768, 512 }) {
        ok &= check(GGML_TYPE_Q4_0,   ne0, 5, 0.0,    11);
        ok &= check(GGML_TYPE_IQ4_NL, ne0, 5, 0.0,    12);
        ok &= check(GGML_TYPE_Q3_K,   ne0, 5, 0.0,    13);
        ok &= check(GGML_TYPE_Q6_K,   ne0, 3, 0.0,    14);
        ok &= check(GGML_TYPE_IQ4_XS, ne0, 5, 1.0e-3, 15);
        ok &= check(GGML_TYPE_IQ3_S,  ne0, 5, 1.0e-3, 16);
    }
    ok &= check(GGML_TYPE_Q4_0, 1024, 7, 0.0, 17);
    printf("%s\n", ok ? "repack host check: OK" : "repack host check: FAILED");
    return ok ? 0 : 1;
}
