#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);

bool ggml_cuda_qsa_pool_ok(const ggml_tensor * get_rows, const ggml_tensor * scale, const ggml_tensor * rms_norm,
        const ggml_tensor * mul, const ggml_tensor * rope);

void ggml_cuda_op_qsa_pool_norm_rope(ggml_backend_cuda_context & ctx, const ggml_tensor * get_rows, int r,
        const ggml_tensor * scale, const ggml_tensor * rms_norm, const ggml_tensor * mul, ggml_tensor * rope);

void ggml_cuda_op_qsa_score(ggml_backend_cuda_context & ctx, const ggml_tensor * keys, const ggml_tensor * q, int n_head,
        const ggml_tensor * bias, const ggml_tensor * cell_blk, const ggml_tensor * mask, ggml_tensor * dst);
