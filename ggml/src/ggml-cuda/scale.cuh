#include "common.cuh"

#define CUDA_SCALE_BLOCK_SIZE 256

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// SCALE followed by SILU on its result, written to silu_node
void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * scale_node, ggml_tensor * silu_node);
