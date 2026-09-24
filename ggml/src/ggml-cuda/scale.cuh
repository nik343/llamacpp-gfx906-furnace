#include "common.cuh"

#define CUDA_SCALE_BLOCK_SIZE 256

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// SCALE followed by SILU on its result, written to silu_node
// yq (optional): also write the result as q8_1 blocks (see ggml_cuda_repack_xq_emit_target)
void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * scale_node, ggml_tensor * silu_node, void * yq = nullptr);
