#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// hc_post with its weights given as raw values w, used as s_out * sigmoid(s_in * w)
void ggml_cuda_op_dsv4_hc_post_act(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * raw_post, float s_in, float s_out);
