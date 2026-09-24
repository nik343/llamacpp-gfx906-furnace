#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// cgraph (optional): lets the op read s0 through an elided get_rows (see ggml_cuda_gdn_state_gather_elidable)
void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cgraph * cgraph = nullptr);

// true when get_rows node gr only feeds a gated_delta_net state input, so it can be skipped
bool ggml_cuda_gdn_state_gather_elidable(const ggml_cgraph * cgraph, const ggml_tensor * gr);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache, const ggml_cgraph * cgraph = nullptr);
