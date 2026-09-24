#include "scale.cuh"
#include "unary.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

static __global__ void scale_f32(const float * x, float * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = scale * x[i] + bias;
    }
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32, launch_params, x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    scale_f32_cuda(src0_d, dst_d, scale, bias, ggml_nelements(src0), stream);
}

// SCALE -> SILU (qwen4exp hc mix): one pass, same rounding as the two ops. yq: also write
// the result as q8_1 (the arithmetic of quantize_q8_1; needs nelements % 32 == 0)
static __global__ void scale_silu_f32(const float * x, float * dst, const float scale, const float bias, const int64_t nelements,
        block_q8_1 * yq) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        const float v = ggml_cuda_op_silu_single(scale * x[i] + bias);
        dst[i] = v;
        if (yq != nullptr) {
            float amax = fabsf(v);
            float sum  = v;
            amax = warp_reduce_max<QK8_1>(amax);
            sum  = warp_reduce_sum<QK8_1>(sum);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
            yq[i / QK8_1].qs[i % QK8_1] = q;
            if (i % QK8_1 == 0) {
                yq[i / QK8_1].ds = make_half2(d, sum);
            }
        }
    }
}

void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, ggml_tensor * scale_node, ggml_tensor * silu_node, void * yq) {
    const ggml_tensor * src0 = scale_node->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && silu_node->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(silu_node));
    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(silu_node));

    float scale;
    float bias;
    memcpy(&scale, (float *) scale_node->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) scale_node->op_params + 1, sizeof(float));

    const int64_t nelements = ggml_nelements(src0);
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    if (nelements % QK8_1 != 0) {
        yq = nullptr;
    }
    scale_silu_f32<<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) src0->data, (float *) silu_node->data, scale, bias, nelements, (block_q8_1 *) yq);
}
