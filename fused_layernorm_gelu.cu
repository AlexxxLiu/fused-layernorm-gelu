// Fused LayerNorm + GELU CUDA kernel.
//
// Baseline (PyTorch) does:
//   y = layer_norm(x, gamma, beta)   -> reads x, writes y      (2 HBM passes)
//   z = gelu(y)                      -> reads y, writes z      (2 HBM passes)
//
// Fused version does it in one pass: read x once, write z once.
// For memory-bound shapes this is the whole story -- the FLOPs are trivial,
// the win comes from halving HBM traffic and killing one kernel launch.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// tanh approximation of GELU, same one PyTorch uses for approximate='tanh'
__device__ __forceinline__ float gelu_tanh(float x) {
    const float kSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/pi)
    const float kCoeff = 0.044715f;
    float inner = kSqrt2OverPi * (x + kCoeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// One block handles one row (i.e. one token's hidden vector).
// BLOCK must be a multiple of WARP_SIZE and <= 1024.
template <int BLOCK>
__global__ void fused_layernorm_gelu_kernel(
    const float* __restrict__ input,   // [N, H]
    const float* __restrict__ gamma,   // [H]
    const float* __restrict__ beta,    // [H]
    float* __restrict__ output,        // [N, H]
    int H,
    float eps) {

    const int row = blockIdx.x;
    const float* __restrict__ in_row = input + (long long)row * H;
    float* __restrict__ out_row = output + (long long)row * H;

    constexpr int NUM_WARPS = BLOCK / WARP_SIZE;
    __shared__ float s_sum[NUM_WARPS];
    __shared__ float s_sqsum[NUM_WARPS];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    // ---- pass 1: accumulate sum and sum-of-squares for this row ----
    float local_sum = 0.0f;
    float local_sqsum = 0.0f;
    for (int i = threadIdx.x; i < H; i += BLOCK) {
        float v = in_row[i];
        local_sum += v;
        local_sqsum += v * v;
    }

    // reduce within each warp
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum   += __shfl_down_sync(FULL_MASK, local_sum, offset);
        local_sqsum += __shfl_down_sync(FULL_MASK, local_sqsum, offset);
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sqsum[warp_id] = local_sqsum;
    }
    __syncthreads();

    // reduce across warps using warp 0
    if (warp_id == 0) {
        float v  = (lane < NUM_WARPS) ? s_sum[lane]   : 0.0f;
        float v2 = (lane < NUM_WARPS) ? s_sqsum[lane] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v  += __shfl_down_sync(FULL_MASK, v, offset);
            v2 += __shfl_down_sync(FULL_MASK, v2, offset);
        }
        if (lane == 0) {
            float mean = v / (float)H;
            // E[x^2] - E[x]^2. Fine for fp32 at these magnitudes; swap in
            // Welford here if you ever push this to fp16 accumulation.
            float var = v2 / (float)H - mean * mean;
            var = var < 0.0f ? 0.0f : var;
            s_mean = mean;
            s_rstd = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    const float mean = s_mean;
    const float rstd = s_rstd;

    // ---- pass 2: normalize, affine, gelu, write ----
    for (int i = threadIdx.x; i < H; i += BLOCK) {
        float v = (in_row[i] - mean) * rstd;
        v = v * gamma[i] + beta[i];
        out_row[i] = gelu_tanh(v);
    }
}

// Pick a block size that keeps threads busy without oversubscribing tiny rows.
static int choose_block_size(int H) {
    if (H >= 2048) return 1024;
    if (H >= 1024) return 512;
    if (H >= 256)  return 256;
    return 128;
}

torch::Tensor fused_layernorm_gelu(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "only fp32 for now");
    TORCH_CHECK(gamma.is_cuda() && beta.is_cuda(), "gamma/beta must be CUDA tensors");

    auto x = input.contiguous();
    auto g = gamma.contiguous();
    auto b = beta.contiguous();

    const int H = x.size(-1);
    const long long N = x.numel() / H;
    TORCH_CHECK(g.numel() == H && b.numel() == H, "gamma/beta must have size H");

    auto out = torch::empty_like(x);

    const int block = choose_block_size(H);
    const dim3 grid((unsigned)N);
    auto stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH(BS)                                                        \
        fused_layernorm_gelu_kernel<BS><<<grid, BS, 0, stream>>>(             \
            x.data_ptr<float>(), g.data_ptr<float>(), b.data_ptr<float>(),    \
            out.data_ptr<float>(), H, (float)eps)

    switch (block) {
        case 1024: LAUNCH(1024); break;
        case 512:  LAUNCH(512);  break;
        case 256:  LAUNCH(256);  break;
        default:   LAUNCH(128);  break;
    }
    #undef LAUNCH

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_layernorm_gelu", &fused_layernorm_gelu,
          "Fused LayerNorm + GELU (CUDA, fp32)",
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5);
}
