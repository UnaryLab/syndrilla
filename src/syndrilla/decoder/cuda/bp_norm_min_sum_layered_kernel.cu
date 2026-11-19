#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>

template <typename scalar_t>
__global__ void layer_update_kernel(
    scalar_t* lv, scalar_t* previous, const scalar_t* syndrome,
    const int64_t* cols, const int64_t* layer, scalar_t beta,
    int64_t B, int64_t M, int64_t D, int64_t N1, int64_t L) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * L) return;
    int64_t b = idx / L, c = layer[idx % L], off = (b * M + c) * D;
    scalar_t min0 = (scalar_t)INFINITY, min1 = (scalar_t)INFINITY;
    int64_t arg = -1;
    bool parity = syndrome[b * M + c] != (scalar_t)0;
    for (int64_t k = 0; k < D; ++k) {
        int64_t v = cols[c * D + k];
        if (v == N1 - 1) continue;
        scalar_t a = lv[b * N1 + v] - previous[off + k];
        parity ^= a <= (scalar_t)0;
        scalar_t mag = a < (scalar_t)0 ? -a : a;
        if (mag < min0) { min1 = min0; min0 = mag; arg = k; }
        else if (mag < min1) min1 = mag;
    }
    for (int64_t k = 0; k < D; ++k) {
        int64_t v = cols[c * D + k];
        if (v == N1 - 1) continue;
        scalar_t old = previous[off + k];
        scalar_t a = lv[b * N1 + v] - old;
        scalar_t msg = beta * (k == arg ? min1 : min0);
        if ((a <= (scalar_t)0) ^ parity) msg = -msg;
        scalar_t delta = msg - old;
        lv[b * N1 + v] = lv[b * N1 + v] + delta;
        previous[off + k] = msg;
    }
}

void layer_update(torch::Tensor lv, torch::Tensor previous, torch::Tensor syndrome,
                  torch::Tensor cols, torch::Tensor layer, double beta) {
    for (const auto& t : {lv, previous, syndrome, cols, layer}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "layer tensors must be contiguous CUDA tensors");
        TORCH_CHECK(t.device() == lv.device(), "layer tensors must share a device");
    }
    TORCH_CHECK(lv.dim() == 2 && previous.dim() == 3 && cols.dim() == 2 &&
                syndrome.dim() == 2 && layer.dim() == 1, "invalid layer tensor dimensions");
    TORCH_CHECK(cols.scalar_type() == at::kLong && layer.scalar_type() == at::kLong,
                "layer indices must be int64");
    TORCH_CHECK(previous.scalar_type() == lv.scalar_type() && syndrome.scalar_type() == lv.scalar_type(),
                "layer values must share a dtype");
    int64_t B = lv.size(0), M = cols.size(0), D = cols.size(1), L = layer.numel();
    TORCH_CHECK(previous.size(0) == B && previous.size(1) == M && previous.size(2) == D &&
                syndrome.size(0) == B && syndrome.size(1) == M, "invalid layer tensor shapes");
    if (B * L == 0) return;
    int64_t blocks = (B * L + 255) / 256;
    TORCH_CHECK(blocks <= INT_MAX, "layer launch exceeds grid limit");
    c10::cuda::CUDAGuard guard(lv.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                   lv.scalar_type(), "layer_update", [&] {
        layer_update_kernel<scalar_t><<<blocks, 256, 0, stream>>>(
            lv.data_ptr<scalar_t>(), previous.data_ptr<scalar_t>(), syndrome.data_ptr<scalar_t>(),
            cols.data_ptr<int64_t>(), layer.data_ptr<int64_t>(), (scalar_t)beta,
            B, M, D, lv.size(1), L);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_update", &layer_update);
}
