#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream()

#define THREADS 256
static inline int grid1d(int n) { return (n + THREADS - 1) / THREADS; }

__global__ void k_iter_histogram(
    const int64_t* __restrict__ iters,  // [B]         per-sample stop iteration
    int64_t*       __restrict__ hist,   // [num_bins]  output counts (pre-zeroed)
    int B, int num_bins
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    long long v = (long long)iters[b];
    if (v < 0) v = 0;
    if (v >= num_bins) v = num_bins - 1;
    atomicAdd(reinterpret_cast<unsigned long long*>(hist) + v, 1ULL);
}

// Histogram of `iters` (any integer dtype) into a length-num_bins int64 tensor on the
// same CUDA device. Mirrors torch.bincount(iters.clamp(min=0), minlength=num_bins),
// also folding any value >= num_bins into the last bin.
torch::Tensor iter_histogram_cuda(torch::Tensor iters, int64_t num_bins) {
    TORCH_CHECK(iters.is_cuda(), "iters must be a CUDA tensor");
    TORCH_CHECK(num_bins > 0, "num_bins must be positive");
    auto iters_i64 = iters.to(torch::kInt64).contiguous();
    int B = (int)iters_i64.numel();
    auto hist = torch::zeros(
        {num_bins}, torch::dtype(torch::kInt64).device(iters.device()));
    if (B == 0) return hist;
    auto stream = at::cuda::getCurrentCUDAStream();
    k_iter_histogram<<<grid1d(B), THREADS, 0, stream>>>(
        iters_i64.data_ptr<int64_t>(), hist.data_ptr<int64_t>(), B, (int)num_bins);
    return hist;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("iter_histogram", &iter_histogram_cuda,
          "Histogram of per-sample stop iterations into [0, num_bins) for the "
          "iter_speedup warm-up (CUDA)",
          py::arg("iters"), py::arg("num_bins"));
}
