#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <limits>
#include <type_traits>

// MODE: binary memory BP, oscillation/threshold prior switching, self-corrected min-sum.
template <typename T, int MODE>
__global__ void decode_kernel(
    const T* base, const uint8_t* syndrome, const int* row, const int* col,
    const int* vn, int N, int M, int E, int VD, int max_iter, T alpha,
    int interval, T strength, T* prior, T* sent, T* incoming, int* flips,
    T* llr, uint8_t* hard, int64_t* iterations, int64_t* converged) {
    int b = blockIdx.x, tid = threadIdx.x;
    base += (int64_t)b * N;
    syndrome += (int64_t)b * M;
    prior += (int64_t)b * N;
    llr += (int64_t)b * N;
    hard += (int64_t)b * N;
    flips += (int64_t)b * N;
    sent += (int64_t)b * (E + 1);
    incoming += (int64_t)b * (E + 1);
    __shared__ int mismatch, selected, least;
    for (int n = tid; n < N; n += blockDim.x) {
        prior[n] = llr[n] = base[n];
        hard[n] = base[n] <= T(0);
        flips[n] = 0;
    }
    for (int e = tid; e < E; e += blockDim.x) sent[e] = base[col[e]];
    __syncthreads();
    for (int iter = 1; iter <= max_iter; ++iter) {
        // Each check owns all of its outgoing edges; no inter-check writes overlap.
        for (int c = tid; c < M; c += blockDim.x) {
            for (int e = row[c]; e < row[c + 1]; ++e) {
                T value;
                if constexpr (MODE == 2) {
                    T mag = std::numeric_limits<T>::infinity();
                    bool neg = syndrome[c] != 0;
                    for (int j = row[c]; j < row[c + 1]; ++j) if (j != e) {
                        T q = sent[j];
                        neg ^= q < T(0);
                        mag = fmin(mag, fabs(q));
                    }
                    value = neg ? -mag : mag;
                } else {
                    T product = syndrome[c] ? T(-1) : T(1);
                    for (int j = row[c]; j < row[c + 1]; ++j)
                        if (j != e) product *= tanh(sent[j] / T(2));
                    // Saturation keeps degree-one checks and strong evidence finite.
                    T bound = T(1) - std::numeric_limits<T>::epsilon();
                    product = fmax(-bound, fmin(bound, product));
                    value = T(2) * atanh(product);
                }
                incoming[e] = value;
            }
        }
        __syncthreads();
        for (int n = tid; n < N; n += blockDim.x) {
            T total = T(0);
            for (int k = 0; k < VD; ++k) {
                int e = vn[n * VD + k];
                if (e < E) total += incoming[e];
            }
            T posterior = prior[n] + total / (MODE == 0 ? alpha : T(1));
            uint8_t decision = posterior <= T(0);
            if constexpr (MODE == 1) flips[n] += decision != hard[n];
            hard[n] = decision;
            llr[n] = posterior;
        }
        __syncthreads();
        if (tid == 0) mismatch = 0;
        __syncthreads();
        for (int c = tid; c < M; c += blockDim.x) {
            uint8_t parity = 0;
            for (int e = row[c]; e < row[c + 1]; ++e) parity ^= hard[col[e]];
            if (parity != syndrome[c]) atomicOr(&mismatch, 1);
        }
        __syncthreads();
        if (mismatch == 0 || iter == max_iter) {
            if (tid == 0) {
                iterations[b] = iter;
                converged[b] = mismatch == 0;
            }
            break;
        }
        if constexpr (MODE == 1) {
            if (iter % interval == 0) {
                if (tid == 0) {
                    selected = -1;
                    least = 0;
                    int count = 0;
                    T reliability = std::numeric_limits<T>::infinity();
                    for (int n = 0; n < N; ++n) {
                        T r = fabs(llr[n]);
                        if (r < fabs(llr[least])) least = n;
                        if (flips[n] > count ||
                            (flips[n] > 0 && flips[n] == count && r < reliability)) {
                            selected = n;
                            count = flips[n];
                            reliability = r;
                        }
                    }
                    if (selected >= 0) flips[selected] = 0;
                }
                __syncthreads();
                for (int n = tid; n < N; n += blockDim.x) {
                    T next = (n == selected || n == least) ? -strength : base[n];
                    llr[n] += next - prior[n];
                    prior[n] = next;
                }
                __syncthreads();
            }
        }
        for (int e = tid; e < E; e += blockDim.x) {
            T candidate = llr[col[e]] - incoming[e];
            if constexpr (MODE == 2) {
                // Recompute the extrinsic sum to avoid inf-inf at degree-one checks.
                int n = col[e];
                candidate = prior[n];
                for (int k = 0; k < VD; ++k) {
                    int j = vn[n * VD + k];
                    if (j < E && j != e) candidate += incoming[j];
                }
                T previous = sent[e];
                if ((candidate < T(0) && previous > T(0)) ||
                    (candidate > T(0) && previous < T(0))) candidate = T(0);
            }
            sent[e] = candidate;
        }
        __syncthreads();
    }
}

std::vector<torch::Tensor> decode(
    torch::Tensor base, torch::Tensor syndrome, torch::Tensor row,
    torch::Tensor col, torch::Tensor vn, int64_t max_iter, int64_t mode,
    double alpha, int64_t interval, double strength) {
    for (auto t : {base, syndrome, row, col, vn}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "inputs must be contiguous CUDA tensors");
        TORCH_CHECK(t.device() == base.device(), "inputs must share a CUDA device");
    }
    TORCH_CHECK(base.dim() == 2 && syndrome.dim() == 2 && row.dim() == 1 &&
                col.dim() == 1 && vn.dim() == 2, "invalid input dimensions");
    int64_t B = base.size(0), N = base.size(1), M = syndrome.size(1), E = col.numel();
    TORCH_CHECK(syndrome.size(0) == B && row.numel() == M + 1 && vn.size(0) == N + 1,
                "graph and batch dimensions do not match");
    TORCH_CHECK(syndrome.scalar_type() == torch::kUInt8 && row.scalar_type() == torch::kInt32 &&
                col.scalar_type() == torch::kInt32 && vn.scalar_type() == torch::kInt32,
                "syndrome must be uint8 and graph indices int32");
    TORCH_CHECK(N > 0 && N <= INT_MAX && M <= INT_MAX && E < INT_MAX && B <= INT_MAX &&
                vn.size(1) <= INT_MAX && max_iter > 0 && max_iter <= INT_MAX,
                "unsupported graph size or iteration limit");
    TORCH_CHECK(mode >= 0 && mode <= 2 && std::isfinite(alpha) && alpha > 0 &&
                interval > 0 && interval <= INT_MAX && std::isfinite(strength) && strength > 0,
                "invalid decoder parameters");
    c10::cuda::CUDAGuard guard(base.device());
    auto prior = torch::empty_like(base), llr = torch::empty_like(base);
    auto sent = torch::empty({B, E + 1}, base.options());
    auto incoming = torch::empty_like(sent);
    auto flips = torch::empty({B, N}, base.options().dtype(torch::kInt32));
    auto hard = torch::empty({B, N}, base.options().dtype(torch::kUInt8));
    auto iterations = torch::zeros({B}, base.options().dtype(torch::kInt64));
    auto converged = torch::zeros_like(iterations);
    if (B > 0) {
        AT_DISPATCH_FLOATING_TYPES(base.scalar_type(), "bp_interventions", [&] {
            auto launch = [&](auto tag) {
                constexpr int MODE = decltype(tag)::value;
                decode_kernel<scalar_t, MODE><<<B, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    base.data_ptr<scalar_t>(), syndrome.data_ptr<uint8_t>(), row.data_ptr<int>(),
                    col.data_ptr<int>(), vn.data_ptr<int>(), N, M, E, vn.size(1), max_iter,
                    scalar_t(alpha), interval, scalar_t(strength), prior.data_ptr<scalar_t>(),
                    sent.data_ptr<scalar_t>(), incoming.data_ptr<scalar_t>(), flips.data_ptr<int>(),
                    llr.data_ptr<scalar_t>(), hard.data_ptr<uint8_t>(),
                    iterations.data_ptr<int64_t>(), converged.data_ptr<int64_t>());
            };
            if (mode == 0) launch(std::integral_constant<int, 0>{});
            if (mode == 1) launch(std::integral_constant<int, 1>{});
            if (mode == 2) launch(std::integral_constant<int, 2>{});
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {llr, hard, iterations, converged};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode", &decode); }
