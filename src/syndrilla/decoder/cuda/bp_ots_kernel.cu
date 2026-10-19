// OTS prior switch, Algorithm 1 of Dimitris Chytas, Michele Pacenti,
// Nithin Raveendran, Mark F. Flanagan and Bane Vasic,
// Enhanced Message-Passing Decoding of Degenerate Quantum Codes
// Utilizing Trapping Set Dynamics, IEEE Communications Letters, 2024.
// https://doi.org/10.1109/LCOMM.2024.3356312
// Empty oscillator set: bias only the least-reliable node (undefined j1 in paper).
// The base BP loop owns VN updates, hard decisions, convergence and layouts.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <algorithm>
#include <limits>

constexpr int THREADS = 128;

template <typename T>
__global__ void prior_switch(T* llr, T* prior, const T* channel, int64_t* count,
                             const bool* active, const T* incoming, int64_t* nodes, int64_t n, int64_t degree, T strength) {
    const int row = blockIdx.x, thread = threadIdx.x;
    if (!active[row]) {
        if (thread == 0) nodes[2 * row] = nodes[2 * row + 1] = -1;
        return;
    }
    __shared__ int64_t largest[THREADS], oscillator[THREADS], least[THREADS];
    __shared__ T osc_mag[THREADS], least_mag[THREADS];
    int64_t best_count = 0, best_node = n, least_node = n;
    T best_mag = INFINITY, low_mag = INFINITY;
    for (int64_t v = thread; v < n; v += THREADS) {
        const T mag = fabs(llr[row * (n + 1) + v]);
        const int64_t flips = count[row * n + v];
        if (mag < low_mag || (mag == low_mag && v < least_node)) {
            low_mag = mag; least_node = v;
        }
        if (flips > best_count || (flips > 0 && flips == best_count &&
                (mag < best_mag || (mag == best_mag && v < best_node)))) {
            best_count = flips; best_mag = mag; best_node = v;
        }
    }
    largest[thread] = best_count; oscillator[thread] = best_node; osc_mag[thread] = best_mag;
    least[thread] = least_node; least_mag[thread] = low_mag;
    __syncthreads();
    for (int width = THREADS / 2; width; width /= 2) {
        if (thread < width) {
            const int next = thread + width;
            if (least_mag[next] < least_mag[thread] ||
                    (least_mag[next] == least_mag[thread] && least[next] < least[thread])) {
                least[thread] = least[next]; least_mag[thread] = least_mag[next];
            }
            if (largest[next] > largest[thread] || (largest[next] > 0 && largest[next] == largest[thread] &&
                    (osc_mag[next] < osc_mag[thread] ||
                    (osc_mag[next] == osc_mag[thread] && oscillator[next] < oscillator[thread])))) {
                largest[thread] = largest[next]; oscillator[thread] = oscillator[next]; osc_mag[thread] = osc_mag[next];
            }
        }
        __syncthreads();
    }
    const int64_t selected = largest[0] > 0 ? oscillator[0] : -1;
    if (thread == 0) {
        nodes[2 * row] = selected; nodes[2 * row + 1] = least[0];
        if (selected >= 0) count[row * n + selected] = 0;
    }
    for (int64_t v = thread; v < n; v += THREADS) {
        const int64_t index = row * n + v;
        const T value = (v == selected || v == least[0]) ? -strength : channel[index];
        prior[index] = value;
        T posterior = value;
        for (int64_t slot = 0; slot < degree; ++slot)
            posterior += incoming[(row * degree + slot) * (n + 1) + v];
        llr[row * (n + 1) + v] = posterior;
    }
}

torch::Tensor switch_priors(torch::Tensor llr, torch::Tensor prior, torch::Tensor channel,
                           torch::Tensor counts, torch::Tensor active, torch::Tensor incoming, double strength) {
    TORCH_CHECK(llr.is_cuda() && prior.is_cuda() && channel.is_cuda() && counts.is_cuda() && active.is_cuda() && incoming.is_cuda(),
                "OTS tensors must be CUDA tensors");
    TORCH_CHECK(llr.is_contiguous() && prior.is_contiguous() && channel.is_contiguous() &&
                counts.is_contiguous() && active.is_contiguous() && incoming.is_contiguous(), "OTS tensors must be contiguous");
    TORCH_CHECK(counts.scalar_type() == torch::kInt64 && active.scalar_type() == torch::kBool,
                "OTS counts must be int64 and active must be bool");
    TORCH_CHECK(llr.scalar_type() == prior.scalar_type() && channel.scalar_type() == prior.scalar_type() && incoming.scalar_type() == prior.scalar_type(),
                "OTS floating dtypes must match");
    const int64_t b = prior.size(0), n = prior.size(1);
    TORCH_CHECK(llr.dim() == 2 && llr.size(0) == b && llr.size(1) == n + 1 &&
                counts.sizes() == prior.sizes() && channel.sizes() == prior.sizes() && active.numel() == b &&
                incoming.dim() == 3 && incoming.size(0) == b && incoming.size(2) == n + 1,
                "OTS tensor shapes do not match");
    c10::cuda::CUDAGuard guard(llr.device());
    auto nodes = torch::empty({b, int64_t(2)}, counts.options());
    if (b) {
        AT_DISPATCH_FLOATING_TYPES(llr.scalar_type(), "ots_prior_switch", [&] {
            prior_switch<scalar_t><<<b, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                llr.data_ptr<scalar_t>(), prior.data_ptr<scalar_t>(), channel.data_ptr<scalar_t>(),
                counts.data_ptr<int64_t>(), active.data_ptr<bool>(), incoming.data_ptr<scalar_t>(),
                nodes.data_ptr<int64_t>(), n, incoming.size(1), scalar_t(strength));
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return nodes;
}

// Probability-domain SPA, in the same padded check-slot order as bp_sum_prod.
// Only the CN arithmetic is new. The base extension still owns VN updates,
// hard decisions, syndrome checks, convergence, layouts and cap handling.
template <typename T>
__global__ void spa_factors(const T* values, const T* incoming, const int32_t* col,
                            const int64_t* active, T* factors, int64_t total,
                            int64_t edges, int64_t input_width, int64_t n,
                            bool from_lv, T epsilon, T saturation) {
    for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total; index += int64_t(blockDim.x) * gridDim.x) {
        const int64_t row = index / edges, edge = index % edges;
        if (active[row] != -1) continue;
        const int32_t variable = col[edge];
        T value = variable == n ? T(INFINITY) : (from_lv
            ? values[row * input_width + variable] - incoming[row * (edges + 1) + edge]
            : values[row * input_width + edge]);
        if (variable != n) {
            value = value > saturation ? saturation : value;
            value = value < -saturation ? -saturation : value;
        }
        T probability = T(1) / (T(1) + exp(value));
        probability = probability < epsilon ? epsilon : probability;
        probability = probability > T(1) - epsilon ? T(1) - epsilon : probability;
        factors[row * (edges + 1) + edge] = T(1) - T(2) * probability;
    }
}

template <typename T>
__global__ void spa_checks(const T* factors, const T* signs, const int32_t* ptr,
                           const int32_t* col, const int64_t* active, T* incoming,
                           int64_t edges, int64_t m, int64_t n, int64_t padded_degree,
                           bool padded, int width, T epsilon) {
    const int64_t check = int64_t(blockIdx.x) * (blockDim.x / width) + threadIdx.x / width;
    const int lane = threadIdx.x % width;
    const int64_t row = blockIdx.y;
    if (check >= m || active[row] != -1) return;
    const int64_t start = padded ? check * padded_degree : ptr[check];
    const int64_t length = padded ? padded_degree : ptr[check + 1] - start;
    const T dummy = T(1) - T(2) * epsilon;
    for (int64_t target = lane; target < length; target += width) {
        const int64_t edge = start + target;
        if (col[edge] == n) {
            incoming[row * (edges + 1) + edge] = T(0);
            continue;
        }
        T left = T(1), right = T(1);
        for (int64_t slot = 0; slot < target; ++slot)
            left *= factors[row * (edges + 1) + start + slot];
        for (int64_t slot = padded_degree - 1; slot > target; --slot)
            right *= slot < length ? factors[row * (edges + 1) + start + slot] : dummy;
        const T product = left * right;
        T probability = (T(1) - signs[row * m + check] * product) / T(2);
        probability = probability < epsilon ? epsilon : probability;
        probability = probability > T(1) - epsilon ? T(1) - epsilon : probability;
        incoming[row * (edges + 1) + edge] = log((T(1) - probability) / probability);
    }
}

void spa_cn(torch::Tensor values, torch::Tensor signs, torch::Tensor ptr,
            torch::Tensor col, torch::Tensor incoming, torch::Tensor active,
            int64_t n, int64_t padded_degree, bool warp, bool from_lv, double saturation) {
    TORCH_CHECK(values.is_cuda() && incoming.is_cuda() && signs.is_cuda(), "SPA tensors must be CUDA tensors");
    TORCH_CHECK(values.is_contiguous() && incoming.is_contiguous() && signs.is_contiguous(),
                "SPA tensors must be contiguous");
    TORCH_CHECK(values.scalar_type() == incoming.scalar_type() && signs.scalar_type() == incoming.scalar_type(),
                "SPA floating dtypes must match");
    TORCH_CHECK(col.scalar_type() == torch::kInt32 && ptr.scalar_type() == torch::kInt32 &&
                active.scalar_type() == torch::kInt64, "SPA index dtypes must match base BP");
    const int64_t b = incoming.size(0), edges = col.numel(), m = signs.size(1);
    if (!b || !edges) return;
    c10::cuda::CUDAGuard guard(values.device());
    auto factors = torch::empty_like(incoming);
    const int width = warp ? 32 : 1;
    const int blocks = int(std::min<int64_t>((b * edges + 255) / 256, 65535));
    const dim3 check_blocks((m + (128 / width) - 1) / (128 / width), b);
    AT_DISPATCH_FLOATING_TYPES(incoming.scalar_type(), "ots_spa_cn", [&] {
        const scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon();
        spa_factors<scalar_t><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            values.data_ptr<scalar_t>(), incoming.data_ptr<scalar_t>(), col.data_ptr<int32_t>(),
            active.data_ptr<int64_t>(), factors.data_ptr<scalar_t>(), b * edges,
            edges, values.size(1), n, from_lv, epsilon, scalar_t(saturation));
        spa_checks<scalar_t><<<check_blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            factors.data_ptr<scalar_t>(), signs.data_ptr<scalar_t>(), ptr.data_ptr<int32_t>(),
            col.data_ptr<int32_t>(), active.data_ptr<int64_t>(), incoming.data_ptr<scalar_t>(),
            edges, m, n, padded_degree, col.dim() == 2, width, epsilon);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("switch_priors", &switch_priors);
    m.def("spa_cn", &spa_cn);
}

