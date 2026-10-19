// SymBreak study port: Yin et al., https://arxiv.org/abs/2412.02885,
// Algorithm 1 and Sec. IV-C, V. The torch module defines the unspecified choices.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

using State = std::vector<torch::Tensor>;
enum { COL, SYND, PRIOR, LV, MSG, RESET, VN, HARD, ACTIVE, SPLITS, ITER,
       CONV, DIFF, MONITOR, PREVIOUS, ORIGINAL, KEEP, SPLITTABLE, STATE_SIZE };

template <typename T>
__global__ void bp_step(
    const int64_t* col, const T* syndrome, const T* prior, T* lv, T* msg,
    bool* reset, const int64_t* incident, uint8_t* hard, bool* active,
    int64_t* iterations, int64_t* converges, int64_t* mismatch,
    const T* original, const int64_t* original_col,
    int64_t C, int64_t D, int64_t N, int64_t M, int64_t VD,
    T beta, T limit, int64_t iteration, bool early, bool gather, bool sign_parity,
    bool check) {
    int64_t b = blockIdx.x, E = C * D, N1 = N + 1;
    if (!active[b]) return;
    for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
        int64_t off = b * E + c * D;
        T min0 = INFINITY, min1 = INFINITY;
        int64_t arg = 0;
        bool parity = syndrome[b * C + c] != T(0);
        T sign_product = parity ? T(-1) : T(1);
        for (int64_t k = 0; k < D; ++k) {
            int64_t v = col[off + k];
            T a = reset[off + k] ? prior[b * N1 + v] : lv[b * N1 + v] - msg[off + k];
            parity ^= a <= T(0);
            sign_product *= a <= T(0) ? T(-1) : T(1);
            T mag = abs(a);
            if (mag < min0) { min1 = min0; min0 = mag; arg = k; }
            else if (mag < min1) min1 = mag;
        }
        for (int64_t k = 0; k < D; ++k) {
            int64_t v = col[off + k];
            T a = reset[off + k] ? prior[b * N1 + v] : lv[b * N1 + v] - msg[off + k];
            T value = beta * (k == arg ? min1 : min0);
            if (sign_parity) {
                if ((a <= T(0)) ^ parity) value = -value;
            } else {
                value = (beta * (a <= T(0) ? T(-1) : T(1)) * sign_product) * (k == arg ? min1 : min0);
            }
            msg[off + k] = v == N ? T(0) : max(-limit, min(limit, value));
            reset[off + k] = false;
        }
    }
    __syncthreads();
    if (gather) {
        for (int64_t v = threadIdx.x; v <= N; v += blockDim.x) {
            T sum = 0;
            for (int64_t j = 0; j < VD; ++j) {
                int64_t edge = incident[(b * N1 + v) * VD + j];
                if (edge >= 0) sum = sum + msg[b * E + edge];
            }
            lv[b * N1 + v] = sum + prior[b * N1 + v];
        }
    } else {
        // Scatter in edge order, without atomics, to keep the reference sum order.
        if (threadIdx.x == 0) {
            for (int64_t v = 0; v <= N; ++v) lv[b * N1 + v] = T(0);
            for (int64_t edge = 0; edge < E; ++edge) {
                int64_t v = col[b * E + edge];
                lv[b * N1 + v] = lv[b * N1 + v] + msg[b * E + edge];
            }
        }
        __syncthreads();
        for (int64_t v = threadIdx.x; v <= N; v += blockDim.x)
            lv[b * N1 + v] = lv[b * N1 + v] + prior[b * N1 + v];
    }
    __syncthreads();
    for (int64_t v = threadIdx.x; v <= N; v += blockDim.x)
        hard[b * N1 + v] = lv[b * N1 + v] <= T(0);
    __syncthreads();
    if (threadIdx.x == 0) {
        iterations[b] = iteration;
        if (!check) return;
        int64_t d = 0;
        for (int64_t c = 0; c < M; ++c) {
            int parity = 0;
            for (int64_t k = 0; k < D; ++k) parity ^= hard[b * N1 + original_col[c * D + k]];
            d += T(parity) != original[b * M + c];
        }
        mismatch[b] = d;
        converges[b] = d == 0;
        if (early && d == 0) active[b] = false;
    }
}

__device__ bool overlaps(int64_t v, const int64_t* g, int64_t GD, int64_t N) {
    if (v == N) return false;
    for (int64_t j = 0; j < GD; ++j) if (v == g[j]) return true;
    return false;
}

template <typename T>
__device__ bool zero_neighbours(int64_t v, const int64_t* col, const T* syndrome,
                               int64_t C, int64_t D, int64_t parent) {
    for (int64_t c = 0; c < C; ++c) {
        if (c == parent || syndrome[c] == T(0)) continue;
        for (int64_t j = 0; j < D; ++j) if (col[c * D + j] == v) return false;
    }
    return true;
}

__device__ bool partition_possible(int64_t chosen, int64_t remaining,
                                   int64_t available_overlap, int64_t available_other,
                                   int64_t k) {
    for (int option = 0; option < (k % 2 ? 1 : 2); ++option) {
        int64_t target = k % 2 ? k : k - 1 + 2 * option;
        int64_t need = target - chosen;
        if (need >= 0 && need <= remaining && need <= available_overlap &&
            remaining - need <= available_other) return true;
    }
    return false;
}

template <typename T>
__global__ void monitor_split(
    int64_t* col, T* syndrome, const T* lv, T* msg, bool* reset, int64_t* incident,
    const uint8_t* hard, bool* active, int64_t* splits, const int64_t* mismatch,
    int64_t* monitor, int64_t* previous, bool* keep, bool* splittable,
    const int64_t* opposite, int64_t C, int64_t D, int64_t N, int64_t M,
    int64_t VD, int64_t G, int64_t GD, int64_t B, int64_t kmax,
    int64_t max_splits, bool guided, bool early) {
    int64_t b = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= B || !active[b]) return;
    int64_t d = mismatch[b], E = C * D, N1 = N + 1;
    if (previous[b] >= 0) {
        if (d >= previous[b]) ++monitor[b];
        else if (monitor[b] > 0) --monitor[b];
    }
    previous[b] = d;
    if (early && monitor[b] >= kmax) { active[b] = false; return; }
    if (d == 0 || splits[b] >= max_splits || M + splits[b] >= C) return;
    int64_t gbest = -1;
    T best = INFINITY;
    for (int64_t g = 0; g < G; ++g) {
        T sum = 0;
        int64_t degree = 0;
        for (int64_t j = 0; j < GD; ++j) {
            int64_t v = opposite[g * GD + j];
            if (v != N) { sum = sum + abs(lv[b * N1 + v]); ++degree; }
        }
        if (degree && sum / T(degree) < best) { best = sum / T(degree); gbest = g; }
    }
    if (gbest < 0) return;
    const int64_t* g = opposite + gbest * GD;
    int64_t cbest = -1, chosen_overlap = 0, chosen_degree = 0;
    best = INFINITY;
    for (int64_t c = 0; c < C; ++c) {
        if (!splittable[b * C + c]) continue;
        T sum = 0;
        int64_t degree = 0, overlap = 0;
        for (int64_t j = 0; j < D; ++j) {
            int64_t v = col[b * E + c * D + j];
            if (v != N) { sum = sum + abs(lv[b * N1 + v]); ++degree; }
            overlap += overlaps(v, g, GD, N);
        }
        if (degree > 1 && overlap > 0 && overlap % 2 == 0 && sum / T(degree) < best) {
            best = sum / T(degree); cbest = c;
            chosen_overlap = overlap; chosen_degree = degree;
        }
    }
    if (cbest < 0) return;
    int64_t base = b * E + cbest * D;
    int64_t k = chosen_overlap / 2, count_first = k % 2 ? k : k - 1;
    int64_t count_other = max(int64_t(0), chosen_degree / 2 - count_first);
    int s1 = 0;
    if (guided) {
        int64_t allowed_overlap = 0, allowed_other = 0;
        for (int64_t j = 0; j < D; ++j) {
            int64_t v = col[base + j];
            if (v == N || !zero_neighbours(v, col + b * E, syndrome + b * C, C, D, cbest)) continue;
            if (overlaps(v, g, GD, N)) ++allowed_overlap;
            else ++allowed_other;
        }
        int64_t half = chosen_degree / 2;
        if (!partition_possible(0, half, allowed_overlap, allowed_other, k)) return;
        for (int64_t j = 0; j < D; ++j) keep[b * D + j] = false;
        int64_t last = -1, selected_overlap = 0;
        // Greedy feasibility gives the first valid qubit-id combination without a degree limit.
        for (int64_t selected = 0; selected < half; ++selected) {
            int64_t next = N, next_slot = -1;
            for (int64_t j = 0; j < D; ++j) {
                int64_t v = col[base + j];
                if (v <= last || v >= next ||
                    !zero_neighbours(v, col + b * E, syndrome + b * C, C, D, cbest)) continue;
                int64_t rest_overlap = 0, rest_other = 0;
                for (int64_t l = 0; l < D; ++l) {
                    int64_t q = col[base + l];
                    if (q <= v || q == N ||
                        !zero_neighbours(q, col + b * E, syndrome + b * C, C, D, cbest)) continue;
                    if (overlaps(q, g, GD, N)) ++rest_overlap;
                    else ++rest_other;
                }
                if (partition_possible(selected_overlap + overlaps(v, g, GD, N),
                                       half - selected - 1, rest_overlap, rest_other, k)) {
                    next = v; next_slot = j;
                }
            }
            keep[b * D + next_slot] = true;
            selected_overlap += overlaps(next, g, GD, N);
            last = next;
        }
    } else {
        for (int64_t j = 0; j < D; ++j) {
            int64_t v = col[base + j], rank = 0;
            bool overlap = overlaps(v, g, GD, N);
            T reliability = abs(lv[b * N1 + v]);
            for (int64_t l = 0; l < D; ++l) {
                int64_t q = col[base + l];
                if (q == N || overlaps(q, g, GD, N) != overlap) continue;
                T r = abs(lv[b * N1 + q]);
                rank += r > reliability || (r == reliability && q < v);
            }
            bool retain = v != N && rank < (overlap ? count_first : count_other);
            keep[b * D + j] = retain;
            if (retain) s1 ^= hard[b * N1 + v];
        }
    }
    int64_t new_c = M + splits[b];
    syndrome[b * C + new_c] = int(syndrome[b * C + cbest]) ^ s1;
    syndrome[b * C + cbest] = s1;
    splittable[b * C + cbest] = false;
    splittable[b * C + new_c] = false;
    for (int64_t j = 0; j < D; ++j) {
        int64_t v = col[base + j];
        if (v == N || keep[b * D + j]) continue;
        int64_t old_edge = cbest * D + j, new_edge = new_c * D + j;
        col[b * E + new_edge] = v;
        col[base + j] = N;
        msg[base + j] = T(0);
        reset[b * E + new_edge] = true;
        int64_t* edges = incident + (b * N1 + v) * VD;
        for (int64_t l = 0; l < VD; ++l) if (edges[l] == old_edge) {
            edges[l] = new_edge;
            // New rows append after all old rows, so restore increasing edge order.
            for (int64_t p = l; p + 1 < VD && edges[p + 1] >= 0 && edges[p] > edges[p + 1]; ++p) {
                int64_t swap = edges[p]; edges[p] = edges[p + 1]; edges[p + 1] = swap;
            }
            break;
        }
    }
    ++splits[b];
}

void check_state(const State& s) {
    TORCH_CHECK(s.size() == STATE_SIZE, "invalid SymBreak state");
    for (const auto& tensor : s) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), "SymBreak state must be contiguous CUDA tensors");
        TORCH_CHECK(tensor.device() == s[LV].device(), "SymBreak state must share a device");
    }
    for (int index : {COL, VN, SPLITS, ITER, CONV, DIFF, MONITOR, PREVIOUS})
        TORCH_CHECK(s[index].scalar_type() == at::kLong, "SymBreak indices and counters must be int64");
    for (int index : {SYND, PRIOR, LV, MSG, ORIGINAL})
        TORCH_CHECK(s[index].scalar_type() == s[LV].scalar_type(), "SymBreak values must share a dtype");
    for (int index : {RESET, ACTIVE, KEEP, SPLITTABLE})
        TORCH_CHECK(s[index].scalar_type() == at::kBool, "SymBreak masks must be bool");
    TORCH_CHECK(s[HARD].scalar_type() == at::kByte, "SymBreak hard decisions must be uint8");
}

void step(State s, torch::Tensor original_col, double beta, double limit, int64_t iteration,
          bool early, bool gather, bool sign_parity, bool check) {
    check_state(s);
    TORCH_CHECK(original_col.is_cuda() && original_col.is_contiguous() && original_col.scalar_type() == at::kLong,
                "original columns must be contiguous CUDA int64");
    int64_t B = s[COL].size(0), C = s[COL].size(1), D = s[COL].size(2);
    if (!B) return;
    c10::cuda::CUDAGuard guard(s[LV].device());
    AT_DISPATCH_FLOATING_TYPES(s[LV].scalar_type(), "symbreak_step", [&] {
        bp_step<scalar_t><<<B, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            s[COL].data_ptr<int64_t>(), s[SYND].data_ptr<scalar_t>(), s[PRIOR].data_ptr<scalar_t>(),
            s[LV].data_ptr<scalar_t>(), s[MSG].data_ptr<scalar_t>(), s[RESET].data_ptr<bool>(),
            s[VN].data_ptr<int64_t>(), s[HARD].data_ptr<uint8_t>(), s[ACTIVE].data_ptr<bool>(),
            s[ITER].data_ptr<int64_t>(), s[CONV].data_ptr<int64_t>(), s[DIFF].data_ptr<int64_t>(),
            s[ORIGINAL].data_ptr<scalar_t>(), original_col.data_ptr<int64_t>(),
            C, D, s[LV].size(1) - 1, s[ORIGINAL].size(1), s[VN].size(2),
            scalar_t(beta), scalar_t(limit), iteration, early, gather, sign_parity, check);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void split(State s, torch::Tensor opposite, int64_t kmax, int64_t max_splits, bool guided, bool early) {
    check_state(s);
    TORCH_CHECK(opposite.is_cuda() && opposite.is_contiguous() && opposite.scalar_type() == at::kLong,
                "opposite columns must be contiguous CUDA int64");
    int64_t B = s[COL].size(0);
    if (!B) return;
    c10::cuda::CUDAGuard guard(s[LV].device());
    AT_DISPATCH_FLOATING_TYPES(s[LV].scalar_type(), "symbreak_split", [&] {
        monitor_split<scalar_t><<<(B + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            s[COL].data_ptr<int64_t>(), s[SYND].data_ptr<scalar_t>(), s[LV].data_ptr<scalar_t>(),
            s[MSG].data_ptr<scalar_t>(), s[RESET].data_ptr<bool>(), s[VN].data_ptr<int64_t>(),
            s[HARD].data_ptr<uint8_t>(), s[ACTIVE].data_ptr<bool>(), s[SPLITS].data_ptr<int64_t>(),
            s[DIFF].data_ptr<int64_t>(), s[MONITOR].data_ptr<int64_t>(), s[PREVIOUS].data_ptr<int64_t>(),
            s[KEEP].data_ptr<bool>(), s[SPLITTABLE].data_ptr<bool>(), opposite.data_ptr<int64_t>(),
            s[COL].size(1), s[COL].size(2),
            s[LV].size(1) - 1, s[ORIGINAL].size(1), s[VN].size(2),
            opposite.size(0), opposite.size(1), B, kmax, max_splits, guided, early);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step", &step);
    m.def("split", &split);
}
