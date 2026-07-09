/*
 * union_find_kernel.cu -- standalone CUDA Union-Find decoder for the union_find_cuda
 * extension. Self-contained: depends on NOTHING in union_find.py.
 *
 * One file per decoder kernel, matching bp_kernel.cu / osd0_kernel.cu. The whole
 * algorithm (robin-hood table, lattice build, serial grow/fuse/peel) lives in the
 * portable header union_find_serial.cuh, a line-for-line transliteration of
 * union_find.py's _RobinTable / build_lattice_from_parity / decode_shot. Because that
 * decode is a reference-faithful serial fusion-forest decode, this kernel is bit-exact
 * to decode_shot: bit-for-bit identical vs the C++ chaeyeunpark reference on toric codes, and
 * valid (H c == s) on surface / open-boundary codes.
 *
 * Parallelism is across shots: one thread decodes one shot with its own global-memory
 * scratch. The serial decode is inherently sequential (robin-hood iteration order is
 * load-bearing), so this is correctness-first, not a within-shot-parallel kernel.
 *
 * Two Python entry points:
 *   uf_build_lattice(H)  -- host: build the tsl-order detector graph once, return the
 *                           CSR adjacency + (V, B, M, N). Cached by the Python module.
 *   uf_serial_exact(...) -- device: decode a [B, M] syndrome batch to [B, N] corrections.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>

#include "union_find_serial.cuh"

#define UF_CHECK(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

// ════════════════════════════════════════════════════════════════════════════
// Device kernel -- one thread per shot, full serial exact decode
// ════════════════════════════════════════════════════════════════════════════
__global__ void k_uf_serial_exact(
    const int* __restrict__ synd_all,    // [B, M]
    int*       __restrict__ corr_all,    // [B, N]
    int*       __restrict__ scratch_all, // [B, scratch_ints]
    const int* __restrict__ conn_off,    // [V+1]
    const int* __restrict__ conn_nbr,    // [E2]
    const int* __restrict__ conn_q,      // [E2]
    const int* __restrict__ vcc,         // [V]
    int V, int N, int M, int Bbd, int Bshots, long scratch_ints
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= Bshots) return;

    UFScratch s;
    uf_carve(&s, scratch_all + (long)b * scratch_ints, V, N);
    uf_decode_shot(V, N, M, Bbd, conn_off, conn_nbr, conn_q, vcc,
                   synd_all + (long)b * M, &s);

    int* corr = corr_all + (long)b * N;
    for (int q = 0; q < N; q++) corr[q] = s.corr[q];
}

// ════════════════════════════════════════════════════════════════════════════
// Host entry: build the detector graph (once per parity matrix)
// ════════════════════════════════════════════════════════════════════════════
//
// H is a [M, N] int32 CPU tensor (0/1). Returns the CSR adjacency in tsl order plus
// the scalar shape. The returned tensors are on CPU; the Python module moves them to
// the CUDA device and caches them.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           int64_t, int64_t, int64_t, int64_t>
uf_build_lattice_ext(torch::Tensor H) {
    TORCH_CHECK(H.dim() == 2, "H must be 2-D [M, N]");
    auto Hc = H.to(torch::kInt32).cpu().contiguous();
    const int M = (int)Hc.size(0), N = (int)Hc.size(1);
    const int* Hp = Hc.data_ptr<int32_t>();

    // qubit_parities: for each column, the (increasing) detector rows it hits.
    std::vector<int> qw(N, 0), qr0(N, -1), qr1(N, -1);
    for (int p = 0; p < M; p++) {
        const int* row = Hp + (long)p * N;
        for (int q = 0; q < N; q++) {
            if (row[q] == 0) continue;
            if (qw[q] == 0) qr0[q] = p;
            else if (qw[q] == 1) qr1[q] = p;
            else TORCH_CHECK(false, "column ", q, " has weight > 2; H is not graphlike");
            qw[q]++;
        }
    }

    int V = 0, B = 0;
    std::vector<int> conn_off, conn_nbr, conn_q, vcc;
    uf_build_lattice(M, N, qw.data(), qr0.data(), qr1.data(),
                     &V, &B, conn_off, conn_nbr, conn_q, vcc);

    auto opt = torch::dtype(torch::kInt32);
    auto to_t = [&](const std::vector<int>& v) {
        return torch::from_blob((void*)v.data(), {(long)v.size()}, opt).clone();
    };
    return std::make_tuple(to_t(conn_off), to_t(conn_nbr), to_t(conn_q), to_t(vcc),
                           (int64_t)V, (int64_t)B, (int64_t)M, (int64_t)N);
}

// ════════════════════════════════════════════════════════════════════════════
// Host entry: decode a syndrome batch (device)
// ════════════════════════════════════════════════════════════════════════════
torch::Tensor uf_serial_exact_cuda(
    torch::Tensor synd,      // [B, M] int32 (CUDA)
    torch::Tensor conn_off,  // [V+1]  int32 (CUDA)
    torch::Tensor conn_nbr,  // [E2]   int32 (CUDA)
    torch::Tensor conn_q,    // [E2]   int32 (CUDA)
    torch::Tensor vcc,       // [V]    int32 (CUDA)
    int64_t V, int64_t N, int64_t M, int64_t B, int64_t block_size
) {
    UF_CHECK(synd); UF_CHECK(conn_off); UF_CHECK(conn_nbr);
    UF_CHECK(conn_q); UF_CHECK(vcc);
    const int Bshots = (int)synd.size(0);
    auto corr = torch::empty({Bshots, (long)N},
                             torch::dtype(torch::kInt32).device(synd.device()));
    if (Bshots == 0) return corr;

    const long scratch_ints = uf_scratch_ints((int)V, (int)N);
    auto scratch = torch::empty({(long)Bshots, scratch_ints},
                                torch::dtype(torch::kInt32).device(synd.device()));

    const int blk = (int)block_size;
    const int grid = (Bshots + blk - 1) / blk;
    auto stream = at::cuda::getCurrentCUDAStream();
    k_uf_serial_exact<<<grid, blk, 0, stream>>>(
        synd.data_ptr<int32_t>(),
        corr.data_ptr<int32_t>(),
        scratch.data_ptr<int32_t>(),
        conn_off.data_ptr<int32_t>(),
        conn_nbr.data_ptr<int32_t>(),
        conn_q.data_ptr<int32_t>(),
        vcc.data_ptr<int32_t>(),
        (int)V, (int)N, (int)M, (int)B, Bshots, scratch_ints);
    return corr;
}

int64_t uf_scratch_ints_ext(int64_t V, int64_t N) {
    return uf_scratch_ints((int)V, (int)N);
}

// ════════════════════════════════════════════════════════════════════════════
// Python bindings
// ════════════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uf_build_lattice", &uf_build_lattice_ext,
          "Build the tsl-order detector graph from H; returns CSR adjacency + shape");
    m.def("uf_serial_exact", &uf_serial_exact_cuda,
          "Bit-exact serial Union-Find decode of a syndrome batch (CUDA)");
    m.def("uf_scratch_ints", &uf_scratch_ints_ext,
          "Per-shot scratch size (int count) for [V, N]");
}
