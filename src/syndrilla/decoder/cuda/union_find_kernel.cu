#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>

#include "union_find_serial.cuh"

#define UF_CHECK(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

__global__ void k_uf_serial_exact(
    const int* __restrict__ synd_all,    // [B, M]
    int*       __restrict__ corr_all,    // [B, N]
    int*       __restrict__ scratch_all, // [B, scratch_ints]
    const int* __restrict__ conn_off,    // [V+1]
    const int* __restrict__ conn_nbr,    // [E2]
    const int* __restrict__ conn_q,      // [E2]
    const int* __restrict__ vcc,         // [V]
    int V, int N, int M, int Bbd, int Bshots, int64_t scratch_ints
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= Bshots) return;

    UFScratch s;
    uf_carve(&s, scratch_all + (int64_t)b * scratch_ints, V, N);
    uf_decode_shot(V, N, M, Bbd, conn_off, conn_nbr, conn_q, vcc,
                   synd_all + (int64_t)b * M, &s);

    int* corr = corr_all + (int64_t)b * N;
    for (int q = 0; q < N; q++) corr[q] = s.corr[q];
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           int64_t, int64_t, int64_t, int64_t>
uf_build_lattice_ext(torch::Tensor idx, int64_t M_, int64_t N_) {
    TORCH_CHECK(idx.layout() == torch::kStrided, "idx must be a dense tensor, not sparse");
    TORCH_CHECK(idx.scalar_type() == torch::kInt64, "idx must be int64");
    TORCH_CHECK(idx.dim() == 2 && idx.size(0) == 2, "idx must be [2, nnz]");
    auto ic = idx.cpu().contiguous();
    const int M = (int)M_, N = (int)N_;
    const int64_t nnz = ic.size(1);
    const int64_t* rp = ic.data_ptr<int64_t>();
    const int64_t* cp = rp + nnz;

    // qubit_parities: for each column, the (increasing) detector rows it hits.
    // idx is coalesced COO (row-major), so each column's rows arrive in ascending order.
    std::vector<int> qw(N, 0), qr0(N, -1), qr1(N, -1);
    for (int64_t k = 0; k < nnz; k++) {
        TORCH_CHECK(rp[k] >= 0 && rp[k] < M && cp[k] >= 0 && cp[k] < N,
                    "idx entry ", k, " out of range for [", M, ", ", N, "]");
        const int p = (int)rp[k], q = (int)cp[k];
        if (qw[q] == 0) qr0[q] = p;
        else if (qw[q] == 1) qr1[q] = p;
        else TORCH_CHECK(false, "column ", q, " has weight > 2; H is not graphlike");
        qw[q]++;
    }

    int V = 0, B = 0;
    std::vector<int> conn_off, conn_nbr, conn_q, vcc;
    uf_build_lattice(M, N, qw.data(), qr0.data(), qr1.data(),
                     &V, &B, conn_off, conn_nbr, conn_q, vcc);

    auto opt = torch::dtype(torch::kInt32);
    auto to_t = [&](const std::vector<int>& v) {
        return torch::from_blob((void*)v.data(), {(int64_t)v.size()}, opt).clone();
    };
    return std::make_tuple(to_t(conn_off), to_t(conn_nbr), to_t(conn_q), to_t(vcc),
                           (int64_t)V, (int64_t)B, (int64_t)M, (int64_t)N);
}

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
    auto corr = torch::empty({Bshots, (int64_t)N},
                             torch::dtype(torch::kInt32).device(synd.device()));
    if (Bshots == 0) return corr;

    const int64_t scratch_ints = uf_scratch_ints((int)V, (int)N);
    auto scratch = torch::empty({(int64_t)Bshots, scratch_ints},
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uf_build_lattice", &uf_build_lattice_ext,
          "Build the tsl-order detector graph from H; returns CSR adjacency + shape");
    m.def("uf_serial_exact", &uf_serial_exact_cuda,
          "Bit-exact serial Union-Find decode of a syndrome batch (CUDA)");
    m.def("uf_scratch_ints", &uf_scratch_ints_ext,
          "Per-shot scratch size (int count) for [V, N]");
}
