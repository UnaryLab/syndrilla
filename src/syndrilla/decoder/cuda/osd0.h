/*
 * osd0.h — host-side dispatch prototypes for the osd_0_cuda extension.
 *
 * osd0_kernel.cu defines these and binds them to Python. Keeping the prototypes
 * in this header lets gf2.cuh and osd0_kernel.cu share a single source of truth
 * for the host/device boundary.
 *
 * Packed-tensor conventions (all CUDA, contiguous, row-major):
 *   H_packed  : int64 [M, W]      bit-packed parity check matrix, shared across
 *                                 samples (reinterpreted as uint64 in kernels).
 *   synd      : uint8 [B, M]      per-sample syndrome bits (0/1).
 *   order     : int32 [B, N]      per-sample column processing order (by LLR).
 *   e_out     : uint8 [B, N]      per-sample OSD-0 error estimate (output).
 *   aug       : int64 [B, M, W]   per-sample augmented matrix [H | s] (per-step).
 *   row_pcol  : int32 [B, M]      pivot column of each row, -1 if none.
 *   pivot_row : int32 [B]         pivot row chosen at the current step, -1 none.
 */
#pragma once

#include <torch/extension.h>

#define OSD_CHECK(x)                                                   \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");           \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// ── Fused path: whole OSD-0 solve per sample in one launch ──────────────────
void osd0_fused_cuda(torch::Tensor H_packed, torch::Tensor synd,
                     torch::Tensor order, torch::Tensor e_out,
                     int64_t N, int64_t A_rank, int64_t block_size);

// Shared-memory the fused kernel needs (bytes) and the device opt-in limit.
int64_t fused_smem_bytes(int64_t M, int64_t W);
int64_t fused_smem_limit();

// ── Per-step path: modular kernels driven by a Python loop ──────────────────
void osd_pivot_cuda(torch::Tensor aug, torch::Tensor order,
                    torch::Tensor row_pcol, torch::Tensor pivot_row,
                    int64_t step, int64_t N);

void osd_eliminate_cuda(torch::Tensor aug, torch::Tensor order,
                        torch::Tensor pivot_row, int64_t step, int64_t N);

void osd_solve_cuda(torch::Tensor aug, torch::Tensor row_pcol,
                    torch::Tensor e_out, int64_t N);
