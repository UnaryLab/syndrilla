#pragma once

#include <torch/extension.h>

#define OSD_CHECK(x)                                                   \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");           \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

void osd0_fused_cuda(torch::Tensor H_packed, torch::Tensor synd,
                     torch::Tensor order, torch::Tensor e_out,
                     int64_t N, int64_t A_rank, int64_t block_size);

// Shared-memory the fused kernel needs (bytes) and the device opt-in limit.
int64_t fused_smem_bytes(int64_t M, int64_t W);
int64_t fused_smem_limit();

void osd_pivot_cuda(torch::Tensor aug, torch::Tensor order,
                    torch::Tensor row_pcol, torch::Tensor pivot_row,
                    int64_t step, int64_t N);

void osd_eliminate_cuda(torch::Tensor aug, torch::Tensor order,
                        torch::Tensor pivot_row, int64_t step, int64_t N);

void osd_solve_cuda(torch::Tensor aug, torch::Tensor row_pcol,
                    torch::Tensor e_out, int64_t N);
