#pragma once

#include <torch/extension.h>

#define OSD_CHECK(x)                                                   \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");           \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// Pack the first K columns of each sample's order into the workspace.
void osd_gather_cuda(torch::Tensor ws, torch::Tensor colptr, torch::Tensor rowidx,
                     torch::Tensor order, int64_t K);

// Gauss-Jordan over workspace columns 0..K-1, one block per sample in shared memory.
void osd0_fused_ws_cuda(torch::Tensor ws, torch::Tensor row_pcol, torch::Tensor found,
                        int64_t K, int64_t A_rank, int64_t block_size);

// Shared-memory the fused kernel needs (bytes) and the device opt-in limit.
int64_t fused_smem_bytes(int64_t M, int64_t W);
int64_t fused_smem_limit();

// Per-step path: load column j and find its pivot; eliminate column j.
void osd_load_col_cuda(torch::Tensor ws, torch::Tensor colw, torch::Tensor row_pcol,
                       torch::Tensor pivbuf, int64_t j);
void osd_step_cuda(torch::Tensor ws, torch::Tensor colw, torch::Tensor row_pcol,
                   torch::Tensor pivbuf, torch::Tensor found, int64_t j, int64_t K);

// Order positions of each sample's pivot columns; stops early once the packed
// syndrome (empty tensor: never) lies in the span of the pivots found. T is held
// in per-sample row pools Pr / Pc; a sample that runs out of slots sets overflow.
void osd_scan_cuda(torch::Tensor Pr, torch::Tensor Pc, torch::Tensor rslot,
                   torch::Tensor cslot, torch::Tensor colptr, torch::Tensor rowidx,
                   torch::Tensor order, torch::Tensor piv_pos, torch::Tensor found,
                   torch::Tensor synd, torch::Tensor stopped, torch::Tensor scan_end,
                   torch::Tensor overflow);

// Read out the OSD-0 estimate from the pivot rows.
void osd_solve_ws_cuda(torch::Tensor ws, torch::Tensor row_pos, torch::Tensor order,
                       torch::Tensor e_out, int64_t K);
