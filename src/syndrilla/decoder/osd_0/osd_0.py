import torch
from loguru import logger


class COOMatrixGF2Batch:
    def __init__(self, H: torch.Tensor, B: int):
        """
        H: Dense GF(2) matrix of shape [M, N], dtype=torch.uint8 or torch.bool
        B: Number of batches
        """
        assert H.dim() == 2, 'H must be a 2D matrix'
        assert H.dtype in [torch.bool, torch.uint8], 'H must be of dtype bool or uint8'

        M, N = H.shape
        self.shape = (B, M, N)
        self.B = B

        # Find non-zero positions
        nz = H.nonzero(as_tuple=False)  # [nnz, 2], each row is [row, col]
        self.nnz = nz.size(0)

        # Repeat for all batches
        self.batch_idx = torch.arange(B).repeat_interleave(self.nnz)
        self.row_idx = nz[:, 0].repeat(B)
        self.col_idx = nz[:, 1].repeat(B)
        self.values = torch.ones(B * self.nnz, dtype=torch.uint8)


    def from_sparse_batch(cls, sparse_batch: list[list[list[int]]], n_row: int, n_col: int):
        """
        Initialize from a list-of-list-of-lists sparse format:
        sparse_batch[b][r] = list of col indices where entry is 1
        """
        batch_idx = []
        row_idx = []
        col_idx = []

        for b, batch in enumerate(sparse_batch):
            for r, row in enumerate(batch):
                for c in row:
                    batch_idx.append(b)
                    row_idx.append(r)
                    col_idx.append(c)

        batch_idx = torch.tensor(batch_idx, dtype=torch.int32)
        row_idx = torch.tensor(row_idx, dtype=torch.int16)
        col_idx = torch.tensor(col_idx, dtype=torch.int16)
        values = torch.ones_like(batch_idx, dtype=torch.bool)

        obj = cls.__new__(cls)
        obj.shape = (len(sparse_batch), n_row, n_col)
        obj.B = len(sparse_batch)
        obj.batch_idx = batch_idx
        obj.row_idx = row_idx
        obj.col_idx = col_idx
        obj.values = values
        obj.nnz = len(values)
        return obj


    def create_empty_batch(B: int, n_row: int, n_col: int):
        """
        Returns an empty COO sparse matrix batch.
        """
        obj = COOMatrixGF2Batch.__new__(COOMatrixGF2Batch)
        obj.shape = (B, n_row, n_col)
        obj.B = B
        obj.batch_idx = torch.empty(0, dtype=torch.int32)
        obj.row_idx = torch.empty(0, dtype=torch.int16)
        obj.col_idx = torch.empty(0, dtype=torch.int16)
        obj.values = torch.empty(0, dtype=torch.bool)
        obj.nnz = 0
        return obj


    def to_dense(self, device=None):
        """
        Convert to dense [B, M, N] torch bool tensor.
        """
        B, M, N = self.shape
        dense = torch.zeros((B, M, N), dtype=torch.bool, device=self.batch_idx.device)
        dense[self.batch_idx, self.row_idx.to(torch.int64), self.col_idx.to(torch.int64)] = self.values.to(torch.bool)
        return dense


class create(torch.nn.Module):
    def __init__(self,
                    decoder_cfg,
                    **kwargs) -> None:

        super(create, self).__init__()
        logger.info('Creating osd-0 decoder.')

        # set up default device
        device_cfg = decoder_cfg.get('device', {})
        self.device = device_cfg.get('device_type', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device not in {'cuda', 'cpu', torch.device('cuda'), torch.device('cpu')}:
            logger.warning(f'Invalid input device <{self.device}>, default to avaliable device in your machine.')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device == 'cuda':
            device_idx = device_cfg.get('device_idx', 0)
            if device_idx >= torch.cuda.device_count():
                logger.warning(f'Invalid input device index <{device_idx}>, default to avaliable device in your machine.')
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device(f'cuda:{device_idx}')

        # set up default dtype
        self.dtype = decoder_cfg.get('dtype', 'float64')
        if self.dtype not in {'float32', 'float64', 'bfloat16', 'float16'}:
            logger.warning(f'Invalid input data type <{self.dtype}>, default to <torch.float64>.')
            self.dtype = 'float64'
        self.dtype = torch.__dict__[self.dtype]
        self.algo = 'osd_0'

        bundle = kwargs.get('bundle')
        if bundle is None:
            raise ValueError('osd_0 requires a pre-loaded MatrixBundle via the `bundle` kwarg.')
        check_type = decoder_cfg.get('check_type', 'hx')
        H_shape = bundle.select(check_type)[0]
        self.num_max_iter = H_shape[1]

        logger.info('Complete.')


    def forward(self, io_dict):
        logger.info('Initializing osd-0 decoding.')

        device = io_dict['synd'].device
        not_converged_mask = (io_dict['converge'] == 0)
        idx = not_converged_mask.nonzero(as_tuple=True)[0].to(device)
        soft_decision_sub = io_dict['llr'][idx]
        synd_sub = io_dict['synd'][idx]

        logger.info('Complete.')

        logger.info('Starting decoding.')

        _, cols_batch = torch.sort(soft_decision_sub, descending=False, stable=True)

        H = io_dict['H_matrix'].to(dtype=torch.bool)

        L_batch, U_batch, rows_lu_batch, cols_lu_batch, max_iter = self.LU_decomposition_batch(H, cols_batch)

        del H, cols_batch
        torch.cuda.empty_cache()

        OSD_sub_result = self.LU_forward_backward_solve_batch(L_batch, U_batch, rows_lu_batch, cols_lu_batch, synd_sub)
        del L_batch, U_batch, rows_lu_batch, cols_lu_batch, synd_sub
        torch.cuda.empty_cache()

        final_result = io_dict['e_v'].clone().to(dtype=torch.uint8, device=device)
        final_result[idx] = OSD_sub_result.to(device)
        del OSD_sub_result
        torch.cuda.empty_cache()

        logger.info('Complete.')

        ones_converge = torch.ones_like(io_dict['converge'])
        io_dict.update({
            'e_v': final_result,
            'iter': max_iter,
            'converge': ones_converge
        })
        return io_dict


    def LU_decomposition_batch(self, H, cols_batch):
        """
        Perform LU decomposition on GF(2) for a batch of column orderings.

        Args:
            H (torch.Tensor): [M, N] binary matrix (same for all batches)
            cols_batch (torch.Tensor): [B, N]

        Returns:
            L_batch: [B, M, A_rank]
            U_batch: [B, A_rank, N]
            rows_batch: [B, M]
            cols_batch_out: [B, N]
        """
        B, N = cols_batch.shape
        M = H.shape[0]

        device = H.device
        A_rank= self.gf2_rank(H)
        B_batch_sparse = COOMatrixGF2Batch(H, B)
        del H
        torch.cuda.empty_cache()

        L_batch_sparse = COOMatrixGF2Batch.create_empty_batch(B, M, A_rank)
        U_batch_sparse = COOMatrixGF2Batch.create_empty_batch(B, A_rank, N)

        arange_M = torch.arange(M, device=device, dtype=torch.int32)
        arange_N = torch.arange(N, device=device, dtype=torch.int32)
        arange_B = torch.arange(B, device=device, dtype=torch.int32)

        rows_batch = arange_M.unsqueeze(0).expand(B, M).clone()  # [B, M]
        rinv_batch = arange_M.unsqueeze(0).expand(B, M).clone()  # [B, M]

        cinv_batch = torch.zeros((B, N), dtype=torch.int32, device=device)
        cols_batch = cols_batch.to(torch.int32)
        cinv_batch[arange_B.unsqueeze(1), cols_batch] = arange_N.unsqueeze(0)

        cols_batch_out = cols_batch

        # cinv construct
        batch_indices = torch.arange(B, device=device, dtype=torch.int32).unsqueeze(1).expand(B, N)
        cinv_batch[batch_indices, cols_batch] = torch.arange(N, device=device, dtype=torch.int32).unsqueeze(0).expand(B, N)

        B_idx_full = torch.arange(B, device=device, dtype=torch.int32)            # shape [B]
        M_idx_full = torch.arange(M, device=device, dtype=torch.int32)            # shape [M]
        N_idx_full = torch.arange(N, device=device, dtype=torch.int32)            # shape [N]

        batch_indices_base = torch.arange(B, device=device, dtype=torch.int32).view(B, 1)
        row_indices = M_idx_full.view(1, 1, -1)
        row_indices = row_indices.to(dtype=torch.int16)
        B_idx = B_idx_full.view(B, 1).expand(B, M)            # [B, M]
        M_idx = M_idx_full.view(1, M).expand(B, M)          # [B, M]
        used_col_count = torch.zeros(B, dtype=torch.int16, device=device)

        for i in range(A_rank):
            # Step 1: For each batch, find the first valid pivot starting from row i,
            # where the pivot condition is B[b, :, cols[b, k]] == 1 and rinv >= i

            # ----------------------------------------------------------------------------
            # curr_cols: [B, N-i]
            curr_cols = cols_batch_out[:, i:N]  # columns from i to N-1
            B, ni = curr_cols.shape

            query_b = batch_indices_base.expand(B, ni).reshape(-1).to(device)   # [B * (N-i)]
            query_c = curr_cols.reshape(-1).to(device)                          # [B * (N-i)]
            # [Q, NNZ] = [B*(N-i), B*nnz]
            coo_b = B_batch_sparse.batch_idx.to(device)
            coo_r = B_batch_sparse.row_idx.to(device)
            coo_c = B_batch_sparse.col_idx.to(device)
            coo_v = B_batch_sparse.values.to(device)

            # key: (b * N + c)
            query_j = torch.arange(ni, device=device).repeat(B).to(device)        # [Q]
            query_key = query_b * N + query_c                         # [Q]
            coo_key = coo_b * N + coo_c                               # [NNZ]

            coo_key_sorted, sort_idx = torch.sort(coo_key)            # [NNZ]
            coo_r_sorted = coo_r[sort_idx]

            lower = torch.searchsorted(coo_key_sorted, query_key, side='left')   # [Q]
            upper = torch.searchsorted(coo_key_sorted, query_key, side='right')  # [Q]

            repeats = upper - lower
            mask = repeats > 0
            query_idx = torch.arange(query_key.size(0), device=device)

            query_idx_valid = query_idx[mask]
            repeats_valid = repeats[mask]
            lower_valid = lower[mask]

            q_idx_expanded = query_idx_valid.repeat_interleave(repeats_valid)         # [M]
            total_matches = repeats_valid.sum()
            offsets = lower_valid.repeat_interleave(repeats_valid)  # [M]

            range_increments = torch.arange(total_matches, device=device) - torch.cumsum(
                torch.nn.functional.pad(repeats_valid, (1, 0))[:-1], dim=0).repeat_interleave(repeats_valid)
            coo_idx = offsets + range_increments  # [M]

            r_idx = coo_r_sorted[coo_idx]
            b_idx = query_b[q_idx_expanded]
            j_idx = query_j[q_idx_expanded]

            del query_j, query_key, coo_key
            del curr_cols, query_b, query_c
            del query_idx_valid, repeats_valid, lower_valid
            del q_idx_expanded, total_matches, offsets, range_increments, coo_idx
            del lower, upper, repeats, mask, query_idx
            del coo_key_sorted, sort_idx, coo_r_sorted
            torch.cuda.empty_cache()

            candidate_mask = torch.zeros((B, ni, rinv_batch.shape[1]), dtype=torch.bool, device=device)
            candidate_mask.index_put_((b_idx, j_idx, r_idx), torch.ones_like(r_idx, dtype=torch.bool), accumulate=True)

            rinv_expand = rinv_batch.unsqueeze(1).expand(B, ni, rinv_batch.shape[1])  # [B, ni, N]
            candidate_mask &= (rinv_expand >= i)

            del rinv_expand
            del b_idx, j_idx, r_idx
            torch.cuda.empty_cache()

            valid_rows = torch.where(candidate_mask, row_indices, M)  # [B, N-i, M]

            del candidate_mask
            torch.cuda.empty_cache()
            # For each column, find the first valid row with a pivot
            min_rows, _ = valid_rows.min(dim=2)  # [B, N-i]

            # Determine which column gives the earliest valid row (i.e., first pivot)
            row_found_mask = min_rows < M
            valid_col_indices = N_idx_full[i:].view(1, -1).expand(B, N - i)  # [B, N-i]
            valid_col_vals = torch.where(row_found_mask, valid_col_indices, N)  # [B, N-i]
            min_col_vals, min_col_idx = valid_col_vals.min(dim=1)  # [B]
            min_row_vals = min_rows[B_idx_full, min_col_idx]  # [B]

            used_col_count = torch.maximum(used_col_count, min_col_vals + 1)
            # Write results
            pivot_row = torch.full((B,), -1, dtype=torch.long, device=device)
            pivot_col = torch.full((B,), -1, dtype=torch.long, device=device)

            pivot_col = cols_batch_out[B_idx_full, min_col_vals]  # [B]
            pivot_row = min_row_vals  # [B]
            # Step 2: swap cols[i] <-> pivot_col
            old_col_i = cols_batch_out[:, i].clone()

            pivot_cinv = cinv_batch[B_idx_full, pivot_col]

            # swap cols
            cols_batch_out[:, i] = pivot_col
            cols_batch_out[B_idx_full, pivot_cinv] = old_col_i

            # update cinv
            cinv_batch[B_idx_full, old_col_i] = pivot_cinv
            cinv_batch[B_idx_full, pivot_col] = i

            # Step 3: swap rows
            row_i = rows_batch[:, i].clone()

            pivot_row = pivot_row.to(dtype=torch.long)

            pivot_ri = rinv_batch[B_idx_full, pivot_row]

            # swap rows
            rows_batch[:, i] = pivot_row
            rows_batch[B_idx_full, pivot_ri] = row_i

            # update rinv
            rinv_batch[B_idx_full, row_i] = pivot_ri
            rinv_batch[B_idx_full, pivot_row] = i

            # B_idx: [B, M], M_idx: [B, M], pivot_col: [B]
            pivot_col_expand = pivot_col.view(B, 1).expand(B, M)       # [B, M]
            B_idx_flat = B_idx.reshape(-1)  # [B*M]
            M_idx_flat = M_idx.reshape(-1)  # [B*M]
            pivot_col_flat = pivot_col_expand.reshape(-1)  # [B*M]

            query_tuples = torch.stack([
                B_idx_flat,
                M_idx_flat,
                pivot_col_flat
            ], dim=1)

            del B_idx_flat, M_idx_flat, pivot_col_flat
            del old_col_i, pivot_cinv, row_i, pivot_ri
            del valid_rows, min_rows, row_found_mask, valid_col_indices, valid_col_vals
            del min_col_vals, min_col_idx, min_row_vals
            torch.cuda.empty_cache()

            coo_tuples = torch.stack([
            coo_b,
            coo_r,
            coo_c
            ], dim=1)
            coo_keys = self.tuple_hash(coo_tuples)        # [nnz_total]
            query_keys = self.tuple_hash(query_tuples)    # [B*M]

            row_val_flat = torch.isin(query_keys, coo_keys).to(torch.bool)  # [B*M]
            row_val = row_val_flat.view(B, M)  # [B, M]

            rinv_r = rinv_batch[B_idx, M_idx]                     # [B, M]
            gt_mask = (rinv_r > i) & (row_val == 1)
            eq_mask = (rinv_r == i) & (row_val == 1)
            lt_mask = (rinv_r < i) & (row_val == 1)

            # XOR elimination: B[r] ^= B[pivot_row]
            pivot_rows = pivot_row.view(B, 1, 1).expand(B, 1, N)   # [B, 1, N]
            pivot_B = torch.zeros(B, N, dtype=torch.uint8, device=device)

            target_r = pivot_row[coo_b]  # shape: [nnz]
            mask = coo_r == target_r

            pb = coo_b[mask]
            pc = coo_c[mask]
            pv = coo_v[mask]

            pivot_B[pb, pc] = pv
            pivot_B = pivot_B.unsqueeze(1)

            del pivot_rows, target_r, pb, pc, pv, coo_keys, query_keys, row_val_flat
            del query_tuples, coo_tuples
            torch.cuda.empty_cache()

            coo_b, coo_r, coo_c, coo_v = self.apply_xor_sparse(
                coo_b, coo_r, coo_c, coo_v,
                gt_mask=gt_mask,
                pivot_row=pivot_row,
                pivot_row_tensor=pivot_B.squeeze(1)  # shape [B, N]
            )
            B_batch_sparse.batch_idx = coo_b
            B_batch_sparse.row_idx = coo_r
            B_batch_sparse.col_idx = coo_c
            B_batch_sparse.values = coo_v

            # find index
            eq_b, eq_m = torch.nonzero(eq_mask, as_tuple=True)
            gt_b, gt_m = torch.nonzero(gt_mask, as_tuple=True)
            lt_b, lt_m = torch.nonzero(lt_mask, as_tuple=True)
            # --- Update L_sparse ---
            L_b = torch.cat([eq_b, gt_b]).to(device)
            L_r = torch.cat([eq_m, gt_m]).to(device)
            L_c = torch.full_like(L_r, i, dtype=torch.int32, device = device)
            L_v = torch.ones_like(L_r, dtype=torch.uint8, device = device)

            L_batch_sparse.batch_idx = torch.cat([L_batch_sparse.batch_idx.to(device), L_b])
            L_batch_sparse.row_idx   = torch.cat([L_batch_sparse.row_idx.to(device), L_r])
            L_batch_sparse.col_idx   = torch.cat([L_batch_sparse.col_idx.to(device), L_c])
            L_batch_sparse.values    = torch.cat([L_batch_sparse.values.to(device), L_v])

            # --- Update U_sparse ---
            # term1: U[lt_b, rinv_r[lt_b, lt_m], pivot_col_expand[lt_b, lt_m]]
            U_b1 = lt_b.to(device)
            U_r1 = rinv_r[lt_b, lt_m].to(device)
            U_c1 = pivot_col_expand[lt_b, lt_m].to(device)

            # term2: U[eq_b, i, pivot_col_expand[eq_b, eq_m]]
            U_b2 = eq_b.to(device)
            U_r2 = torch.full_like(eq_b, i, dtype=torch.int32, device = device)
            U_c2 = pivot_col_expand[eq_b, eq_m].to(device)

            U_b = torch.cat([U_b1, U_b2]).to(device)
            U_r = torch.cat([U_r1, U_r2]).to(device)
            U_c = torch.cat([U_c1, U_c2]).to(device)
            U_v = torch.ones_like(U_b, dtype=torch.uint8, device = device)

            U_batch_sparse.batch_idx = torch.cat([U_batch_sparse.batch_idx.to(device), U_b])
            U_batch_sparse.row_idx   = torch.cat([U_batch_sparse.row_idx.to(device), U_r])
            U_batch_sparse.col_idx   = torch.cat([U_batch_sparse.col_idx.to(device), U_c])
            U_batch_sparse.values    = torch.cat([U_batch_sparse.values.to(device), U_v])

            # ----------------------------------------------------------------
            del pivot_row, pivot_col, pivot_col_expand
            del coo_b, coo_r, coo_c, coo_v
            del rinv_r, gt_mask, eq_mask, lt_mask
            del eq_b, eq_m, gt_b, gt_m, lt_b, lt_m
            del L_b, L_r, L_c, L_v
            del U_b1, U_r1, U_c1, U_b2, U_r2, U_c2
            del U_b, U_r, U_c, U_v

            # torch.cuda.empty_cache()

        L_batch_dense = L_batch_sparse.to_dense()
        U_batch_dense = U_batch_sparse.to_dense()
        del B_batch_sparse, L_batch_sparse, U_batch_sparse, cols_batch, B_idx_full, M_idx_full, N_idx_full, batch_indices, batch_indices_base, B_idx, M_idx, row_indices
        torch.cuda.empty_cache()

        return L_batch_dense, U_batch_dense, rows_batch, cols_batch_out, used_col_count

    def mod2sparse_forward_sub_batch(self, L_batch, rows_batch, x_batch):
        """
        Batch version of mod2 forward substitution solving L y = x over GF(2)

        Args:
            L_batch: [B, M, K], lower-triangular binary matrix
            rows_batch: [B, K], row permutation
            x_batch: [B, M], right-hand side

        Returns:
            y_batch: [B, K]
        """
        device = L_batch.device
        B, M, K = L_batch.shape
        y_batch = torch.zeros((B, K), dtype=torch.uint8, device=device)
        x_batch = x_batch.to(device)

        for i in range(K):
            ii = rows_batch[:, i]  # [B]
            ii = ii.to(device)
            # Gather L[ii, :] → shape: [B, K]
            L_rows_i = L_batch[torch.arange(B, device=device), ii]  # [B, K]
            L_rows_i = L_rows_i.to(device)
            batch_indices = torch.arange(B, device=device)
            batch_indices = batch_indices.to(device)
            x_i = x_batch[batch_indices, ii]       # [B]

            # mask for lower-triangular elements (j < i)
            lower_mask = torch.arange(K, device=device).view(1, K) < i  # [1, K]
            lower_mask = lower_mask.to(device)
            L_lower = L_rows_i & lower_mask  # [B, K], mask out upper part

            # XOR accumulate: b = dot(L[ii, :i], y[:i])
            b = torch.sum(L_lower & y_batch, dim=1) % 2  # [B]

            # Get diagonal value: L[ii, i]
            d = L_rows_i[:, i]  # [B]

            # Check solvability
            if torch.any((d == 0) & (b != x_i)):
                logger.warning('No solution in forward substitution')

            y_batch[:, i] = b ^ x_i

        del x_batch, batch_indices, x_i, b, lower_mask, L_rows_i, ii, L_batch, rows_batch
        torch.cuda.empty_cache()
        return y_batch


    def mod2sparse_backward_sub_batch(self, U_batch, cols_batch, y_batch):
        """
        Batch version of backward substitution solving U z = y over GF(2)

        Args:
            U_batch: [B, K, rr], upper-triangular matrix
            cols_batch: [B, K], column permutation
            y_batch: [B, K], right-hand side

        Returns:
            z_batch: [B, rr]
        """
        B, K, rr = U_batch.shape
        device = U_batch.device
        z_batch = torch.zeros((B, rr), dtype=torch.uint8, device=device)

        for i in reversed(range(K)):
            ii = cols_batch[:, i]  # [B]
            U_i = U_batch[:, i, :]  # [B, rr]
            y_i = y_batch[:, i]     # [B]
            ii = ii.to(device)
            # Create a one-hot mask corresponding to ii: shape [B, rr]
            onehot = torch.zeros_like(U_i)
            onehot[torch.arange(B, device=device), ii] = 1

            # d: Whether the diagonal element U[i, ii] == 1 exists
            d = (U_i & onehot).sum(dim=1)  # [B], result is either 0 or 1

            # b: XOR sum over columns other than ii (i.e., z[j] where j ≠ ii and U[i, j] == 1)
            b = torch.sum(U_i & z_batch, dim=1) % 2  # [B]
            b = b ^ (U_i[torch.arange(B), ii] & z_batch[torch.arange(B), ii]) # subtract the duplicated z[ii]

            # check if it's solvable
            unsolvable = (d == 0) & (b != y_i)
            if torch.any(unsolvable):
                logger.warning('No solution in backward substitution')

            # solve z[ii] = b ^ y[i]
            z_batch[torch.arange(B, device=device), ii] = (b ^ y_i).to(torch.uint8)

        del onehot, b, d, U_i, y_i, U_batch, cols_batch, y_batch
        torch.cuda.empty_cache()
        return z_batch


    def LU_forward_backward_solve_batch(self, L, U, rows, cols, synd):
        forward_b = self.mod2sparse_forward_sub_batch(L, rows, synd)
        osd0_decoding = self.mod2sparse_backward_sub_batch(U, cols, forward_b)

        return osd0_decoding


    def gf2_rank(self, matrix):
        """
        Function: gf2_rank
            Computes the rank of a binary matrix over GF(2).
        
        Input:
            matrix (torch.Tensor): A 2D tensor with shape (n_rows, n_cols),
            containing only 0s and 1s (binary values).
        
        Output:
            rank (int): The rank of the input matrix over the finite field GF(2),
            determined using Gaussian elimination with XOR operations.
        """
        mat = matrix.clone()
        n_rows, n_cols = mat.shape
        rank = 0

        for col in range(n_cols):
            pivot_row = None

            for row in range(rank, n_rows):
                if mat[row, col] == 1:
                    pivot_row = row
                    break

            if pivot_row is not None:
                if pivot_row != rank:
                    mat[[rank, pivot_row]] = mat[[pivot_row, rank]]

                for row in range(rank + 1, n_rows):
                    if mat[row, col] == 1:
                        mat[row] ^= mat[rank]
                rank += 1

        return rank


    def print_row_ones(matrix: torch.Tensor):
        """
        'Print which columns have a value of 1 in each row.'
        Args:
            matrix (torch.Tensor): A 2D tensor with binary values (0/1).
        """
        if matrix.dim() != 2:
            logger.warning('Input must be a 2D tensor.')

        for row_idx, row in enumerate(matrix):
            ones_idx = (row == 1).nonzero(as_tuple=True)[0]
            if len(ones_idx) == 0:
                logger.info(f"{row_idx}: (none)")
            else:
                ones_list = ones_idx.tolist()
                logger.info(f"{row_idx}: {', '.join(map(str, ones_list))}")


    def H_to_sparse(self, H_dense):
        """
        Convert a binary matrix to sparse format.
        Args:
            H (torch.Tensor): A 2D tensor with binary values (0/1).
        Returns:
            sparse_H : Sparse representation of H.
        """
        if H_dense.dim() != 2:
            logger.warning('Input must be a 2D tensor.')

        n_row, _ = H_dense.shape
        sparse_H = [
            torch.nonzero(H_dense[i], as_tuple=False).squeeze(-1).tolist()
            for i in range(n_row)
        ]
        return sparse_H


    def tuple_hash(self, t):
        # Assume B < 65536, M < 65536, N < 65536
        return (t[:, 0].to(torch.int64) << 32) | (t[:, 1].to(torch.int64) << 16) | t[:, 2].to(torch.int64)


    def apply_xor_sparse(self, coo_b, coo_r, coo_c, coo_v, gt_mask, pivot_row, pivot_row_tensor):
        device = coo_b.device
        B, N = pivot_row_tensor.shape
        B_idx_xor, M_idx_xor = torch.nonzero(gt_mask, as_tuple=True)

        if B_idx_xor.numel() == 0:
            return coo_b, coo_r, coo_c, coo_v

        # === Step 1: Expand pivot row to get new (b, r, c)
        pivot_vals = pivot_row_tensor[B_idx_xor]  # shape [P, N]
        row_sums = pivot_vals.sum(dim=1)
        repeat_idx = torch.arange(pivot_vals.size(0), device=device).repeat_interleave(row_sums)

        new_b = B_idx_xor[repeat_idx]
        new_r = M_idx_xor[repeat_idx]
        new_c = torch.nonzero(pivot_vals, as_tuple=True)[1]
        new_v = torch.ones_like(new_b, dtype=torch.uint8)

        # === Step 2: Concatenate and sort hashed keys (low-memory variant)
        base = 1_000_000
        def tuple_hash(b, r, c):
            return b * base * base + r * base + c

        old_key = tuple_hash(coo_b, coo_r, coo_c)
        new_key = tuple_hash(new_b, new_r, new_c)

        all_key = torch.cat([old_key, new_key])
        all_val = torch.cat([coo_v, new_v]).to(torch.int32)

        # === Step 3: Sort keys for memory-efficient unique
        sorted_key, sorted_idx = torch.sort(all_key)
        sorted_val = all_val[sorted_idx]

        # === Step 4: XOR accumulation via unique_consecutive
        is_new = torch.ones_like(sorted_key, dtype=torch.bool)
        is_new[1:] = sorted_key[1:] != sorted_key[:-1]

        segment_ids = torch.cumsum(is_new, dim=0) - 1
        num_segments = segment_ids[-1] + 1

        # Segment XOR: use bincount and modulo 2 to simulate GF(2)
        xor_counts = torch.bincount(segment_ids, weights=sorted_val, minlength=num_segments)
        keep = xor_counts % 2 == 1
        kept_keys = sorted_key[is_new][keep]

        if kept_keys.numel() == 0:
            return (
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.uint8, device=device),
            )

        # Decode keys
        b = kept_keys // (base * base)
        r = (kept_keys // base) % base
        c = kept_keys % base
        v = torch.ones_like(b, dtype=torch.uint8)

        return b, r, c, v
