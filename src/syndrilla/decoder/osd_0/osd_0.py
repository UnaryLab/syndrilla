import torch
from loguru import logger

from syndrilla.decoder.knobs import knob
from syndrilla.decoder.osd_0.osd_0_cuda import create as _osd_0_cuda
from syndrilla.utils import parse_device_dtype


class create(torch.nn.Module):
    """
    OSD-0 post-decoder in plain PyTorch (CPU or GPU), the same algorithm as osd_0_cuda.

    Columns are taken in stable ascending LLR order. A greedy scan picks each
    column that is independent of the pivot columns found so far, with the
    lowest row without a pivot that holds a bit as its pivot row. The scan
    keeps the row transform T (the eliminated matrix is T @ H), bit-packed and
    transposed, and the reduced syndrome T @ s, and eliminates only the rows
    without a pivot. It stops once T @ s is zero on every row without a pivot:
    s then lies in the span of the pivots found and the solution on them is
    unique. Back substitution on the pivot rows, which are upper triangular in
    pivot order, gives the estimate (see _scan).

    A syndrome outside the column space of H never stops early and scans the
    whole order; its estimate is the solution on the pivot rows only.

    Outputs: e_v (uint8) with the OSD estimate on the samples the previous
    decoder did not converge on; iter, the input iter (zeros [B] int64 when
    absent) with N on those samples; converge, all ones. Rows where the
    optional bool [B] io_dict['defer'] is True are not decoded and keep their
    input e_v, iter and converge; defer passes through unchanged.

    Config keys beyond the shared ones:
        workspace_bytes : int (optional, default 4 GiB; samples are sorted and
                          scanned in chunks whose working memory fits it, see
                          _scan_chunks; inputs and the [B, N] outputs are outside it)
        osd_early_stop  : bool (default true; false scans the whole order and
                          turns osd_prefix_scan off)
        osd_prefix_scan : bool (default true; false sorts the whole order once)
        osd_solve_by_pivots : bool (default true; false back-substitutes over
                          M pivot slots instead of the largest pivot count)
        osd_skip_converged : bool (default true; false decodes every sample and
                          keeps the result only where converge is 0)
        osd_column_scan : bool (default true; false runs a dense Gauss-Jordan
                          on [H | s] over the whole order, see _gauss_jordan)
        osd_packed_transform : bool (default true; false keeps T as a dense
                          bool matrix, see _scan_dense)
        pruning_opt     : bool (default true; false makes the defaults of
                          osd_early_stop, osd_prefix_scan, osd_solve_by_pivots
                          and osd_skip_converged false)
        memory_opt      : bool (default true; false makes the defaults of
                          osd_column_scan and osd_packed_transform false and of
                          workspace_bytes 1 << 60)
        fusion_opt, mapping_opt, gather_opt : no OSD members, ignored
    """

    def __init__(self,
                    decoding_cfg,
                    **kwargs) -> None:

        super(create, self).__init__()
        logger.info('Creating osd-0 decoder.')

        # set up default device
        self.device, _ = parse_device_dtype(decoding_cfg)

        # set up default dtype
        self.dtype = decoding_cfg.get('dtype', 'float64')
        if self.dtype not in {'float32', 'float64', 'bfloat16', 'float16'}:
            logger.warning(f'Invalid input data type <{self.dtype}>, default to <torch.float64>.')
            self.dtype = 'float64'
        self.dtype = torch.__dict__[self.dtype]
        self.algo = 'osd_0'

        bundle = kwargs.get('bundle')
        if bundle is None:
            raise ValueError('osd_0 requires a pre-loaded MatrixBundle via the `bundle` kwarg.')
        check_type = decoding_cfg.get('check_type', 'hx')
        H_shape = bundle.select(check_type)[0]
        self.num_max_iter = H_shape[1]
        self.cols = self._column_rows(bundle.select(check_type)[3].to(self.device))
        self.workspace_bytes = int(knob(decoding_cfg, 'workspace_bytes', 4 << 30))
        self.osd_early_stop = bool(knob(decoding_cfg, 'osd_early_stop', True))
        self.osd_prefix_scan = bool(knob(decoding_cfg, 'osd_prefix_scan', True)) and self.osd_early_stop
        self.osd_solve_by_pivots = bool(knob(decoding_cfg, 'osd_solve_by_pivots', True))
        self.osd_skip_converged = bool(knob(decoding_cfg, 'osd_skip_converged', True))
        self.osd_column_scan = bool(knob(decoding_cfg, 'osd_column_scan', True))
        self.osd_packed_transform = bool(knob(decoding_cfg, 'osd_packed_transform', True))
        # Columns of the reliability order the scan tries first, as in
        # osd_0_cuda; the whole order without osd_prefix_scan or osd_column_scan.
        N = int(H_shape[1])
        self.prefix = N // 16 if N >= 1 << 14 and self.osd_prefix_scan and self.osd_column_scan else N

        logger.info('Complete.')


    def forward(self, io_dict):
        logger.info('Initializing osd-0 decoding.')

        device = io_dict['synd'].device
        converge = io_dict['converge']
        B = converge.shape[0]
        iter_out = io_dict['iter'].clone() if 'iter' in io_dict else torch.zeros(B, dtype=torch.long, device=device)
        # rows marked in io_dict['defer'] are re-decoded by the batch loop, OSD skips them
        defer = io_dict['defer'].to(device=converge.device, dtype=torch.bool) if 'defer' in io_dict else None
        nc = converge == 0 if defer is None else (converge == 0) & ~defer
        idx = nc.nonzero(as_tuple=True)[0].to(device)
        # samples OSD decodes: all non-deferred ones without osd_skip_converged
        if self.osd_skip_converged:
            run = idx
        else:
            run = torch.arange(B, device=device) if defer is None else (~defer).nonzero(as_tuple=True)[0].to(device)
        llr_sub = io_dict['llr'].to(self.dtype)[run]
        synd_sub = io_dict['synd'][run].to(torch.bool)
        N = self.cols.shape[0] - 1

        logger.info('Complete.')

        logger.info('Starting decoding.')

        # The scan first runs on the first k = prefix columns of the order. A
        # sample that does not stop inside them is solved again on the first
        # 4k columns (skipped when 4k >= N), then on the full order.
        prev = min(self.prefix, N)
        e_sub, stopped = self._scan_chunks(llr_sub, synd_sub, prev)
        if prev < N:
            for width in ([4 * prev] if 4 * prev < N else []) + [N]:
                redo = (~stopped).nonzero(as_tuple=True)[0]
                if not redo.numel():
                    break
                e_sub[redo], stopped[redo] = self._scan_chunks(llr_sub[redo], synd_sub[redo], width)

        final_result = io_dict['e_v'].clone().to(dtype=torch.uint8, device=device)
        final_result[idx] = e_sub if self.osd_skip_converged else e_sub[nc.to(device)[run]]
        if idx.numel() > 0:
            iter_out = iter_out.to(device)
            iter_out[idx] = self.num_max_iter

        logger.info('Complete.')

        io_dict.update({
            'e_v': final_result,
            'iter': iter_out,
            'converge': torch.ones_like(converge) if defer is None else torch.where(defer, converge, torch.ones_like(converge))
        })
        return io_dict


    @staticmethod
    def _column_rows(H):
        """[N + 1, D] row indices of each column of H (dense or sparse COO), padded
        with M; the extra column N is all padding. D is the largest column weight
        rounded up to a power of two."""
        M, N = H.shape
        if H.is_sparse:
            H = H.coalesce()
            r, c = H.indices()[:, H.values().bool()]
        else:
            r, c = H.nonzero(as_tuple=True)
        c, by_col = torch.sort(c, stable=True)
        r = r[by_col]
        count = torch.bincount(c, minlength=N + 1)
        start = torch.cumsum(count, 0) - count
        D = 1 << (max(int(count.max()), 1) - 1).bit_length()
        table = torch.full((N + 1, D), M, dtype=torch.long, device=H.device)
        table[c, torch.arange(c.numel(), device=H.device) - start[c]] = r
        return table


    def _scan_chunks(self, llr, synd, width):
        """_scan over the batch in chunks of samples whose working memory fits
        workspace_bytes. Per sample, with U = ceil((M + 1) / 64) words (bytes): the
        packed transposed row transform, 8*(M+1)*U, plus the row-update
        temporaries, 4*(M+1)*U; the gathered column words and their XOR fold,
        16*D*U; the per-row vectors (a T column, its nonzero indices, the pivot
        rows and columns), 64*(M+1); the packed per-sample vectors, 48*U; and
        the order (its sort and the kept int64 copy) plus the uint8 estimate,
        60*N. Without osd_packed_transform the chunks run _scan_dense, without
        osd_column_scan _gauss_jordan, each sized by the bytes in its docstring."""
        M = synd.shape[1]
        N, D = self.cols.shape[0] - 1, self.cols.shape[1]
        U = (M >> 6) + 1  # bit M exists and stays zero
        cols = self.cols.to(synd.device)
        if not self.osd_column_scan:
            scan, per_sample = self._gauss_jordan, 3 * M * (N + 1) + 60 * N
        elif not self.osd_packed_transform:
            scan, per_sample = self._scan_dense, 2 * M * (M + 1) + M * D + 24 * M + 48 * N
        else:
            scan = self._scan
            per_sample = 12 * (M + 1) * U + 16 * D * U + 64 * (M + 1) + 48 * U + 60 * N
        step = max(1, self.workspace_bytes // per_sample)
        parts = [scan(cols, llr[i:i + step], synd[i:i + step], width)
                 for i in range(0, max(len(llr), 1), step)]
        return torch.cat([e for e, _ in parts]), torch.cat([d for _, d in parts])


    def _scan(self, cols, llr, synd, width):
        """OSD-0 over the first width columns of each sample's reliability
        order, in two phases. Returns e [B, N] uint8 and stopped [B] bool.

        Scan: T is kept transposed and bit-packed, TuT[b, k] = column k of T as
        bits over rows u (bit u of word u >> 6). Column c of T @ H is the XOR of
        TuT at c's rows. The pivot is the lowest row without a pivot that holds
        a bit; every other such row u gets T_u ^= T_p, which is TuT[k] ^= v for
        each k in T_p. Pivot rows are not updated, so the pivot rows of T @ H
        are upper triangular in pivot order. The reduced syndrome T @ s follows
        the same row operations, and a sample stops once it is zero on every
        row without a pivot (never without osd_early_stop).

        Solve: back substitution on the pivot rows. y starts as T @ s; the
        last pivot (in scan order) whose row holds a bit of y is final, so its
        column is set in e and y ^= T @ h_c. This repeats until y is zero on
        every pivot row: the loop runs the batch maximum of popcount(e) steps,
        with one host sync each. It reads the first K pivot slots, K the batch
        maximum pivot count (M without osd_solve_by_pivots); the slots past a
        sample's pivot count hold row M and column N, which are inert."""
        order = _osd_0_cuda._order_prefix(llr, width).long()
        B, W = order.shape
        N, M = cols.shape[0] - 1, synd.shape[1]
        U = (M >> 6) + 1  # bit M exists and stays zero
        dev = synd.device
        ar = torch.arange(B, device=dev)
        b64 = torch.arange(64, device=dev)
        k = torch.arange(M, device=dev)
        one = torch.ones((), dtype=torch.long, device=dev)

        def pack(bits):
            """[B, M] -> [B, U] int64, bit u in word u >> 6; bits M and up
            are zero (bit M is read for the padding row M)."""
            x = torch.zeros(B, U * 64, dtype=torch.long, device=dev)
            x[:, :M] = bits
            return (x.view(B, U, 64) << b64).sum(2)  # distinct bits: sum is OR

        def column(c):
            """[B, U] T @ h_c, the XOR of TuT at c's rows (D a power of two)."""
            g = TuT[ar[:, None], cols[c]]
            while g.shape[1] > 1:
                h = g.shape[1] >> 1
                g = g[:, :h] ^ g[:, h:]
            return g[:, 0]

        # row M stays zero: the padding of cols gathers it
        TuT = torch.zeros(B, M + 1, U, dtype=torch.long, device=dev)
        TuT[:, k, k >> 6] = one << (k & 63)
        sres = pack(synd)
        alive = pack(torch.ones(B, M, dtype=torch.long, device=dev))
        piv_r = torch.full((B, M + 1), M, dtype=torch.long, device=dev)
        piv_c = torch.full((B, M + 1), N, dtype=torch.long, device=dev)
        found = torch.zeros(B, dtype=torch.long, device=dev)
        live = (sres != 0).any(1) | (not self.osd_early_stop)  # not yet stopped
        piece = max(1, B * (M + 1) // 4)  # T rows per update step
        j = 0
        while j < W and (j & 7 or bool(live.any())):
            c = order[:, j]
            v = column(c) & alive & -live.long()[:, None]
            nz = v != 0
            has = nz.any(1)
            w = nz.to(torch.uint8).argmax(1)
            bit = ((v[ar, w][:, None] >> b64) & 1).argmax(1)
            alive[ar, w] ^= has.long() << bit
            v &= alive  # without the pivot row
            sp = (sres[ar, w] >> bit) & 1
            sres ^= v & -sp[:, None]
            piv_r[ar, found] = torch.where(has, (w << 6) + bit, M)
            piv_c[ar, found] = torch.where(has, c, N)
            found += has
            upd = ((TuT[ar, :, w] >> bit[:, None]) & 1).bool() & (v != 0).any(1)[:, None]
            bi, ki = upd.nonzero(as_tuple=True)
            for a in range(0, bi.numel(), piece):
                b_, k_ = bi[a:a + piece], ki[a:a + piece]
                TuT[b_, k_] ^= v[b_]
            live &= ((sres & alive) != 0).any(1) | (not self.osd_early_stop)
            j += 1

        e = torch.zeros(B, N + 1, dtype=torch.uint8, device=dev)
        K = (int(found.max()) if self.osd_solve_by_pivots else M) if B else 0
        pr, pc = piv_r[:, :K], piv_c[:, :K]  # padded with row M / column N
        y = sres
        while K:
            hit = ((y.gather(1, pr >> 6) >> (pr & 63)) & 1).bool()
            last = hit.any(1)
            if not bool(last.any()):
                break
            i = K - 1 - hit.flip(1).to(torch.uint8).argmax(1)
            c = torch.where(last, pc[ar, i], N)
            e[ar, c] = 1
            y ^= column(c)
        return e[:, :N], ~live


    def _scan_dense(self, cols, llr, synd, width):
        """_scan with T as a dense bool [M, M + 1] matrix and Gauss-Jordan on
        every row: the pivot rows of T @ H form the identity on the pivot
        columns, and the estimate is T @ s on the pivot rows. Same pivots, stop
        rule and outputs as _scan. Bytes per sample: T and the same-size
        temporary of each update, 2*M*(M+1); the gathered column bits, M*D;
        the per-row vectors, 24*M; and the order plus the estimate, 48*N."""
        order = _osd_0_cuda._order_prefix(llr, width).long()
        B, W = order.shape
        N, M = cols.shape[0] - 1, synd.shape[1]
        dev = synd.device
        ar = torch.arange(B, device=dev)
        # column M stays zero: the padding of cols gathers it
        T = torch.eye(M, M + 1, dtype=torch.bool, device=dev).repeat(B, 1, 1)
        sres = synd.clone()  # T @ s
        alive = torch.ones(B, M, dtype=torch.bool, device=dev)  # rows without a pivot
        # H column of each pivot row; N for rows without a pivot
        row_col = torch.full((B, M), N, dtype=torch.long, device=dev)
        done = ~sres.any(1) & self.osd_early_stop
        j = 0
        while j < W and not bool(done.all()):
            c = order[:, j]
            # Column c of T @ H: XOR of T's columns at c's rows.
            g = T.gather(2, cols[c][:, None, :].expand(B, M, cols.shape[1]))
            v = (g.sum(2, dtype=torch.uint8) & 1).bool() & ~done[:, None]
            cand = v & alive
            has = cand.any(1)
            if bool(has.any()):
                # lowest row without a pivot that has the bit
                p = cand.to(torch.uint8).argmax(1)
                v[ar, p] = False
                v &= has[:, None]
                T ^= v[:, :, None] & T[ar, p][:, None, :]
                sres ^= v & sres[ar, p][:, None]
                alive[ar[has], p[has]] = False
                row_col[ar[has], p[has]] = c[has]
                done |= ~(sres & alive).any(1) & self.osd_early_stop
            j += 1
        e = torch.zeros(B, N + 1, dtype=torch.uint8, device=dev)
        # rows without a pivot land in column N
        e.scatter_(1, row_col, sres.to(torch.uint8))
        return e[:, :N], done


    def _gauss_jordan(self, cols, llr, synd, width):
        """Dense Gauss-Jordan on [H | s] over the first width columns of each
        sample's reliability order, no early stop. The pivot of a column is the
        lowest row without a pivot that holds a bit, as in _scan, so the pivot
        rows and the estimate (s reduced, on the pivot rows) are the same.
        Returns e [B, N] uint8 and stopped [B] bool (all true). Bytes per
        sample: [H | s], the same-size temporary of each update and the
        gathered H, 3*M*(N+1); the order plus the estimate, 60*N."""
        order = _osd_0_cuda._order_prefix(llr, width).long()
        B, W = order.shape
        N, M = cols.shape[0] - 1, synd.shape[1]
        dev = synd.device
        ar = torch.arange(B, device=dev)
        H = torch.zeros(M + 1, N + 1, dtype=torch.bool, device=dev)
        H[cols, torch.arange(N + 1, device=dev)[:, None]] = True  # row M: padding
        A = torch.cat([H[:M, order].permute(1, 0, 2), synd[:, :, None]], 2)  # [B, M, W + 1]
        alive = torch.ones(B, M, dtype=torch.bool, device=dev)  # rows without a pivot
        # H column of each pivot row; N for rows without a pivot
        row_col = torch.full((B, M), N, dtype=torch.long, device=dev)
        for j in range(W):
            v = A[:, :, j].clone()
            cand = v & alive
            has = cand.any(1)
            p = cand.to(torch.uint8).argmax(1)  # lowest row without a pivot
            v[ar, p] = False
            v &= has[:, None]
            A ^= v[:, :, None] & A[ar, p][:, None, :]
            alive[ar[has], p[has]] = False
            row_col[ar[has], p[has]] = order[ar[has], j]
        e = torch.zeros(B, N + 1, dtype=torch.uint8, device=dev)
        e.scatter_(1, row_col, A[:, :, W].to(torch.uint8))
        return e[:, :N], torch.ones(B, dtype=torch.bool, device=dev)
