import torch
import torch.nn.functional as F
import sys, os, time, copy, math
import yaml

from loguru import logger

sys.path.append(os.getcwd())

from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.syndrome import create_syndrome
from syndrilla.logical_check import create_check


def train_bp(
    decoder_yaml='examples/alist/bp_hx.decoder.yaml',
    error_yaml='examples/alist/bsc.error.yaml',
    syndrome_yaml='examples/alist/perfect.syndrome.yaml',
    logical_check_yaml='examples/alist/lx.check.yaml',
    batch_size=500,
    num_epochs=50,
    lr=1e-3,
    eval_batch_size=1000,
    eval_target_errors=200,
    save_path='tests/test_outputs/trained_bp.pt',
):
    """Train the bp_norm_min_sum decoder with learnable per-iteration weights."""

    # ------------------------------------------------------------------ #
    # 1. Create modules (decoder, error model, syndrome, logical check)
    # ------------------------------------------------------------------ #
    # patch the decoder yaml to enable trainable mode
    with open(decoder_yaml, 'r') as f:
        decoder_cfg_raw = yaml.safe_load(f)

    decoder_cfg_raw['decoder']['trainable'] = True
    # use a reasonable max_iter for training (fewer iterations = faster gradients)
    train_max_iter = decoder_cfg_raw['decoder'].get('max_iter', 50)
    if train_max_iter > 50:
        decoder_cfg_raw['decoder']['max_iter'] = 50
        train_max_iter = 50

    # write a temporary yaml for the trainable decoder
    tmp_decoder_yaml = 'tests/test_outputs/tmp_train_decoder.yaml'
    os.makedirs(os.path.dirname(tmp_decoder_yaml), exist_ok=True)
    with open(tmp_decoder_yaml, 'w') as f:
        yaml.dump(decoder_cfg_raw, f)

    decoders = create_decoder(tmp_decoder_yaml)
    decoder = decoders[0]

    error_model = create_error_model(error_yaml)
    syndrome_generator = create_syndrome(syndrome_yaml)
    logical_check = create_check(logical_check_yaml)

    shape = decoder.H_shape
    dtype = decoder.dtype
    device = decoder.device

    if decoder.check_type.lower() == 'hx':
        l_matrix = decoder.lx_matrix
    else:
        l_matrix = decoder.lz_matrix

    # ------------------------------------------------------------------ #
    # 2. Set up optimizer
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    logger.success(f'Trainable parameters:')
    for name, p in decoder.named_parameters():
        if p.requires_grad:
            logger.success(f'  {name}: {p.shape}')

    # ------------------------------------------------------------------ #
    # 3. Evaluate before training (baseline)
    # ------------------------------------------------------------------ #
    baseline_ler = evaluate(decoder, error_model, syndrome_generator, logical_check,
                            l_matrix, shape, dtype, device,
                            eval_batch_size, eval_target_errors)
    logger.success(f'Baseline logical error rate: {baseline_ler:.6f}')

    # ------------------------------------------------------------------ #
    # 4. Training loop
    # ------------------------------------------------------------------ #
    best_ler = baseline_ler
    best_state = copy.deepcopy(decoder.state_dict())

    for epoch in range(1, num_epochs + 1):
        decoder.train()
        epoch_loss = 0.0
        num_batches = 0

        # generate a fresh batch of errors each epoch
        zero_qubits = torch.zeros([batch_size, shape[1]], dtype=dtype)
        error_vector, error_dataloader = error_model.inject_error(zero_qubits, batch_size)

        for err, llr, _ in error_dataloader:
            err = err.to(device)
            synd = syndrome_generator.measure_syndrome(err, decoder)

            io_dict = {
                'synd': synd,
                'llr0': llr,
                'H_matrix': decoder.H_matrix,
            }

            # forward
            optimizer.zero_grad()
            io_dict = decoder(io_dict)

            # soft BER loss: use the soft LLR output
            # negative LLR means error predicted → target is the true error
            target = err.to(device).to(dtype)
            loss = F.binary_cross_entropy_with_logits(-io_dict['llr'], target)

            # backward + step
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)

        # evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            ler = evaluate(decoder, error_model, syndrome_generator, logical_check,
                           l_matrix, shape, dtype, device,
                           eval_batch_size, eval_target_errors)
            improved = ''
            if ler < best_ler:
                best_ler = ler
                best_state = copy.deepcopy(decoder.state_dict())
                improved = ' *'
            logger.success(f'Epoch {epoch:3d} | loss={avg_loss:.6f} | LER={ler:.6f}{improved}')
        else:
            logger.info(f'Epoch {epoch:3d} | loss={avg_loss:.6f}')

    # ------------------------------------------------------------------ #
    # 5. Save best model
    # ------------------------------------------------------------------ #
    decoder.load_state_dict(best_state)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state, save_path)
    logger.success(f'Best model saved to {save_path} (LER={best_ler:.6f})')

    # print learned weights
    logger.success(f'Learned CN weights: {decoder.weights_cn.data}')
    logger.success(f'Learned LLR weights: {decoder.weights_llr.data}')

    # clean up temp file
    if os.path.exists(tmp_decoder_yaml):
        os.remove(tmp_decoder_yaml)

    return best_ler


@torch.no_grad()
def evaluate(decoder, error_model, syndrome_generator, logical_check,
             l_matrix, shape, dtype, device, batch_size, target_errors):
    """Run evaluation and return logical error rate."""
    decoder.eval()

    num_errors = 0
    num_samples = 0

    while num_errors < target_errors:
        zero_qubits = torch.zeros([batch_size, shape[1]], dtype=dtype)
        error_vector, error_dataloader = error_model.inject_error(zero_qubits, batch_size)

        for err, llr, _ in error_dataloader:
            err = err.to(device)
            synd = syndrome_generator.measure_syndrome(err, decoder)

            io_dict = {
                'synd': synd,
                'llr0': llr,
                'H_matrix': decoder.H_matrix,
            }

            io_dict = decoder(io_dict)

            check_result = logical_check.check(io_dict['e_v'], err, l_matrix, io_dict['converge'])

            num_errors += check_result.sum().item()
            num_samples += batch_size

    ler = num_errors / num_samples if num_samples > 0 else 0.0
    return ler


if __name__ == '__main__':
    train_bp()
