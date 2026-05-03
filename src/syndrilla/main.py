import torch
import re
import sys, os, time
import pyfiglet, argparse, time
import numpy as np
import yaml
import subprocess
from loguru import logger
from syndrilla.utils import bcolors, read_yaml, get_path, parse_device_dtype
from syndrilla.decoder import create_decoder
from syndrilla.error_model import create_error_model
from syndrilla.syndrome import create_syndrome
from syndrilla.metric import report_metric, save_metric, MetricState, BatchTracker
from syndrilla.matrix import load_matrices
from syndrilla.logical_check import create_check
from syndrilla.interface import create_interface
from syndrilla.vote import create_vote


def parse_commandline_args():
    """
    parse command line inputs
    """
    parser = argparse.ArgumentParser(
        description='A PyTorch-based numerical simulator for decoders in quantum error correction.')
    parser.add_argument('-r', '--run_dir', type=str, default='tests/test_outputs',
                        help = 'Run directory to store outputs.')
    parser.add_argument('-d', '--decoder_yaml', type=str, default=None,
                        help = 'Path to decoder yaml.')
    parser.add_argument('-e', '--error_yaml', type=str, default=None,
                        help = 'Path to error model yaml.')
    parser.add_argument('-c', '--logical_yaml', type=str, default=None,
                        help = 'Path to logical error check yaml.')
    parser.add_argument('-s', '--syndrome_yaml', type=str, default=None,
                        help = 'Path to syndrome yaml.')
    parser.add_argument('-ckpt', '--checkpoint_yaml', type=str, default=None,
                        help = 'Path to checkpoint result yaml to resume from.')
    parser.add_argument('-bs', '--batch_size', type=int, default=10000,
                        help = 'Number of samples run each batch.')
    parser.add_argument('-te', '--target_error', type=int, default=100,
                        help = 'Total number of errors to stop decoding.')
    parser.add_argument('-m', '--matrix_yaml', type=str, default=None,
                        help = 'Path to matrix yaml.')
    parser.add_argument('-i', '--interface_yaml', type=str, default=None,
                        help = 'Path to interface yaml (replaces -e, -c, -s and matrix configs).')
    parser.add_argument('-l', '--log_level', type=str, default='INFO',
                        help = 'Level of logger.')
    parser.add_argument('-dr', '--rounds', type=int, default=1,
                        help = 'Number of syndrome measurement rounds (majority vote when > 1).')
    parser.add_argument('-vs', '--vote_stage', type=str, default=None,
                        help = 'Where to apply majority vote. If not set, no voting is invoked '
                               '(syndromes/decoder outputs keep the rounds dimension). '
                               '"syndrome" = vote on syndromes then decode, '
                               '"decoder_N" = vote after decoder N (0-based), '
                               'remaining decoders run once on voted result. '
                               'e.g. decoder_0, decoder_1, decoder (= last).')

    return parser.parse_args()


def main():
    args = parse_commandline_args()

    # set up output log
    logger.remove()
    output_log = args.run_dir + '/main' + '-' + str(time.time()) + '.log'
    logger.add(output_log, level=args.log_level)

    # set up banner
    ascii_banner = pyfiglet.figlet_format('SYNDRILLA')
    print(bcolors.Magenta + ascii_banner + bcolors.ENDC)
    ascii_banner = pyfiglet.figlet_format('UnaryLab')
    print(bcolors.Yellow + ascii_banner + bcolors.ENDC)
    ascii_banner = pyfiglet.figlet_format('https://github.com/UnaryLab/syndrilla', font='term')
    print(bcolors.UNDERLINE + bcolors.Green + ascii_banner + bcolors.ENDC)

    if args.interface_yaml is not None:
        logger.success(f'\n----------------------------------------------\nStep 1: Create interface\n----------------------------------------------')
        interface = create_interface(
            args.interface_yaml,
            error_yaml=args.error_yaml,
            syndrome_yaml=args.syndrome_yaml,
            decoder_yaml=args.decoder_yaml,
        )
        logger.success(f'\n----------------------------------------------\nStep 2: Create error model\n----------------------------------------------')
        error_model = interface.error_model
        logger.success(f'\n----------------------------------------------\nStep 3: Create syndrome measurer\n----------------------------------------------')
        syndrome_generator = interface.syndrome_generator
        logger.success(f'\n----------------------------------------------\nStep 4: Create logical error checker\n----------------------------------------------')
        logical_check = interface.logical_check
        bundle = interface.matrix_bundle
        decoders = interface.decoders
    else:
        logger.success(f'\n----------------------------------------------\nStep 1: Create decoder\n----------------------------------------------')
        decoder_cfg = read_yaml(get_path(args.decoder_yaml))['decoder']
        matrix_cfg = read_yaml(get_path(args.matrix_yaml))['matrix']
        bundle = load_matrices(matrix_cfg, *parse_device_dtype(decoder_cfg))
        decoders = create_decoder(cfg=decoder_cfg, bundle=bundle)

        logger.success(f'\n----------------------------------------------\nStep 2: Create error model\n----------------------------------------------')
        error_model = create_error_model(args.error_yaml)

        logger.success(f'\n----------------------------------------------\nStep 3: Create syndrome measurer\n----------------------------------------------')
        syndrome_generator = create_syndrome(args.syndrome_yaml)

        logger.success(f'\n----------------------------------------------\nStep 4: Create logical error checker\n----------------------------------------------')
        logical_check = create_check(args.logical_yaml)

    num_decoders = len(decoders)
    dtype = decoders[0].dtype
    decoder_device = decoders[0].device

    voter = create_vote(cfg={'method': 'majority_vote'})

    number_channel = error_model.number_channel
    if args.decoder_yaml is not None:
        check_type = read_yaml(get_path(args.decoder_yaml))['decoder'].get('check_type', 'hx')
    else:
        check_type = 'hx'
    shape, _, _, _ = bundle.Hx_matrix.get_index()
    H_matrix = bundle.select(check_type)[3]

    algo_name = []
    num_max_iter = []
    for decoder in decoders:
        decoder.eval()
        algo_name.append(decoder.algo)
        num_max_iter.append(getattr(decoder, 'num_max_iter', 0))

    H_file_name = bundle.get_H_file_name(check_type, number_channel)
    l_matrix = bundle.get_l_matrix(check_type, number_channel)
    check_num = bundle.get_check_num(check_type, number_channel)

    num_err = 0
    num_batches = 0

    # initialize metric state
    metrics = MetricState(num_decoders, number_channel, decoder_device)

    logger.success(f'\n----------------------------------------------\nStep 5: Check checkpoint file\n----------------------------------------------')
    # To check whether there is a resume yaml
    if args.checkpoint_yaml is not None:
        if not os.path.isfile(args.checkpoint_yaml):
            raise FileNotFoundError(f'Checkpoint file not found: {args.checkpoint_yaml}')
        metrics, ckpt_meta = MetricState.from_checkpoint(args.checkpoint_yaml, number_channel, decoder_device)
        metrics.validate_checkpoint(ckpt_meta, args.batch_size, args.target_error, dtype, error_model.rate, H_file_name)
        num_err = ckpt_meta['num_err']
        num_batches = ckpt_meta['batch_count']
    else:
        logger.info(f'No input Checkpoint file.')

    while num_err <= args.target_error:
        logger.success(f'\n----------------------------------------------\nStep 6: Generate error\n----------------------------------------------')
        bt = BatchTracker(num_decoders, number_channel, shape, dtype, decoder_device)

        # create error
        zero_qubits = torch.zeros([args.batch_size, shape[1]], dtype=dtype)
        error_vector, error_dataloader = error_model.inject_error(zero_qubits, args.batch_size)
        num_batches += 1

        avg_error_rate = torch.mean(torch.sum(error_vector, 1) / shape[1])
        logger.info(f'Specified error rate <{error_model.rate}>.')
        logger.info(f'Generated error rate <{avg_error_rate}>.')

        for err, llr, _ in error_dataloader:
            bt.record_error(err)

            rounds = getattr(syndrome_generator, 'rounds', 1)
            vote_recorded = False

            logger.success(f'\n----------------------------------------------\nStep 7: Measure syndrome\n----------------------------------------------')
            synd = syndrome_generator.measure_syndrome(err, decoders[0])
            synd = voter.apply(synd, number_channel, rounds=rounds, vote_stage=args.vote_stage, current_stage='syndrome')
            if not vote_recorded and voter.last_sample_count > 0:
                metrics.accumulate_vote(voter.last_match_counts, voter.last_sample_count, voter.last_voted_stage, voter.last_rounds)
                vote_recorded = True

            io_dict = {
                'synd': synd,
                'llr0': llr,
                'H_matrix': H_matrix
            }

            logger.success(f'\n----------------------------------------------\nStep 8: Decode\n----------------------------------------------')
            for decoder_idx in range(num_decoders):
                start_time = time.time()
                io_dict = decoders[decoder_idx](io_dict)
                elapsed = time.time() - start_time

                decoder_stage = f'decoder_{decoder_idx}'
                io_dict['e_v'] = voter.apply(io_dict['e_v'], number_channel, rounds=rounds, vote_stage=args.vote_stage, current_stage=decoder_stage)
                if not vote_recorded and voter.last_sample_count > 0:
                    metrics.accumulate_vote(voter.last_match_counts, voter.last_sample_count, voter.last_voted_stage, voter.last_rounds)
                    vote_recorded = True
                io_dict['synd'] = voter.apply(io_dict['synd'], number_channel, rounds=rounds, vote_stage=args.vote_stage, current_stage=decoder_stage)
                for key in ('llr', 'converge', 'iter'):
                    if key in io_dict and io_dict[key].ndim > 1:
                        io_dict[key] = voter.select_round(io_dict[key], rounds=rounds, vote_stage=args.vote_stage, current_stage=decoder_stage)
                bt.record_decoder(decoder_idx, io_dict, elapsed)
                
            logger.success(f'\n----------------------------------------------\nStep 9: Check logical error rate\n----------------------------------------------')

            has_obs_flips = hasattr(syndrome_generator, 'observable_flips') and syndrome_generator.observable_flips is not None
            check_error = syndrome_generator.observable_flips.to(dtype) if has_obs_flips else bt.e_all

            check = [[] for _ in range(num_decoders)]
            for i in range(num_decoders):
                check[i] = logical_check.check(bt.e_v_all[i], check_error, l_matrix, bt.converge_all[i + 1])
            num_err += int(torch.sum(check[num_decoders-1]))
            logger.info(f'number of errors at the current batch {num_err}/{args.target_error}')

            # report and accumulate metrics
            logger.success(f'\n----------------------------------------------\nStep 10: Aggregate metrics\n----------------------------------------------')
            if number_channel == 1:
                bt.e_v_all = [t.unsqueeze(1).expand(-1, number_channel, -1) for t in bt.e_v_all]
                check = [t.unsqueeze(1).expand(-1, number_channel) for t in check]
            for i in range(num_decoders):
                batch_result = report_metric(num_max_iter[i], bt.e_all, bt.e_v_all[i], bt.iter_all[i], bt.time_iter_all[i], check[i], bt.converge_all[i], bt.converge_all[i+1], i)
                metrics.accumulate(i, batch_result)

            if num_batches % 100 == 0:
                logger.success(f'\n----------------------------------------------\nStep 11: Save batch log\n----------------------------------------------')
                all_metrics = metrics.get_all_metrics(num_batches, algo_name)
                save_metric(all_metrics, args.run_dir + '/', args.batch_size, args.target_error, str(dtype), error_model.rate, num_batches, num_err, H_file_name, check_num, vote_info=metrics.get_vote_info())
                logger.success(f'Saved log to <{output_log}>.')
                logger.success(f'Saved metric results to <{args.run_dir}>.')

    logger.success(f'\n----------------------------------------------\nStep 12: Save final log\n----------------------------------------------')
    all_metrics = metrics.get_all_metrics(num_batches, algo_name)
    save_metric(all_metrics, args.run_dir + '/', args.batch_size, args.target_error, str(dtype), error_model.rate, num_batches, num_err, H_file_name, check_num, 1, vote_info=metrics.get_vote_info())
    logger.success(f'Saved log to <{output_log}>.')
    logger.success(f'Saved metric results to <{args.run_dir}>.')


if __name__ == '__main__':
    main()
