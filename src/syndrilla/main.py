import argparse
import sys
import time

import pyfiglet
import torch
from loguru import logger

from syndrilla.decoder import assert_trainable, create_decoder, resolve_configs
from syndrilla.error_model import create_error_model
from syndrilla.interface import create_interface
from syndrilla.logical_check import create_check
from syndrilla.loss import create_loss
from syndrilla.matrix import load_matrices
from syndrilla.metric import (
    BatchTracker,
    MetricState,
)
from syndrilla.syndrome import create_syndrome
from syndrilla.utils import ExtraQueue, bcolors, get_path, parse_device_dtype, read_yaml


def parse_commandline_args():
    """
    parse command line inputs
    """
    parser = argparse.ArgumentParser(
        description="A PyTorch-based numerical simulator for decoders in quantum error correction."
    )
    parser.add_argument(
        "-r",
        "--run_dir",
        type=str,
        default="tests/test_outputs",
        help="Run directory to store outputs, for both decoding and training.",
    )
    parser.add_argument(
        "-d", "--decoder_yaml", type=str, default=None, help="Path to decoder yaml."
    )
    parser.add_argument(
        "-e", "--error_yaml", type=str, default=None, help="Path to error model yaml."
    )
    parser.add_argument(
        "-c",
        "--logical_yaml",
        type=str,
        default=None,
        help="Path to logical error check yaml.",
    )
    parser.add_argument(
        "-s", "--syndrome_yaml", type=str, default=None, help="Path to syndrome yaml."
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint_yaml",
        type=str,
        default=None,
        help="Path to checkpoint result yaml to resume from. With -t, the training run's <stem>_result.yaml, which is resumed alongside its -tckpt.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=10000,
        help="Number of samples run each batch.",
    )
    parser.add_argument(
        "-te",
        "--target_error",
        type=int,
        default=100,
        help="Total number of errors to stop decoding.",
    )
    parser.add_argument(
        "-m", "--matrix_yaml", type=str, default=None, help="Path to matrix yaml."
    )
    parser.add_argument(
        "-i",
        "--interface_yaml",
        type=str,
        default=None,
        help="Path to interface yaml (replaces -e, -c, -s and matrix configs).",
    )
    parser.add_argument(
        "-l", "--log_level", type=str, default="INFO", help="Level of logger."
    )

    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train the decoder instead of decoding. Writes <decoder>_<check>_<size>{_best.pt,_last.pt,_result.yaml,_train.log} to --run_dir.",
    )
    parser.add_argument(
        "-ls",
        "--loss_yaml",
        type=str,
        default=None,
        help="Path to loss yaml (see examples/alist/logical_centric.loss.yaml).",
    )
    # not '-tc': argparse splits that into '-t -c', so a typo would start a training
    # run instead of failing. '-tckpt' mirrors the decode-side '-ckpt'.
    parser.add_argument(
        "-tckpt",
        "--train_checkpoint",
        type=str,
        default=None,
        help="Path to a run's *.pt, to continue that run where it stopped.",
    )

    return parser.parse_args()


def main():
    args = parse_commandline_args()

    # set up output log
    logger.remove()
    if args.train:
        logger.add(sys.stderr, level="WARNING")
    else:
        output_log = args.run_dir + "/main" + "-" + str(time.time()) + ".log"
        logger.add(output_log, level=args.log_level)

    # set up banner
    ascii_banner = pyfiglet.figlet_format("SYNDRILLA")
    print(bcolors.Magenta + ascii_banner + bcolors.ENDC)
    ascii_banner = pyfiglet.figlet_format("UnaryLab")
    print(bcolors.Yellow + ascii_banner + bcolors.ENDC)
    ascii_banner = pyfiglet.figlet_format(
        "https://github.com/UnaryLab/syndrilla", font="term"
    )
    print(bcolors.UNDERLINE + bcolors.Green + ascii_banner + bcolors.ENDC)

    if args.train_checkpoint is not None and not args.train:
        raise ValueError(
            "-tckpt resumes a training run, so it needs -t. To decode from a "
            "checkpoint, put its path under the decoder yaml's `checkpoint` key."
        )

    if args.train and (args.checkpoint_yaml is None) != (args.train_checkpoint is None):
        given, missing = (
            ("-tckpt", "-ckpt <run>_result.yaml")
            if args.checkpoint_yaml is None
            else ("-ckpt", "-tckpt <run>_last.pt")
        )
        raise ValueError(
            f"Resuming a training run takes both of its checkpoints, but {given} was "
            f"given without {missing}. Both were printed by the run that wrote them."
        )

    required = [("-d", "decoder_yaml")]
    if args.interface_yaml is None:
        required += [
            ("-m", "matrix_yaml"),
            ("-e", "error_yaml"),
            ("-s", "syndrome_yaml"),
            ("-ls", "loss_yaml") if args.train else ("-c", "logical_yaml"),
        ]
    elif args.train:
        required += [("-ls", "loss_yaml")]
    missing = [flag for flag, name in required if getattr(args, name) is None]
    if missing:
        mode = "Training" if args.train else "Decoding"
        raise ValueError(f"{mode} requires {' '.join(missing)}.")

    decoder_cfg = read_yaml(get_path(args.decoder_yaml))["decoder"]

    if args.train:
        trained_cfg = resolve_configs(decoder_cfg, f"<{args.decoder_yaml}>")[-1]
        metrics = MetricState.train_initial(
            trained_cfg.get("train"), args.run_dir, args.decoder_yaml
        )
        torch.manual_seed(metrics.cfg["error_random_seed"])
        yaml_ckpt = trained_cfg.get("checkpoint")
        if args.train_checkpoint is not None and yaml_ckpt:
            raise ValueError(
                f"-tckpt <{args.train_checkpoint}> and the decoder yaml's "
                f"`config.checkpoint` <{yaml_ckpt}> both supply weights; drop the yaml key."
            )

    if args.interface_yaml is not None:
        logger.success(
            "\n----------------------------------------------\nStep 1: Create interface\n----------------------------------------------"
        )
        interface = create_interface(
            args.interface_yaml,
            error_yaml=args.error_yaml,
            syndrome_yaml=args.syndrome_yaml,
            decoder_yaml=args.decoder_yaml,
            training=args.train,
        )
        logger.success(
            "\n----------------------------------------------\nStep 2: Create error model\n----------------------------------------------"
        )
        error_model = interface.error_model
        logger.success(
            "\n----------------------------------------------\nStep 3: Create syndrome measurer\n----------------------------------------------"
        )
        syndrome_generator = interface.syndrome_generator
        logger.success(
            "\n----------------------------------------------\nStep 4: Create logical error checker\n----------------------------------------------"
        )
        # the interface leaves this None when training, which reads its loss straight
        # off the decoder output and never checks a logical error
        logical_check = interface.logical_check
        bundle = interface.matrix_bundle
        decoders = interface.decoders

        if args.train:
            loss = create_loss(args.loss_yaml, decoder=decoders[-1])
    else:
        logger.success(
            "\n----------------------------------------------\nStep 1: Create decoder\n----------------------------------------------"
        )
        matrix_cfg = read_yaml(get_path(args.matrix_yaml))["matrix"]
        bundle = load_matrices(matrix_cfg, *parse_device_dtype(decoder_cfg))
        decoders = create_decoder(cfg=decoder_cfg, bundle=bundle, training=args.train)

        logger.success(
            "\n----------------------------------------------\nStep 2: Create error model\n----------------------------------------------"
        )
        error_model = create_error_model(args.error_yaml, training=args.train)

        logger.success(
            "\n----------------------------------------------\nStep 3: Create syndrome measurer\n----------------------------------------------"
        )
        syndrome_generator = create_syndrome(args.syndrome_yaml, training=args.train)

        if args.train:
            loss = create_loss(args.loss_yaml, decoder=decoders[-1])

        logger.success(
            "\n----------------------------------------------\nStep 4: Create logical error checker\n----------------------------------------------"
        )
        # training never checks logical errors: it reads its loss straight off the decoder output, so the checker is never built
        logical_check = None if args.train else create_check(args.logical_yaml)

    error_model.rounds = getattr(syndrome_generator, "rounds", 1)

    num_decoders = len(decoders)
    dtype = decoders[0].dtype
    decoder_device = decoders[0].device

    number_channel = error_model.number_channel
    expected_channel = 2 if getattr(decoders[0], "_base_synd_ndim", 2) == 3 else 1
    if number_channel != expected_channel:
        # the interface path builds the error model itself, so there may be no -e to name
        source = args.error_yaml or f"<{args.interface_yaml}>'s error model"
        raise ValueError(
            f"Error model <{source}> has <{number_channel}> channel(s), but "
            f"decoder <{decoders[0].algo}> decodes <{expected_channel}>. "
        )
    check_type = decoder_cfg.get("check_type", "hx")
    shape, _, _, _ = bundle.Hx_matrix.get_index()
    H_matrix = bundle.select(check_type)[3]

    algo_name = []
    num_max_iter = []
    for decoder in decoders:
        decoder.eval()
        algo_name.append(decoder.algo)
        num_max_iter.append(getattr(decoder, "num_max_iter", 0))

    H_file_name = bundle.get_H_file_name(check_type, number_channel)
    l_matrix = bundle.get_l_matrix(check_type, number_channel)
    check_num = bundle.get_check_num(check_type, number_channel)

    num_err = 0

    inner0 = getattr(decoders[0], "decoder", decoders[0])
    cap_on = getattr(inner0, "cap", None) is not None
    if cap_on and getattr(syndrome_generator, "rounds", 1) != 1:
        logger.warning("rebatch_speedup supports rounds==1 only; disabling the cap.")
        cap_on = False

    logger.success(
        "\n----------------------------------------------\nStep 5: Check checkpoint file\n----------------------------------------------"
    )

    if args.train:
        assert_trainable(decoders)
        num_batches = metrics.train_resume_checkpoint(
            decoders[-1],
            args.batch_size,
            args.train_checkpoint,
            args.checkpoint_yaml,
            decoder_device,
            error_model,
            loss,
            H_file_name,
        )
    else:
        metrics, num_err, num_batches = MetricState.resume_checkpoint(
            args.checkpoint_yaml,
            num_decoders,
            number_channel,
            decoder_device,
            args.batch_size,
            args.target_error,
            dtype,
            error_model.rate,
            H_file_name,
        )

    queue = ExtraQueue(
        args.batch_size, args.target_error, decoder_device
    )  # deferred (hard) samples

    max_batches = metrics.total_batches if args.train else float("inf")

    while num_batches < max_batches and (
        num_err <= args.target_error or (cap_on and queue.nonempty)
    ):
        logger.success(
            "\n----------------------------------------------\nStep 6: Generate error\n----------------------------------------------"
        )
        # training reads its loss straight off the decoder and skips Steps 9-11, so
        # there is nothing for it to buffer
        bt = (
            None
            if args.train
            else BatchTracker(
                num_decoders, number_channel, shape, dtype, decoder_device
            )
        )

        # predict_pct offload scheduler: decide WHEN to re-decode the deferred queue
        do_flush, flushing = queue.should_flush(num_err)
        use_extra = cap_on and do_flush
        if cap_on:
            inner0.cap_bypass = use_extra
        if use_extra:
            # re-decode a batch of the hardest deferred samples together (one batch at a time)
            error_dataloader, n = queue.take_batch()
            logger.info(
                f"Decoding <{n}> deferred samples from the extra queue "
                f'({"flush" if flushing else "predict_pct"}).'
            )
        else:
            # create error
            zero_qubits = torch.zeros([args.batch_size, shape[1]], dtype=dtype)
            error_vector, error_dataloader = error_model.inject_error(
                zero_qubits, args.batch_size
            )
            avg_error_rate = torch.mean(torch.sum(error_vector, -1) / shape[1])
            logger.info(f"Specified error rate <{error_model.rate}>.")
            logger.info(f"Generated error rate <{avg_error_rate}>.")
        num_batches += 1

        num_err_before_batch = num_err
        for err, llr, _ in error_dataloader:
            if bt is not None:
                bt.record_error(err)

            logger.success(
                "\n----------------------------------------------\nStep 7: Measure syndrome\n----------------------------------------------"
            )
            synd = syndrome_generator.measure_syndrome(err, decoders[0])

            llr0 = llr
            if hasattr(syndrome_generator, "adjust_llr0"):
                llr0 = syndrome_generator.adjust_llr0(llr0)

            io_dict = {"synd": synd, "llr0": llr0, "H_matrix": H_matrix}

            logger.success(
                "\n----------------------------------------------\nStep 8: Decode\n----------------------------------------------"
            )
            for decoder_idx in range(num_decoders):
                start_time = time.time()
                io_dict = decoders[decoder_idx](io_dict)

                if args.train and decoder_idx == num_decoders - 1:
                    terms = loss.terms(io_dict, err)
                    total = loss.combine(*terms)
                    if decoders[decoder_idx].training:
                        total.backward()
                        decoders[decoder_idx].optimizer.step()
                        decoders[decoder_idx].optimizer.zero_grad(set_to_none=True)
                    metrics.train_update_metric(
                        num_batches,
                        (total, *terms),
                        loss.class_error(io_dict, err),
                    )
                    break
                elapsed = time.time() - start_time
                if bt is not None:
                    bt.record_metric(decoder_idx, io_dict, elapsed)

            if args.train:
                continue  # Steps 9-11 check logical errors, which training does not

            cap_keep = None
            if cap_on and getattr(inner0, "cap_active_last", False):
                keep = bt.converge_all[1].flatten() > 0
                queue.defer(err, llr, ~keep)
                cap_keep = keep.to(bt.e_all.device)
                bt.keep_samples(cap_keep)

            logger.success(
                "\n----------------------------------------------\nStep 9: Check logical error rate\n----------------------------------------------"
            )

            has_obs_flips = (
                hasattr(syndrome_generator, "observable_flips")
                and syndrome_generator.observable_flips is not None
            )
            check_error = (
                syndrome_generator.observable_flips.to(dtype)
                if has_obs_flips
                else bt.e_all
            )
            if cap_keep is not None and has_obs_flips:
                check_error = check_error[cap_keep]

            check = [[] for _ in range(num_decoders)]
            for i in range(num_decoders):
                check[i] = logical_check.check(
                    bt.e_v_all[i], check_error, l_matrix, bt.converge_all[i + 1]
                )
            num_err += int(torch.sum(check[num_decoders - 1]))
            logger.info(
                f"number of errors at the current batch {num_err}/{args.target_error}"
            )

            # report and accumulate metrics
            logger.success(
                "\n----------------------------------------------\nStep 10: Aggregate metrics\n----------------------------------------------"
            )
            if number_channel == 1:
                bt.e_v_all = [
                    t.unsqueeze(1).expand(-1, number_channel, -1) for t in bt.e_v_all
                ]
                check = [t.unsqueeze(1).expand(-1, number_channel) for t in check]
            for i in range(num_decoders):
                batch_result = metrics.report_metric(
                    num_max_iter[i],
                    bt.e_all,
                    bt.e_v_all[i],
                    bt.iter_all[i],
                    bt.time_iter_all[i],
                    check[i],
                    bt.converge_all[i],
                    bt.converge_all[i + 1],
                    i,
                )
                metrics.update_metric(i, batch_result)

            if num_batches % 100 == 0:
                logger.success(
                    "\n----------------------------------------------\nStep 11: Save batch log\n----------------------------------------------"
                )
                all_metrics = metrics.get_all_metrics(num_batches, algo_name, decoders)
                metrics.save_metric(
                    all_metrics,
                    args.run_dir + "/",
                    args.batch_size,
                    args.target_error,
                    str(dtype),
                    error_model.rate,
                    num_batches,
                    num_err,
                    H_file_name,
                    check_num,
                )
                logger.success(f"Saved log to <{output_log}>.")
                logger.success(f"Saved metric results to <{args.run_dir}>.")

        if use_extra:  # measure density from the FIRST extra batch, then freeze
            queue.freeze_density(num_err - num_err_before_batch, n)

    if args.train:
        metrics.train_save_checkpoint()
        return

    logger.success(
        "\n----------------------------------------------\nStep 12: Save final log\n----------------------------------------------"
    )
    all_metrics = metrics.get_all_metrics(num_batches, algo_name, decoders)
    metrics.save_metric(
        all_metrics,
        args.run_dir + "/",
        args.batch_size,
        args.target_error,
        str(dtype),
        error_model.rate,
        num_batches,
        num_err,
        H_file_name,
        check_num,
        1,
    )
    logger.success(f"Saved log to <{output_log}>.")
    logger.success(f"Saved metric results to <{args.run_dir}>.")


if __name__ == "__main__":
    main()
