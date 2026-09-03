import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Get the directory containing the current Python file
current_path = Path(__file__).resolve().parent.parent.parent

# Add to sys.path (if not already there)
if str(current_path) not in sys.path:
    sys.path.insert(0, str(current_path))

from zoo.script.plot_utils import (
    load_results_dict,
    lookup_results_dict,
    tag_to_str,
)


def main():
    base_dir = 'zoo/speedup'
    dir_cpu             = f'{base_dir}/bposd_surface_sweeping_cpu'
    dir_gpu_amd_mi210   = f'{base_dir}/bposd_surface_sweeping_gpu_amd_mi210'
    dir_gpu_nv_a100     = f'{base_dir}/bposd_surface_sweeping_gpu_nv_a100'
    dir_gpu_nv_h200     = f'{base_dir}/bposd_surface_sweeping_gpu_nv_h200'

    dir_gpu_amd_mi210_bs   = f'{base_dir}/bposd_surface_sweeping_gpu_amd_mi210_bs'
    dir_gpu_nv_a100_bs     = f'{base_dir}/bposd_surface_sweeping_gpu_nv_a100_bs'
    dir_gpu_nv_h200_bs     = f'{base_dir}/bposd_surface_sweeping_gpu_nv_h200_bs'

    results_dict_cpu            = load_results_dict(dir_cpu)
    results_dict_gpu_amd_mi210  = load_results_dict(dir_gpu_amd_mi210)
    results_dict_gpu_nv_a100    = load_results_dict(dir_gpu_nv_a100)
    results_dict_gpu_nv_h200    = load_results_dict(dir_gpu_nv_h200)

    results_dict_gpu_amd_mi210_bs  = load_results_dict(dir_gpu_amd_mi210_bs)
    results_dict_gpu_nv_a100_bs    = load_results_dict(dir_gpu_nv_a100_bs)
    results_dict_gpu_nv_h200_bs    = load_results_dict(dir_gpu_nv_h200_bs)

    def plot_gpu_compare(ax, dtype='float64', metric='time'):
        # fix the physical error rate and compare different gpus
        tag_shared = [dtype, '11', 'hx', 'surface']
        tags = []

        if metric == 'time':
            full_decoding_metric = ['decoder_full', 'total time (s)']
        elif metric == 'accuracy':
            full_decoding_metric = ['decoder_full', 'logical error rate']

        # X-axis values
        x_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

        for err in x_ticks:
            tags.append(tag_to_str(tag_shared + [str(err)]))

        amd_mi210 = []
        for tag in tags:
            amd_mi210.append(lookup_results_dict(results_dict_gpu_amd_mi210,   full_decoding_metric + [tag]))

        nv_a100 = []
        for tag in tags:
            nv_a100.append(lookup_results_dict(results_dict_gpu_nv_a100,   full_decoding_metric + [tag]))

        nv_h200 = []
        for tag in tags:
            nv_h200.append(lookup_results_dict(results_dict_gpu_nv_h200,   full_decoding_metric + [tag]))

        colors = plt.get_cmap("tab10").colors[:3]
        markers = ['^', 'o', 's']

        color_labels = ['FP64', 'FP32', 'FP16']
        marker_labels = ['MI210', 'A100', 'H200']

        color_proxies = [Line2D([0], [0], color=c, lw=1) for c in colors]
        marker_proxies = [Line2D([0], [0], color='black', lw=0, marker=m, markerfacecolor='black', linestyle='none', markersize=4) for m in markers]

        # Combine labels
        legend_proxies = color_proxies + marker_proxies
        legend_labels = color_labels + marker_labels

        if dtype == 'float64':
            colors_here = colors[0]
        elif dtype == 'float32':
            colors_here = colors[1]
        elif dtype == 'float16':
            colors_here = colors[2]

        if dtype == 'float64':
            label_amd_mi210 = 'MI210'
            label_nv_a100   = 'A100'
            label_nv_h200   = 'H200'
        else:
            label_amd_mi210 = None
            label_nv_a100   = None
            label_nv_h200   = None

        if dtype == 'float16':
            ax.legend(legend_proxies, legend_labels,
                        loc='best',
                        ncol=3,
                        handletextpad=0.1,  # Reduce space between marker and label
                        borderpad=0.25,      # Reduce space inside legend border
                        columnspacing=0.1,  # Reduce space between columns if multi-column
                        labelspacing=0.1,
                        handlelength=1)

        ax.plot(x_ticks, amd_mi210, marker=markers[0], label=label_amd_mi210, color=colors_here, markersize=4)
        ax.plot(x_ticks, nv_a100, marker=markers[1], label=label_nv_a100, color=colors_here, markersize=4)
        ax.plot(x_ticks, nv_h200, marker=markers[2], label=label_nv_h200, color=colors_here, markersize=4)

        # Axis labels and scale
        ax.set_xlabel("Physical error rate")
        if metric == 'time':
            ax.set_ylabel("Runtime (s)")
        elif metric == 'accuracy':
            ax.set_ylabel("Logical error rate")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])


    def plot_data_format_compare(ax, error_rate='0.02', metric='time'):
        # fig to compare different gpus
        tag_shared = [error_rate, '11', 'hx', 'surface']
        tags = []

        if metric == 'time':
            full_decoding_metric = ['decoder_full', 'total time (s)']
        elif metric == 'accuracy':
            full_decoding_metric = ['decoder_full', 'logical error rate']

        # X-axis values
        x_ticks = [16, 32, 64]

        for err in x_ticks:
            tags.append(tag_to_str(tag_shared + [f'float{err}']))

        amd_mi210 = []
        for tag in tags:
            amd_mi210.append(lookup_results_dict(results_dict_gpu_amd_mi210,   full_decoding_metric + [tag]))

        nv_a100 = []
        for tag in tags:
            nv_a100.append(lookup_results_dict(results_dict_gpu_nv_a100,   full_decoding_metric + [tag]))

        nv_h200 = []
        for tag in tags:
            nv_h200.append(lookup_results_dict(results_dict_gpu_nv_h200,   full_decoding_metric + [tag]))

        colors = plt.get_cmap("tab10").colors[:3]
        markers = ['^', 'o', 's']

        color_labels = ['0.02', '0.05', '0.1']
        marker_labels = ['MI210', 'A100', 'H200']

        color_proxies = [Line2D([0], [0], color=c, lw=1) for c in colors]
        marker_proxies = [Line2D([0], [0], color='black', lw=0, marker=m, markerfacecolor='black', linestyle='none', markersize=4) for m in markers]

        # Combine labels
        legend_proxies = color_proxies + marker_proxies
        legend_labels = color_labels + marker_labels

        if error_rate == '0.02':
            colors_here = colors[0]
        elif error_rate == '0.05':
            colors_here = colors[1]
        elif error_rate == '0.1':
            colors_here = colors[2]

        if error_rate == '0.02':
            label_amd_mi210 = 'MI210'
            label_nv_a100   = 'A100'
            label_nv_h200   = 'H200'
        else:
            label_amd_mi210 = None
            label_nv_a100   = None
            label_nv_h200   = None

        if error_rate == '0.02':
            ax.legend(legend_proxies, legend_labels,
                        loc='best',
                        ncol=3,
                        handletextpad=0.1,  # Reduce space between marker and label
                        borderpad=0.25,      # Reduce space inside legend border
                        columnspacing=0.1,  # Reduce space between columns if multi-column
                        labelspacing=0.1,
                        handlelength=1)

        ax.plot(x_ticks, amd_mi210, marker=markers[0], label=label_amd_mi210, color=colors_here, markersize=4)
        ax.plot(x_ticks, nv_a100, marker=markers[1], label=label_nv_a100, color=colors_here, markersize=4)
        ax.plot(x_ticks, nv_h200, marker=markers[2], label=label_nv_h200, color=colors_here, markersize=4)

        # Axis labels and scale
        ax.set_xlabel("Data format")
        if metric == 'time':
            ax.set_ylabel("Runtime (s)")
        elif metric == 'accuracy':
            ax.set_ylabel("Logical error rate")
        # ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'FP{str(x)}' for x in x_ticks])


    def plot_distance_compare(ax, distance='11', metric='time'):
        # fix gpu and data format and compare different distances
        tag_shared = [distance, 'float64', 'hx', 'surface']
        tags = []

        if metric == 'time':
            full_decoding_metric = ['decoder_full', 'total time (s)']
        elif metric == 'accuracy':
            full_decoding_metric = ['decoder_full', 'logical error rate']

        # X-axis values
        x_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

        for err in x_ticks:
            tags.append(tag_to_str(tag_shared + [str(err)]))

        amd_mi210 = []
        for tag in tags:
            amd_mi210.append(lookup_results_dict(results_dict_gpu_amd_mi210,   full_decoding_metric + [tag]))

        nv_a100 = []
        for tag in tags:
            nv_a100.append(lookup_results_dict(results_dict_gpu_nv_a100,   full_decoding_metric + [tag]))

        nv_h200 = []
        for tag in tags:
            nv_h200.append(lookup_results_dict(results_dict_gpu_nv_h200,   full_decoding_metric + [tag]))

        colors = plt.get_cmap("tab10").colors[:3]
        markers = ['^', 'o', 's']

        color_labels = ['3', '7', '11']
        marker_labels = ['MI210', 'A100', 'H200']

        color_proxies = [Line2D([0], [0], color=c, lw=1) for c in colors]
        marker_proxies = [Line2D([0], [0], color='black', lw=0, marker=m, markerfacecolor='black', linestyle='none', markersize=4) for m in markers]

        # Combine labels
        legend_proxies = color_proxies + marker_proxies
        legend_labels = color_labels + marker_labels

        if distance == '3':
            colors_here = colors[0]
        elif distance == '7':
            colors_here = colors[1]
        elif distance == '11':
            colors_here = colors[2]

        if distance == '3':
            ax.legend(legend_proxies, legend_labels,
                        ncol=3,
                        loc='best',
                        handletextpad=0.1,  # Reduce space between marker and label
                        borderpad=0.25,      # Reduce space inside legend border
                        columnspacing=0.1,  # Reduce space between columns if multi-column
                        labelspacing=0.1,
                        handlelength=1)

        ax.plot(x_ticks, amd_mi210, marker=markers[0], color=colors_here, markersize=4)
        ax.plot(x_ticks, nv_a100, marker=markers[1], color=colors_here, markersize=4)
        ax.plot(x_ticks, nv_h200, marker=markers[2], color=colors_here, markersize=4)

        # Axis labels and scale
        ax.set_xlabel("Physical error rate")
        if metric == 'time':
            ax.set_ylabel("Runtime (s)")
        elif metric == 'accuracy':
            ax.set_ylabel("Logical error rate")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])


    def plot_cpu_speedup_compare(ax, distance='11'):
        # fix gpu and data format and compare different distances
        tag_shared = [distance, 'float64', 'hx', 'surface']
        tags = []

        full_decoding_metric = ['decoder_full', 'total time (s)']

        # X-axis values
        x_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

        for err in x_ticks:
            tags.append(tag_to_str(tag_shared + [str(err)]))

        amd_mi210 = []
        for tag in tags:
            cpu_time = lookup_results_dict(results_dict_cpu,   full_decoding_metric + [tag])
            gpu_time = lookup_results_dict(results_dict_gpu_amd_mi210,   full_decoding_metric + [tag])
            if (cpu_time is not None) and (gpu_time is not None):
                speedup = cpu_time / gpu_time
                amd_mi210.append(speedup)
            else:
                amd_mi210.append(None)

        nv_a100 = []
        for tag in tags:
            cpu_time = lookup_results_dict(results_dict_cpu,   full_decoding_metric + [tag])
            gpu_time = lookup_results_dict(results_dict_gpu_nv_a100,   full_decoding_metric + [tag])
            if (cpu_time is not None) and (gpu_time is not None):
                speedup = cpu_time / gpu_time
                nv_a100.append(speedup)
            else:
                nv_a100.append(None)

        nv_h200 = []
        for tag in tags:
            cpu_time = lookup_results_dict(results_dict_cpu,   full_decoding_metric + [tag])
            gpu_time = lookup_results_dict(results_dict_gpu_nv_h200,   full_decoding_metric + [tag])
            if (cpu_time is not None) and (gpu_time is not None):
                speedup = cpu_time / gpu_time
                nv_h200.append(speedup)
            else:
                nv_h200.append(None)

        colors = plt.get_cmap("tab10").colors[:3]
        markers = ['^', 'o', 's']

        color_labels = ['3', '7', '11']
        marker_labels = ['MI210', 'A100', 'H200']

        color_proxies = [Line2D([0], [0], color=c, lw=1) for c in colors]
        marker_proxies = [Line2D([0], [0], color='black', lw=0, marker=m, markerfacecolor='black', linestyle='none', markersize=4) for m in markers]

        # Combine labels
        legend_proxies = color_proxies + marker_proxies
        legend_labels = color_labels + marker_labels

        if distance == '3':
            colors_here = colors[0]
        elif distance == '7':
            colors_here = colors[1]
        elif distance == '11':
            colors_here = colors[2]

        if distance == '3':
            ax.legend(legend_proxies, legend_labels,
                        ncol=3,
                        loc='best',
                        handletextpad=0.1,  # Reduce space between marker and label
                        borderpad=0.25,      # Reduce space inside legend border
                        columnspacing=0.1,  # Reduce space between columns if multi-column
                        labelspacing=0.1,
                        handlelength=1)

        ax.plot(x_ticks, amd_mi210, marker=markers[0], color=colors_here, markersize=4)
        ax.plot(x_ticks, nv_a100, marker=markers[1], color=colors_here, markersize=4)
        ax.plot(x_ticks, nv_h200, marker=markers[2], color=colors_here, markersize=4)

        # Axis labels and scale
        ax.set_xlabel("Physical error rate")
        ax.set_ylabel("Speedup over CPU")
        ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])


    def plot_batch_compare(ax):
        # fix gpu and data format and compare different distances
        tag_shared = ['0.5', '11', 'float64', 'hx', 'surface']
        tags = []

        # 2^7 ~ 2^20
        start_idx = 7
        end_idx = 20
        for i in range(end_idx + 1 - start_idx):
            tags.append(tag_to_str(tag_shared + [f'{2**(7+i)}']))

        full_decoding_time = ['decoder_full', 'total time (s)']
        full_decoding_bs = ['decoder_full', 'batch size']
        full_decoding_bc = ['decoder_full', 'batch count']

        # X-axis values
        x_ticks = [2**i for i in range(start_idx, end_idx + 1)]

        amd_mi210 = []
        for tag in tags:
            runtime = lookup_results_dict(results_dict_gpu_amd_mi210_bs,   full_decoding_time + [tag])
            bs = lookup_results_dict(results_dict_gpu_amd_mi210_bs,   full_decoding_bs + [tag])
            bc = lookup_results_dict(results_dict_gpu_amd_mi210_bs,   full_decoding_bc + [tag])
            if runtime is not None:
                amd_mi210.append(runtime / (bs * bc))
            else:
                amd_mi210.append(None)

        nv_a100 = []
        for tag in tags:
            runtime = lookup_results_dict(results_dict_gpu_nv_a100_bs,   full_decoding_time + [tag])
            bs = lookup_results_dict(results_dict_gpu_nv_a100_bs,   full_decoding_bs + [tag])
            bc = lookup_results_dict(results_dict_gpu_nv_a100_bs,   full_decoding_bc + [tag])
            if runtime is not None:
                nv_a100.append(runtime / (bs * bc))
            else:
                nv_a100.append(None)

        nv_h200 = []
        for tag in tags:
            runtime = lookup_results_dict(results_dict_gpu_nv_h200_bs,   full_decoding_time + [tag])
            bs = lookup_results_dict(results_dict_gpu_nv_h200_bs,   full_decoding_bs + [tag])
            bc = lookup_results_dict(results_dict_gpu_nv_h200_bs,   full_decoding_bc + [tag])
            if runtime is not None:
                nv_h200.append(runtime / (bs * bc))
            else:
                nv_h200.append(None)

        markers = ['^', 'o', 's']

        marker_labels = ['MI210', 'A100', 'H200']

        marker_proxies = [Line2D([0], [0], color='black', lw=0, marker=m, markerfacecolor='black', linestyle='none', markersize=4) for m in markers]

        # Combine labels
        legend_proxies = marker_proxies
        legend_labels = marker_labels

        colors_here = 'black'

        ax.legend(legend_proxies, legend_labels,
                    loc='best',
                    ncol=3,
                    handletextpad=0.1,  # Reduce space between marker and label
                    borderpad=0.25,      # Reduce space inside legend border
                    columnspacing=0.1,  # Reduce space between columns if multi-column
                    labelspacing=0.1,
                    handlelength=1)

        ax.plot(x_ticks, amd_mi210, marker=markers[0], color=colors_here, markersize=4)
        ax.plot(x_ticks, nv_a100, marker=markers[1], color=colors_here, markersize=4)
        ax.plot(x_ticks, nv_h200, marker=markers[2], color=colors_here, markersize=4)

        # Axis labels and scale
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Runtime per input (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Select specific x-ticks to label
        tick_locs = [2**i for i in range(start_idx, end_idx + 1, 3)]
        tick_labels = [str(fr"$2^{{{i}}}$") for i in range(start_idx, end_idx + 1, 3)]

        # Set the desired ticks and labels
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)


    # plot

    FIGWIDHT = 3.33
    FIGHEIGHT = 2

    # Plot gpu
    fig, ax = plt.subplots(figsize=(FIGWIDHT, FIGHEIGHT))
    metric = 'time'
    plot_gpu_compare(ax, 'float64', metric)
    plot_gpu_compare(ax, 'float32', metric)
    plot_gpu_compare(ax, 'float16', metric)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{base_dir}/{metric}_gpu.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_dir}/{metric}_gpu.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(FIGWIDHT, FIGHEIGHT))
    metric = 'accuracy'
    plot_gpu_compare(ax, 'float64', metric)
    plot_gpu_compare(ax, 'float32', metric)
    plot_gpu_compare(ax, 'float16', metric)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{base_dir}/{metric}_gpu.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_dir}/{metric}_gpu.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


    # Plot data format
    fig, ax = plt.subplots(figsize=(FIGWIDHT, FIGHEIGHT))
    metric = 'time'
    plot_data_format_compare(ax, '0.02', metric)
    plot_data_format_compare(ax, '0.05', metric)
    plot_data_format_compare(ax, '0.1', metric)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{base_dir}/{metric}_data_format.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_dir}/{metric}_data_format.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(FIGWIDHT, FIGHEIGHT))
    metric = 'accuracy'
    plot_data_format_compare(ax, '0.02', metric)
    plot_data_format_compare(ax, '0.05', metric)
    plot_data_format_compare(ax, '0.1', metric)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{base_dir}/{metric}_data_format.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_dir}/{metric}_data_format.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


    # Plot distance
    fig, ax = plt.subplots(figsize=(FIGWIDHT, FIGHEIGHT))
    metric = 'time'
    plot_distance_compare(ax, '3', metric)
    plot_distance_compare(ax, '7', metric)
    plot_distance_compare(ax, '11', metric)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{base_dir}/{metric}_distance.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_dir}/{metric}_distance.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(FIGWIDHT, FIGHEIGHT))
    metric = 'accuracy'
    plot_distance_compare(ax, '3', metric)
    plot_distance_compare(ax, '7', metric)
    plot_distance_compare(ax, '11', metric)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{base_dir}/{metric}_distance.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_dir}/{metric}_distance.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


    # Plot cpu_speedup
    fig, ax = plt.subplots(figsize=(FIGWIDHT, FIGHEIGHT))
    metric = 'time'
    plot_cpu_speedup_compare(ax, '3')
    plot_cpu_speedup_compare(ax, '7')
    plot_cpu_speedup_compare(ax, '11')
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{base_dir}/{metric}_cpu_speedup.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_dir}/{metric}_cpu_speedup.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


    # Plot batch
    fig, ax = plt.subplots(figsize=(FIGWIDHT, FIGHEIGHT))
    metric = 'time'
    plot_batch_compare(ax)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{base_dir}/{metric}_batch.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_dir}/{metric}_batch.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()

