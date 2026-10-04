#!/usr/bin/env python3
"""
Plots the distribution of functions across ranks and number of cliques (sizes).

For a given setup (n vertices, k-cliques), at each rank of the zeroing process,
hypergraphs (boolean functions) have varying numbers of cliques. This module
plots the number of functions having each clique count, broken down by rank.
"""

import argparse
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from lattice_bounds.zeroing_counts import ZeroingCounts


def get_rank_label(zc: ZeroingCounts, rank: int) -> str:
    """Returns a descriptive label for a given rank."""
    strategy = zc.zeroing_strategy
    num_cliques = strategy.num_sets(rank)
    if hasattr(strategy, "num_vertices"):
        num_v = strategy.num_vertices[rank]
        return f"Rank {rank} (v={num_v}, {num_cliques} max cliques)"
    elif hasattr(strategy, "vertex_edge_counts"):
        v, e = strategy.vertex_edge_counts[rank]
        return f"Rank {rank} (v={v}, extra_e={e}, {num_cliques} max cliques)"
    return f"Rank {rank} ({num_cliques} max cliques)"


def plot_rank_sizes(
    zc: ZeroingCounts,
    exact: bool = True,
    log_scale: bool = True,
    ax: Optional[plt.Axes] = None,
    show_zero_points: bool = False,
) -> plt.Axes:
    """Plots number of functions vs. number of cliques for each rank.

    Args:
        zc: ZeroingCounts instance for (n, k, zeroing_type).
        exact: If True, plots exact rank counts. If False, plots cumulative counts.
        log_scale: If True, uses log scale on the y-axis.
        ax: Optional matplotlib Axes to plot on. If None, creates a new figure/axis.
        show_zero_points: If True, includes zero values (relevant for linear scale).

    Returns:
        The matplotlib Axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    num_ranks = zc.zeroing_strategy.num_ranks
    max_cliques = zc.max_cliques
    clique_sizes = np.arange(max_cliques + 1)

    if exact:
        counts_matrix = zc.num_sets_by_size_exact_rank()
        title_type = "Exact Rank"
    else:
        counts_matrix = zc.num_sets_by_size()
        title_type = "Cumulative Rank (Rank $\\le r$)"

    # Use a visually distinct color palette across ranks
    cmap = plt.get_cmap("tab10" if num_ranks <= 10 else "viridis")
    colors = [cmap(i / max(1, num_ranks - 1)) if num_ranks > 10 else cmap(i) for i in range(num_ranks)]

    for rank in range(num_ranks):
        counts = np.array(counts_matrix[rank], dtype=float)
        label = get_rank_label(zc, rank)

        if log_scale:
            # Mask 0 or negative values to avoid math domain errors on log scale
            valid = counts > 0
            x_vals = clique_sizes[valid]
            y_vals = counts[valid]
            if len(x_vals) > 0:
                ax.plot(
                    x_vals,
                    y_vals,
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    alpha=0.85,
                    color=colors[rank],
                    label=label,
                )
        else:
            if not show_zero_points:
                # Truncate trailing zeros beyond the max possible cliques for this rank
                max_rank_cliques = zc.zeroing_strategy.num_sets(rank)
                valid = clique_sizes <= max_rank_cliques
                x_vals = clique_sizes[valid]
                y_vals = counts[valid]
            else:
                x_vals = clique_sizes
                y_vals = counts

            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                markersize=5,
                linewidth=1.8,
                alpha=0.85,
                color=colors[rank],
                label=label,
            )

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Number of Functions (log scale)")
    else:
        ax.set_ylabel("Number of Functions")

    ax.set_xlabel("Number of Cliques ($s$)")
    ax.set_title(
        f"Functions by Clique Count and {title_type}\n"
        f"($n={zc.n}$, $k={zc.k}$, total cliques={max_cliques}, strategy={zc.zeroing_type})"
    )

    ax.set_xlim(-0.5, max_cliques + 0.5)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        framealpha=0.9,
        fontsize=8 if num_ranks > 15 else 9,
    )

    return ax


def main():
    """Command-line runner for plotting rank sizes."""
    parser = argparse.ArgumentParser(
        description="Plot function counts by rank and number of cliques."
    )
    parser.add_argument("-n", type=int, default=6, help="Number of vertices (default: 6)")
    parser.add_argument("-k", type=int, default=3, help="Clique size (default: 3)")
    parser.add_argument(
        "--zeroing-type",
        choices=["vertex", "edge"],
        default="vertex",
        help="Zeroing strategy (default: vertex)",
    )
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="Plot cumulative ranks instead of exact ranks",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear y-axis scale instead of log scale",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="rank_sizes.png",
        help="Output file path (e.g. rank_sizes.png or rank_sizes.pdf)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively",
    )

    args = parser.parse_args()

    zc = ZeroingCounts(args.n, args.k, zeroing_type=args.zeroing_type)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    plot_rank_sizes(
        zc,
        exact=not args.cumulative,
        log_scale=not args.linear,
        ax=ax,
    )
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, bbox_inches="tight")
        print(f"Saved plot to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
