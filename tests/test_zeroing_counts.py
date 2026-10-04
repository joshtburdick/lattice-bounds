"""
Tests for ZeroingCounts and plotting rank sizes.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from lattice_bounds.plot.rank_sizes import plot_rank_sizes
from lattice_bounds.zeroing_counts import ZeroingCounts


def test_vertex_zeroing_counts():
    """Tests exact and cumulative counts for vertex zeroing."""
    n = 6
    k = 3
    zc = ZeroingCounts(n, k, zeroing_type="vertex")
    max_cliques = special.comb(n, k, exact=True)

    exact_rank_counts = zc.num_sets_exact_rank()
    assert sum(exact_rank_counts) == 2**max_cliques

    exact_by_size = zc.num_sets_by_size_exact_rank()
    assert exact_by_size.shape == (zc.zeroing_strategy.num_ranks, max_cliques + 1)
    assert np.all(np.sum(exact_by_size, axis=1) == exact_rank_counts)

    # Sum across ranks for each size should equal comb(max_cliques, s)
    total_by_size = np.sum(exact_by_size, axis=0)
    expected_by_size = [
        special.comb(max_cliques, s, exact=True) for s in range(max_cliques + 1)
    ]
    assert np.all(total_by_size == expected_by_size)


def test_edge_zeroing_counts():
    """Tests exact and cumulative counts for edge zeroing."""
    n = 5
    k = 3
    zc = ZeroingCounts(n, k, zeroing_type="edge")
    max_cliques = special.comb(n, k, exact=True)

    exact_rank_counts = zc.num_sets_exact_rank()
    assert sum(exact_rank_counts) == 2**max_cliques

    exact_by_size = zc.num_sets_by_size_exact_rank()
    assert exact_by_size.shape == (zc.zeroing_strategy.num_ranks, max_cliques + 1)
    assert np.all(np.sum(exact_by_size, axis=1) == exact_rank_counts)


def test_layer_counts():
    """Tests layer counting logic."""
    zc = ZeroingCounts(6, 3, zeroing_type="vertex")
    layer_bounds = [0, 5, 10, 15, 21]
    layers = zc.get_layer_counts(layer_bounds)
    assert layers.shape == (zc.zeroing_strategy.num_ranks, len(layer_bounds) - 1)
    assert np.all(np.sum(layers, axis=1) == zc.num_sets_exact_rank())


def test_plot_rank_sizes():
    """Tests that plot_rank_sizes executes and returns an Axes object without error."""
    zc = ZeroingCounts(5, 3, zeroing_type="vertex")
    fig, ax = plt.subplots()
    returned_ax = plot_rank_sizes(zc, exact=True, log_scale=True, ax=ax)
    assert returned_ax is ax
    plt.close(fig)
