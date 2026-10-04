"""
Counts of sets after zeroing out edges or vertices.

We assume that we zero these out in some order, to obtain a series
of sets of cliques, which are decreasing in size. We number these
sets, calling the empty set "rank 0", one clique "rank 1", ...
up to some rank. (The maximum rank will differ when zeroing out
edges vs. vertices.)

For now, we ignore symmetries. This limits the utility of
the bounds, but it keeps the logic simpler.
"""

import numpy as np
from scipy import special


class ZeroingCounts:
    """Gets counts of sets after zeroing.

    This class counts the number of possible input hypergraphs given
    this setup.
    """

    def __init__(self, n, k, zeroing_type="vertex"):
        """Constructor."""
        self.n = n
        self.k = k
        self.zeroing_type = zeroing_type

        if self.zeroing_type == "vertex":
            self.zeroing_strategy = VertexZeroing(n, k)
        elif self.zeroing_type == "edge":
            self.zeroing_strategy = EdgeZeroing(n, k)
        else:
            raise ValueError("Invalid zeroing type.")

    def num_sets_exact_rank(self):
        """Gets the number of sets of cliques with exact rank `rank`.
        Returns: a 1-D numpy array where the i-th entry is the number
        of sets of cliques with exact rank `i`.
        """
        # first, compute number of sets of cliques, including symmetries
        num_sets = np.array(
            [
                self.zeroing_strategy.num_sets(rank)
                * self.zeroing_strategy.num_symmetries(rank)
                for rank in range(self.zeroing_strategy.num_ranks)
            ]
        )
        num_sets_exact = np.diff(num_sets, prepend=0)
        assert sum(num_sets_exact) == 2 ** special.comb(self.n, self.k, exact=True)
        return num_sets_exact

    def num_sets_by_size(self):
        """Gets the number of sets of cliques by size, for each rank.

        Returns: a list of numpy arrays, where the i-th array has length
        equal to the number of possible sizes for rank i, and the j-th entry
        is the number of sets with rank _up to_ i and size j.
        """
        all_num_sets_by_size = []
        for rank in range(self.zeroing_strategy.num_ranks):
            num_sets = self.zeroing_strategy.num_sets(rank)
            num_sets_by_size = np.array(
                [
                    special.comb(self.zeroing_strategy.num_sets(rank), s, exact=True)
                    for s in range(self.zeroing_strategy.num_sets(rank) + 1)
                ]
            )
            # There's only one empty set of cliques, so we manually set the
            # number of sets with size 0 to 1.
            num_sets_by_size[0] = 1
            all_num_sets_by_size.append(num_sets_by_size)
        return all_num_sets_by_size

    def num_sets_by_size_exact_rank(self):
        """Like num_sets_by_size(), but only counts sets with exactly some rank.

        Returns: a list of numpy arrays, where the i-th array has length
        equal to the number of possible sizes for rank i, and the j-th entry
        is the number of sets with rank i and size j.
        """
        num_sets_by_size_cumulative = self.num_sets_by_size()
        num_sets_by_size_exact_rank = np.diff(
            num_sets_by_size_cumulative, axis=0, prepend=0
        )
        assert np.all(
            np.sum(num_sets_by_size_exact_rank, axis=1) == self.num_sets_exact_rank()
        )
        return num_sets_by_size_exact_rank

    def get_layer_counts(self, layer_bounds):
        """Gets the number of sets at each layer, given the layer bounds.

        layer_bounds: A sorted list of integers, where the ith layer is given by
            `layer_bounds[i]` <= number of cliques < `layer_bounds[i+1]`.
        Returns: a 2-D NumPy array of shape (num_ranks, num_layers),
            where entry (i, j) is the number of sets of cliques with
            exact rank i, in layer j.
        """
        num_sets_by_size_exact_rank = self.num_sets_by_size_exact_rank()
        num_layers = len(layer_bounds)
        layer_counts = np.zeros(
            (self.zeroing_strategy.num_ranks, num_layers), dtype=np.object
        )
        for rank in range(self.zeroing_strategy.num_ranks):
            layer_counts[rank] = np.array(
                [
                    sum(
                        num_sets_by_size_exact_rank[rank][
                            layer_bounds[j] : layer_bounds[j + 1]
                        ]
                    )
                    for j in range(num_layers)
                ]
            )
        assert np.all(np.sum(layer_counts, axis=1) == self.num_sets_exact_rank())
        return layer_counts


class VertexZeroing:
    """Getting counts of possible vertex sets, when zeroing vertices.

    We assume that we zero out vertices in the order n, n-1, ... k+1.
    """

    def __init__(self, n, k):
        """Constructor."""
        self.n = n
        self.k = k

        # We consider the "size" to be the number of vertices
        # present in the set (counting 0 as the empty set of vertices).
        self.num_vertices = [0] + list(range(k, n + 1))
        self.num_ranks = len(self.num_vertices)

    def num_sets(self, rank):
        """Number of sets with rank `rank` (unique up to symmetry)."""
        # The 0-th rank corresponds to the empty set of vertices.
        if rank == 0:
            return 0
        return special.comb(self.num_vertices[rank], self.k, exact=True)


class EdgeZeroing:
    """Getting counts of possible edge sets, when zeroing edges.

    We need to zero out edges in a way that keeps as many cliques
    as possible. To do this, we first zero out edges incident to
    vertex `n`, then `n-1`, ... `k+1`. (For a given vertex, when
    it has fewer than k-1 input edges, we zero out all the other
    input edges, because that vertex can no longer be part of a clique.)

    This means that at any point, the set of input edges is a set
    of fully-connected vertices, plus one vertex which has _some_
    edges zeroed out.
    """

    def __init__(self, n, k):
        """Constructor."""
        self.n = n
        self.k = k

        # Stores the counts for a given number of vertices and extra edges.
        self.vertex_edge_counts = [(0, 0)]
        for num_vertices in range(k, n + 1):
            # We may just have a complete graph of v vertices.
            self.vertex_edge_counts.append((num_vertices, 0))
            # Or we may have that, plus one vertex connected to some
            # of the `v` vertices. There need to be at least enough
            # edges for there to be at least one k-clique, though.
            for e in range(k - 1, n):
                self.vertex_edge_counts.append((v, e))

    def num_sets(self, rank):
        """Number of possible k-cliques in a graph with `num_vertices` and `extra_edges` extra edges."""
        num_vertices, extra_edges = self.vertex_edge_counts[rank]
        num_sets_in_complete_graph = special.comb(num_vertices, self.k, exact=True)
        num_additional_sets = special.comb(num_vertices - k + 1, self.k - 1, exact=True)
        return num_sets_in_complete_graph + num_additional_sets
