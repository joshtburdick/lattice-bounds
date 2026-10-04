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

    @property
    def max_cliques(self):
        """Maximum number of cliques across all ranks."""
        return special.comb(self.n, self.k, exact=True)

    def num_sets_exact_rank(self):
        """Gets the number of sets of cliques with exact rank `rank`.
        Returns: a 1-D numpy array where the i-th entry is the number
        of sets of cliques with exact rank `i`.
        """
        num_sets_cumulative = np.array(
            [
                2 ** self.zeroing_strategy.num_sets(rank)
                for rank in range(self.zeroing_strategy.num_ranks)
            ],
            dtype=object,
        )
        num_sets_exact = np.diff(num_sets_cumulative, prepend=0)
        assert sum(num_sets_exact) == 2**self.max_cliques
        return num_sets_exact

    def num_sets_by_size(self):
        """Gets the number of sets of cliques by size, for each rank.

        Returns: a 2-D numpy array of shape (num_ranks, max_cliques + 1),
        where the (i, j)-th entry is the number of sets with rank _up to_ i
        and size j.
        """
        max_cliques = self.max_cliques
        num_ranks = self.zeroing_strategy.num_ranks
        num_sets_by_size = np.zeros((num_ranks, max_cliques + 1), dtype=object)
        for rank in range(num_ranks):
            num_cliques = self.zeroing_strategy.num_sets(rank)
            for size in range(num_cliques + 1):
                num_sets_by_size[rank, size] = special.comb(
                    num_cliques, size, exact=True
                )
        return num_sets_by_size

    def num_sets_by_size_exact_rank(self):
        """Like num_sets_by_size(), but only counts sets with exactly some rank.

        Returns: a 2-D numpy array of shape (num_ranks, max_cliques + 1),
        where the (i, j)-th entry is the number of sets with exact rank i
        and size j.

        ??? Possibly this should be a list of Numpy arrays, not a 2-D numpy
        array. (As the lower ranks of this will mostly be 0.)
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
        num_layers = len(layer_bounds) - 1
        layer_counts = np.zeros(
            (self.zeroing_strategy.num_ranks, num_layers), dtype=object
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
                ],
                dtype=object,
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
        for num_vertices in range(k, n):
            # We may just have a complete graph of num_vertices vertices.
            self.vertex_edge_counts.append((num_vertices, 0))
            # Or we may have that, plus one vertex connected to some
            # of the `num_vertices` vertices. There need to be at least enough
            # edges for there to be at least one k-clique, though.
            for num_edges in range(k - 1, num_vertices):
                self.vertex_edge_counts.append((num_vertices, num_edges))
        # Finally, the complete graph of n vertices.
        self.vertex_edge_counts.append((n, 0))
        self.num_ranks = len(self.vertex_edge_counts)

    def num_sets(self, rank):
        """Number of possible k-cliques in a graph of some rank."""
        if rank == 0:
            return 0
        num_vertices, extra_edges = self.vertex_edge_counts[rank]
        num_sets_in_complete_graph = special.comb(num_vertices, self.k, exact=True)
        if extra_edges == 0:
            num_additional_sets = 0
        else:
            num_additional_sets = special.comb(extra_edges, self.k - 1, exact=True)
        return num_sets_in_complete_graph + num_additional_sets
