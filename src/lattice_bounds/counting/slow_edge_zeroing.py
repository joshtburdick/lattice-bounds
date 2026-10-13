"""Slower way of counting clique sets.

This is slower, but hopefully is simpler. It's intended as a check
for the faster EdgeZeroing class.

Note: _extremely_ slow for, say, n=7, k=3, even without symmetries.
(This probably isn't a problem, though.)
"""

import itertools
import pdb

import numpy as np


class SlowEdgeZeroing:
    """Counts clique sets by iterating through all possible zeroing orders.

    We represent sets of cliques using integer bitmasks.
    (Although the focus isn't efficiency, this should suffice for comparing
    with the faster EdgeZeroing class for small cases like n=8, k=3).

    Args:
        n: number of vertices
        k: size of cliques
        include_symmetries: whether to include symmetries in the counts
            (default: False).

    Raises:
        ValueError: if include_symmetries is True, as this is not yet implemented.
    """

    def __init__(self, n, k, include_symmetries=False):
        self.n = n
        self.k = k
        self.include_symmetries = include_symmetries

        self.all_cliques = list(itertools.combinations(range(n), k))
        self.num_cliques = len(self.all_cliques)

        self.setup_edge_sets_by_rank()

    def setup_edge_sets_by_rank(self):
        """Computes sets of edges for each 'rank'.

        This uses the same set of edges at each rank.
        Starting from the empty set, we add edges with the goal of increasing
        the number of possible k-cliques as gradually as possible.

        Note that each edge is stored as a sorted tuple (a,b), with a < b.
        """
        edge_sets_by_rank = [set()]
        for num_vertices in range(self.k, self.n + 1):
            # Find the edges in the new clique.
            edges_in_new_clique = frozenset(
                list(itertools.combinations(range(num_vertices), 2))
            )
            # First, add just the clique.
            edge_sets_by_rank.append(edges_in_new_clique)
            # If we've used all the vertices, we're done.
            if num_vertices == self.n:
                break
            # If not: add a new vertex, connected to the first k-1 vertices (enough for it
            # to be part of a new clique); then keep adding edges incident to this vertex
            # (in order) until it's connected to all num_vertices vertices.
            for num_extra_edges in range(self.k - 1, num_vertices):
                new_edges = frozenset(
                    [tuple((i, num_vertices)) for i in range(num_extra_edges)]
                )
                edge_sets_by_rank.append(edges_in_new_clique | new_edges)
        self.edge_sets_by_rank = edge_sets_by_rank

    def count_clique_sets(self):
        """Counts sets of cliques by iteratively zeroing out edges.

        Returns:
            A dictionary mapping the rank to a Numpy array number of clique sets with that rank.
        """
        sets = np.arange(2**self.num_cliques)
        # Total number of cliques of each size, for each rank.
        total_counts = dict()
        for rank in range(len(self.edge_sets_by_rank) - 1, -1, -1):
            edge_set = self.edge_sets_by_rank[rank]
            if self.include_symmetries:
                edge_sets = self._get_permuted_edge_sets(edge_set)
            else:
                edge_sets = [edge_set]
            sets = self._filter_by_edge_sets(sets, edge_sets)
            total_counts[rank] = np.bincount(np.bitwise_count(sets))

        # Compute the number of clique sets with _exactly_ each rank.
        rank_count = {0: np.array([1])}
        for rank in range(1, len(self.edge_sets_by_rank)):
            rank_count[rank] = total_counts[rank] - np.pad(
                total_counts[rank - 1],
                (
                    0,
                    len(total_counts[rank]) - len(total_counts[rank - 1]),
                ),
            )
        return rank_count

    def _get_permuted_edge_sets(self, edge_set):
        """Returns unique edge sets from permuting vertices of `edge_set`."""
        permuted_edge_sets = set()
        for perm in itertools.permutations(range(self.n)):
            permuted_edge_set = frozenset(
                tuple(sorted((perm[i], perm[j]))) for i, j in edge_set
            )
            permuted_edge_sets.add(permuted_edge_set)
        return permuted_edge_sets

    def _get_possible_cliques_mask(self, edge_set):
        """Gets the bitmask of cliques that are supported by the given edge set."""
        cliques = 0
        for i, clique in enumerate(self.all_cliques):
            if all(edge in edge_set for edge in itertools.combinations(clique, 2)):
                cliques |= 1 << i
        return cliques

    def _filter_by_edge_sets(self, clique_sets, edge_sets):
        """Filters a set of clique sets by a set of edge sets.

        Args:
            clique_sets: a set of sets of cliques, represented by a Numpy array
                of integer bitmasks
            edge_sets: a set of edge sets
        Returns:
            A Numpy array of integer bitmasks, representing the set of clique sets
        """
        in_some_edge_set = np.full(clique_sets.shape[0], False)
        for edge_set in edge_sets:
            clique_filter = self._get_possible_cliques_mask(edge_set)
            in_some_edge_set = in_some_edge_set | (
                (clique_sets & clique_filter) == clique_sets
            )
        return clique_sets[in_some_edge_set]
