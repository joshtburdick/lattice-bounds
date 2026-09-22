"""
Counts of sets after zeroing out edges.
"""

from scipy import special


class EdgeZeroingCounter:
    """Gets counts of sets after edge zeroing.

    We assume that we pick a vertex, and first zero out all of its
    edges. This means that at any point, the set of input edges is
    a set of fully-connected vertices, plus one vertex which has
    some edges zeroed out.

    This class counts the number of possible input hypergraphs given
    this setup.
    """

    def __init__(self, n, k):
        """Constructor."""
        self.n = n
        self.k = k

        def num_edges(vertices, extra_edges):
            return special.comb(vertices, 2, exact=True) + extra_edges

        # Stores the counts for a given number of vertices and extra edges.
        # Key: (vertices, extra_edges), Value: count
        self.vertex_edge_counts = {0: (0, 0)}
        for v in range(k, n + 1):
            # We may just have a complete graph of v vertices.
            self.vertex_edge_counts[num_edges(v, 0)] = (v, 0)
            # Or we may have that, plus one vertex connected to some
            # of the `v` vertices. There need to be at least enough
            # edges for there to be at least one k-clique, though.
            for e in range(k - 1, n):
                self.vertex_edge_counts[num_edges(v, e)] = (v, e)
