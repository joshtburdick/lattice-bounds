"""
Counts of sets after zeroing out edges or vertices.

We assume that we zero these out in some order, to obtain a series
of sets of cliques, which are decreasing in size. We number these
sets, calling the empty set "rank 0", one clique "rank 1", ...
up to some rank. (The maximum rank will differ when zeroing out
edges vs. vertices.)
"""

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


class VertexZeroing:
    """Getting counts of possible vertex sets, when zeroing vertices.

    We assume that we zero out vertices in the order n, n-1, ... k+1.
    """

    def __init__(self, n, k):
        """Constructor."""
        self.n = n
        self.k = k

        def num_vertices(v, zeroed_out):
            return v - zeroed_out

        # Stores the counts for a given number of vertices and extra edges.
        # Key: (vertices, extra_edges), Value: count
        self.vertex_edge_counts = {0: (0, 0)}
        for v in range(k, n + 1):
            # We may just have a complete graph of v vertices.
            self.vertex_edge_counts[num_vertices(v, 0)] = (v, 0)
            # Or we may have that, plus one vertex connected to some
            # of the `v` vertices. There need to be at least enough
            # edges for there to be at least one k-clique, though.
            for e in range(k - 1, n):
                self.vertex_edge_counts[num_vertices(v, e)] = (v, e)

    def num_sets(self, num_vertices):
        """Number of sets of cliques with up to num_vertices vertices."""
        return 2 ** special.comb(num_vertices, self.k, exact=True)

    def num_symmetries(self, num_vertices):
        """Number of ways a set of num_vertices can be chosen from the n vertices."""
        return int(special.comb(self.n, num_vertices, exact=True))


class EdgeZeroing:
    """Getting counts of possible edge sets, when zeroing edges.

    We need to zero out edges in a way that keeps as many cliques
    as possible. To do this, we first zero out edges incident to
    vertex `n`, then `n-1`, ... `k+1`. (For a given vertex, when
    it has fewer than k-1 input edges, we zero out all the other
    input edges, because that vertex can no longer be part of a clique.)

    This means that at any point, the set of input edges is a set
    of fully-connected vertices, plus one vertex which has some
    edges zeroed out.
    """

    def __init__(self, n, k):
        """Constructor."""
        self.n = n
        self.k = k

        def num_edges(vertices, extra_edges):
            return int(special.comb(vertices, 2, exact=True)) + extra_edges

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

    def num_sets(self, num_vertices, extra_edges):
        """Number of k-cliques in a graph with `num_vertices` and `extra_edges` extra edges."""
        num_sets_in_complete_graph = special.comb(num_vertices, self.k, exact=True)
        num_additional_sets = special.comb(num_vertices - k + 1, self.k - 1, exact=True)
        return num_sets_in_complete_graph + num_additional_sets

    def num_symmetries(self, num_vertices, extra_edges):
        """Number of ways to choose the edges incident to the extra vertex."""
        # first, we choose num_vertices vertices
        num_clique_choices = int(special.comb(self.n, num_vertices, exact=True))
        # next, we choose an additional vertex
        num_additional_vertex_choices = self.n - num_vertices
        # lastly, we pick some subset of the edges
        num_edge_choices = int(special.comb(num_vertices, extra_edges, exact=True))
        # the number of possible choices is the product of all of these
        return num_clique_choices * num_additional_vertex_choices * num_edge_choices
