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

    def get_layer_counts(self, layer_bounds):
        """Gets the number of sets at each layer, given the layer bounds.

        layer_bounds: A list of integers, where the ith layer is given by
            `layer_bounds[i]` <= number of cliques < `layer_bounds[i+1]`.
        Returns: a 2-D NumPy array of shape (num_ranks, num_layers),
            where entry (i, j) is the number of sets of cliques with
            exact rank i, in layer j.
        """
        pass


class VertexZeroing:
    """Getting counts of possible vertex sets, when zeroing vertices.

    We assume that we zero out vertices in the order n, n-1, ... k+1.
    """

    def __init__(self, n, k):
        """Constructor."""
        self.n = n
        self.k = k

        # We consider the "size" to be the number of vertices
        # present in the set.
        self.num_vertices = list(range(k, n + 1))

    def size(rank):
        """Size of each rank (in this case, number of vertices)."""
        return self.num_vertices[rank]

    def num_sets(rank):
        """Number of sets with rank `rank` (unique up to symmetry)."""
        return special.comb(self.num_vertices[rank], k, exact=True)

    def num_symmetries(rank):
        """Number of symmetries for rank `rank`."""
        return special.comb(self.n, self.num_vertices[rank], exact=True)


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

    def size(self, rank):
        """Size of each rank (in this case, number of vertices)."""
        num_vertices, extra_edges = self.vertex_edge_counts[rank]
        return int(special.comb(num_vertices, 2, exact=True)) + extra_edges

    def num_sets(self, rank):
        """Number of possible k-cliques in a graph with `num_vertices` and `extra_edges` extra edges."""
        num_vertices, extra_edges = self.vertex_edge_counts[rank]
        num_sets_in_complete_graph = special.comb(num_vertices, self.k, exact=True)
        num_additional_sets = special.comb(num_vertices - k + 1, self.k - 1, exact=True)
        return num_sets_in_complete_graph + num_additional_sets

    def num_symmetries(self, rank):
        """Number of ways to choose the edges incident to the extra vertex."""
        num_vertices, extra_edges = self.vertex_edge_counts[rank]
        # first, we choose num_vertices vertices
        num_clique_choices = int(special.comb(self.n, num_vertices, exact=True))
        # next, we choose an additional vertex
        num_additional_vertex_choices = self.n - num_vertices
        # lastly, we pick some subset of the edges
        num_edge_choices = int(special.comb(vertices, extra_edges, exact=True))
        # the number of possible choices is the product of all of these
        return num_clique_choices * num_additional_vertex_choices * num_edge_choices
