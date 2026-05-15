"""
Defines the lattice of zeroing ("zonking") of hypergraphs.
"""

import itertools


def powerset(iterable):
    """
    Returns the powerset of an iterable.
    """
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


class CliqueLattice:
    """Represents the lattice of zeroing of hypergraphs.

    We consider the set of all k-cliques on {0, ..., n-1}.
    """

    def __init__(self, n, k):
        """
        Args:
            n: number of vertices
            k: size of cliques
        """
        self.n = n
        self.k = k
        # the set of all cliques
        self.cliques = frozenset(itertools.combinations(range(n), k))
        # the set of all hypergraphs
        self.hypergraphs = frozenset(powerset(self.cliques))

    def zero_out(self, key):
        """
        Returns the set of cliques remaining, after zeroing out `key`.
        """
        return frozenset([c for c in self.cliques if not key.issubset(c)])

    def zero_out_edges(self, edge):  # pylint: disable=unused-argument
        """
        Finds the cliques remaining, after zeroing out an edge.

        """
        # get the vertices which are relevant
        vertices = set(list(itertools.chain(*self.cliques)))
        # the edges, which we could zero out
        edges = itertools.combinations(vertices, 2)

        def cliques_remaining(edge):
            # note that < is comparing frozensets
            return frozenset([c for c in self.cliques if not frozenset(edge) < c])

        # note that this removes self-loops
        return list({cliques_remaining(e) for e in edges} - {self.cliques})
