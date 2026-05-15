#!/usr/bin/env python3
# Utilities for counting hypergraphs in subsets of vertices.

import math
import pdb
import sys

import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.special


def comb(n, k):
    """Short name for comb()."""
    return scipy.special.comb(n, k, exact=True)


class HypergraphCounter:
    """Counts hypergraphs in subsets of vertices.

    When we restrict to a subset of vertices, presumably we'll
    need to keep track of how many hypergraphs have a given
    number of hyperedges.

    By convention, the empty set is counted as "having k-1 vertices".

    This now uses Python large integers, and so n and k are
    mostly limited by time and memory.
    """

    def __init__(self, n, k):
        """Constructor.

        n: number of vertices in the larger graph
        k: size of cliques (hyperedges)
        """
        self.n = n
        self.k = k
        self.debug_print = None

    def count_hypergraphs_exact_vertices_subgraph(self):
        """Counts hypergraphs in smaller subgraphs.

        FIXME rename this!

        Note that this only includes one subset of the vertices.
        (count_hypergraphs_exact_vertices() includes different labellings
        of all n vertices.)
        The nice thing about these is that since they all "use" some
        number of vertices, they're all distinct.
        This is used by "count_hypergraphs_max_vertices".
        """
        exact_counts = {self.k - 1: np.array([1], dtype="object")}
        # loop through number of vertices
        for i in range(self.k, self.n + 1):
            if self.debug_print:
                print(i, end=" ")
                sys.stdout.flush()
            # start with count of hypergraphs (on all n vertices) with
            # _up to_ this many vertices
            num_cliques = scipy.special.comb(i, self.k, exact=True)
            exact_counts[i] = np.array(
                [
                    scipy.special.comb(num_cliques, r, exact=True)
                    for r in range(num_cliques + 1)
                ],
                dtype="object",
            )
            # also, don't count case with zero hypergraphs (as that's not
            # specific to a particular vertex set)
            exact_counts[i][0] = 0
            # then, subtract off hypergraphs with fewer vertices (if any)
            for j in range(self.k, i):
                n1 = exact_counts[j].shape[0]
                exact_counts[i][:n1] -= (
                    scipy.special.comb(i, j, exact=True) * exact_counts[j]
                )
        if self.debug_print:
            print("\n")
        return exact_counts

    def count_hypergraphs_max_vertices(self):
        """Counts hypergraphs with _up to_ some number of vertices.

        Returns: a dict h, with key v, an int in the range k..n,
            representing the number of vertices.
            h[v] is a numpy array of length ${v \choose k}$.
            h[v][i] is the number of hypergraphs with up to
                $v$ vertices, and exactly $i$ hyperedges.
        """
        # to compute this, first compute number of hypergraphs with
        # exactly some number of vertices used
        exact_counts = self.count_hypergraphs_exact_vertices_subgraph()
        # this will hold the vectors of counts, for each number of vertices
        h = {self.k - 1: np.array([1], dtype="object")}
        # then, loop through the number of vertices
        for i in range(self.k, self.n + 1):
            # start with a count of 0
            num_cliques = scipy.special.comb(i, self.k, exact=True)
            h[i] = np.full([num_cliques + 1], 0, dtype="object")
            # add in number of hypergraphs with up to that many vertices
            for j in range(self.k, i + 1):
                n1 = exact_counts[j].shape[0]
                h[i][:n1] += scipy.special.comb(self.n, j, exact=True) * exact_counts[j]
            # also count the empty set
            h[i][0] = 1
        return h

    def count_hypergraphs_exact_vertices(self):
        """Counts hypergraphs which use exactly some number of vertices.

        Another way of looking at this is that, for each hypergraph, this
        finds the maximal number of vertices which could be zonked.

        There may be a faster / more understandable way to do this,
        but I'm ignoring that for now.
        """
        # get max. vertex counts
        max_vertex_counts = self.count_hypergraphs_max_vertices()
        # initialize the count, for the empty graph
        exact_counts = {self.k - 1: np.array([1], dtype="object")}
        # this will be the totals of all
        smaller_totals = np.zeros(comb(self.n, self.k) + 1, dtype="object")
        # initially, this is just the empty set
        # smaller_totals[0] = 1
        # loop through number of vertices
        for i in range(self.k, self.n + 1):
            # print(f'i = {i}')
            # start with (a copy of) the max. counts for this number of vertices
            b = np.array(max_vertex_counts[i])
            # subtract off total, for one fewer vertices
            a = max_vertex_counts[i - 1]
            b[: len(a)] -= a
            exact_counts[i] = b
        return exact_counts


def count_bits(x):
    """Counts number of bits set in a numpy vector."""
    # we assume numbers are "somewhat small" (if they were
    # large, they'd take a while to loop through anyway)
    num_bits_set = np.zeros(x.shape[0])
    for i in range(30):
        mask = np.array(2**i)
        num_bits_set += ((x & mask) > 0) + 0
    return num_bits_set


class SlowHypergraphCounter:
    """Slower reference implementation of HypergraphCounter."""

    def __init__(self, n, k):
        """Constructor."""
        # restrict problem size
        if scipy.special.comb(n, k, exact=True) >= 40:
            raise ValueError("n choose k would use a lot of memory; exiting")
        self.n = n
        self.k = k
        # numbering for the cliques
        cliques = [frozenset(s) for s in itertools.combinations(range(n), k)]
        self.clique_index = dict(zip(cliques, range(len(cliques))))
        # the sets of cliques
        self.S = np.arange(self.num_functions)
        # the sets of cliques which are "zeroable", and don't overlap S
        self.Z = np.zeros(self.num_functions, dtype="int")

    def count_zero_set(self, cliques_to_count):
        """Adds everywhere that some cliques could be zeroed.

        cliques_to_count: the cliques to count, as an int
            representing a bitset
        Side effects: fills in bits of Z which correspond to cliques
            which aren't in this set (in S), but could be zeroed
        """
        # find sets where those cliques would be missed
        i = (self.S & cliques_to_count) == 0
        # add those to Z
        self.Z[i] = self.Z[i] | cliques_to_count

    def zero_vertices(self):
        """Finds sets of cliques unaffected by zeroing out vertices.

        This finds the sets of cliques which wouldn't be 'hit'
        by zeroing out edges incident to some vertex.
        """
        # loop through the vertices
        for v in range(self.n):
            # find cliques including that vertex
            mask = 0
            vertex_set = frozenset([v])
            # loop through the cliques
            for clique, i in self.clique_index.items():
                # if the edge is in this clique, set the appropriate bit
                if clique > vertex_set:
                    mask |= 2**i
            # zero out that set of cliques
            self.count_zero_set(mask)

    def count_hypergraphs(self):
        """Counts hypergraphs having some number of vertices."""
        pass


# XXX a quick test
if __name__ == "__main__":
    hc = HypergraphCounter(int(sys.argv[1]), int(sys.argv[2]))
    # this is a toy example, but is small enough to check by hand
    # hc = HypergraphCounter(6, 3)
    # slightly larger
    # hc = HypergraphCounter(6, 3)
    # a slightly larger example (which nonetheless seems to require bigints)
    # hc = HypergraphCounter(15, 5)
    # a bigger example
    # hc = HypergraphCounter(30, 3)

    print("exact-number-of-vertex counts (in subgraphs):")
    for v, h in hc.count_hypergraphs_exact_vertices_subgraph().items():
        h = " ".join([str(x) for x in h])
        print(f"{v}: {h}")

    print("up-to-some-number-of-vertex counts:")
    for v, h in hc.count_hypergraphs_max_vertices().items():
        h = " ".join([str(x) for x in h])
        print(f"{v}: {h}")

    print("exact-number-of-vertex counts:")
    exact_counts = hc.count_hypergraphs_exact_vertices()
    for v, h in exact_counts.items():
        h = " ".join([str(x) for x in h])
        print(f"{v}: {h}")

    # check sum of these
    total_hypergraphs = sum([s.sum() for s in exact_counts.values()])
    print(f"total hypergraphs = {total_hypergraphs}")
