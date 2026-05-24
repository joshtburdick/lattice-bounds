#!/usr/bin/env python3
"""Bound for clique, using PICKYCLIQUE function."""

import argparse
import itertools
import sys

import numpy as np
import pandas
import scipy.special
import scipy.stats

from lattice_bounds import gate_basis
from lattice_bounds import hypergraph_counter
from lattice_bounds import pulp_helper


def comb(n, k):
    """Wrapper for comb(), with exact arithmetic."""
    return scipy.special.comb(n, k, exact=True)


def flatten(nested_list):
    """Flattens a nested list into a single list."""
    return list(itertools.chain.from_iterable(nested_list))


# pylint: disable=too-many-instance-attributes
class PickyBound:
    """Attempt at bound for "layers" of the clique problem."""

    def __init__(self, n, k, max_gates=None):
        """Constructor gets graph info, and sets up variable names.

        n: number of vertices in the graph
        k: number of vertices in a clique (>= 3)
        max_gates: maximum number of gates to consider (if this is too small,
            the LP will be infeasible)

        In the following, `picky` = 0 for BUGGYCLIQUE, 1 for PICKYCLIQUE.

        This will have the following variables:
        - ("G", picky, v, n_gates): the number of functions with exactly `n_gates` gates,
            in sets of cliques that have _exactly_ `v` vertices.
        - ("V", picky, v): expected number of gates to detect sets of cliques with
            _exactly_ `v` vertices.
        - ("U", picky, v): expected number gates to detect sets of cliques with
            _up to_ `v` vertices (and including BUGGYCLIQUE functions in PICKYCLIQUE).
        - ("X", picky, n_cliques): expected number of gates to detect sets containing
            `n_cliques` cliques (and any number of vertices).
        ???
        - should we include the empty set of cliques?
        """
        self.n = n
        self.k = k
        if k < 3:
            raise ValueError("k must be >= 3")
        # Number of possible cliques
        self.num_possible_cliques = comb(n, k)
        # Set max_gates (if not provided)
        if max_gates:
            self.max_gates = max_gates
        else:
            # if max. gates isn't specified, use the number of possible cliques
            # plus a buffer. (This is an overestimate of what's needed, and also
            # assumes that the unbounded-fan-in NAND basis is being used.)
            self.max_gates = self.num_possible_cliques + 2

        # Number of functions for each number of vertices, with exactly some number
        # of vertices. (These correspond to the `V` variables.)
        hg_counter = hypergraph_counter.HypergraphCounter(
            self.n,
            self.k,
        )
        # first, the counts for BUGGYCLIQUE
        buggy_counts = hg_counter.count_hypergraphs_exact_vertices()
        # then, the counts for PICKYCLIQUE
        picky_counts = {}
        for v, counts in buggy_counts.items():
            # For each set of cliques, any possible subset of them could be
            # the "NO" cliques (except for all of them, or the empty set).
            picky_counts[v] = counts * (2 ** np.arange(len(counts)) - 2)
        self.function_counts = [buggy_counts, picky_counts]

        # The variables for the LP.
        lp_vars = []
        for picky in [0, 1]:
            for v in range(self.k, self.n + 1):
                lp_vars += [("V", picky, v), ("U", picky, v)]
                # counts for each number of gates
                for g in range(self.max_gates + 1):
                    lp_vars += [("G", picky, v, g)]
        # Averages, by number of vertices, and layer
        for n_cliques in range(self.num_possible_cliques + 1):
            lp_vars += [("X", 0, n_cliques), ("X", 1, n_cliques)]
        # wrapper for LP solver
        self.lp = pulp_helper.PulpHelper(lp_vars)
        # basis for gates
        self.basis = gate_basis.UnboundedFanInNandBasis()

    def add_averaging_constraints(self):
        """Adds constraints on the average number of gates for each group."""
        for picky in [0, 1]:
            for v in range(self.k, self.n + 1):
                coefs = [((v, layer, g), g) for g in range(self.max_gates + 1)]
                # "expected number of gates" = sum(counts * gates) / sum(counts)
                self.lp.add_constraint(
                    [(("V", picky, v), -self.function_counts[picky][v])],
                    "=",
                    0,
                )
                # add constraints on the number of functions in each group
                self.lp.add_constraint(
                    [(("G", picky, v, g), 1) for g in range(self.max_gates + 1)],
                    "=",
                    self.function_counts[picky][v],
                )
            # add constraints connecting "V" and "U"

            # add constraints connecting "X" and "V"

    def add_picky_bound(self):
        """This connects the bounds for BUGGYCLIQUE and PICKYCLIQUE."""
        for i in range(1, self.num_possible_cliques + 1):
            pass

    def add_counting_bounds(self):
        """Adds counting bounds, for a given number of gates."""
        # first, compute number of possible functions
        # for each number of gates
        num_possible_functions = self.basis.num_functions(
            comb(self.n, 2), self.max_gates
        )
        # Get the "V" variables, which represent sets of cliques
        # with exactly `v` vertices.
        vars = [x for x in self.lp.variables if x[0] == "V"]
        # For each number of gates, bound the number of functions with
        # that many gates.
        for g in range(self.max_gates + 1):
            self.lp.add_constraint(
                [(("G", picky, v, g), 1) for (_, picky, v) in vars],
                "<=",
                num_possible_functions[g],
            )

    def add_zeroing_bound(self):
        """Adds bound from zeroing out one vertex.

        We expect to hit at least one clique (and thus one gate). However,
        it seems simplest to use the "U" ("up to some number of vertices") bounds,
        and so we're not guaranteed to hit a clique.

        Therefore, for simplicity, we use a bound of 0.5 "gates hit".
        This seems conservative, as the actual number of cliques (and thus gates)
        is just below 1.

        FIXME:
        - check that 0.5 is a valid bound
        - can we get a tighter bound than 0.5?
        """
        for v in range(self.k + 1, self.n):
            # the bound for BUGGYCLIQUE
            self.lp.add_constraint([(("U", 0, v + 1), 1), (("U", 0, v), -1)], ">=", 0.5)
            # the bound for PICKYCLIQUE will have slightly more slack
            self.lp.add_constraint([(("U", 1, v + 1), 1), (("U", 1, v), -1)], ">=", 0.5)

    def add_upper_bound(self):
        """Adds upper bound.

        This assumes that we're using the unbounded-fan-in NAND gate basis.
        """
        for n_cliques in range(1, self.num_possible_cliques + 1):
            self.lp.add_constraint([(("X", n_cliques), 1)], "<=", n_cliques + 1)

    def get_all_bounds(self):
        """Gets bounds for each possible number of cliques.

        This is the bounds for each possible number of cliques,
        in the scenario that the number of gates for
        CLIQUE is minimized.
        """
        # solve, minimizing number of gates in "highest layer"
        r = self.lp.solve(("X", 0, self.num_possible_cliques))
        if not r:
            return None
        # Get bounds for "expected number of gates" for functions in
        # each layer. To simplify plotting, we include the two endpoints
        # of each layer.
        n_cliques = np.arange(self.num_possible_cliques + 1)
        bounds = np.array([r[("X", 0, nc)] for nc in n_cliques])
        return pandas.DataFrame(
            {
                "Num. cliques": n_cliques,
                "Min. gates": bounds,
            }
        )


def get_bounds(n, k, label, use_zeroing, use_upper):
    """Gets bounds with some set of constraints.

    Args:
        n, k: problem size
        label: name to use for this configuration
        use_zeroing: whether to use zeroing bound
        use_upper: whether to use upper bound
    """
    # ??? track resource usage?
    sys.stderr.write(f"[bounding with n={n}, k={k}]\n")
    bound = PickyBound(n, k)
    bound.add_averaging_constraints()
    bound.add_marginal_constraints()
    bound.add_picky_bound()
    if use_zeroing:
        bound.add_zeroing_bound()
    if use_upper:
        bound.add_upper_bound()
    b = bound.get_all_bounds()
    return b


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bounds using 'picky' clique detection."
    )
    parser.add_argument("n", type=int, help="Number of vertices")
    parser.add_argument("k", type=int, help="Size of cliques")
    parser.add_argument(
        "--result-file", help="Write result to indicated file (rather than stdout)"
    )
    return parser.parse_args()


def main():
    """Main execution entry point."""
    args = parse_args()
    bounds = [
        get_bounds(
            args.n,
            args.k,
            f"use_zeroing={use_zeroing}, use_upper={use_upper}",
            use_zeroing,
            use_upper,
        )
        for use_zeroing in [True, False]
        for use_upper in [True, False]
    ]

    bounds = pandas.concat(bounds)
    out_file = args.result_file if args.result_file else "/dev/stdout"
    with open(out_file, "wt", encoding="utf-8") as f:
        bounds.to_csv(f, index=False)


if __name__ == "__main__":
    main()
