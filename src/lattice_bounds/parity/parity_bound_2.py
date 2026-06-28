#!/usr/bin/env python3
"""Attempt at a lower bound for CLIQUE-PARITY."""

import argparse
import fractions
import itertools
import pdb
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


# Hypergeometric distribution, returning an exact fractions.fraction .
def hyperg_frac(N, K, n, k):
    # based on https://en.wikipedia.org/wiki/Hypergeometric_distribution
    # note that we don't try to optimize this
    return fractions.Fraction(comb(K, k) * comb(N - K, n - k), comb(N, n))


def flatten(nested_list):
    """Flattens a nested list into a single list."""
    return list(itertools.chain.from_iterable(nested_list))


# pylint: disable=too-many-instance-attributes
class ParityBound:
    """Attempt at bound for CLIQUE-PARITY."""

    def __init__(self, n, k, max_gates=None):
        """Constructor gets graph info, and sets up variable names.

        n: number of vertices in the graph
        k: number of vertices in a clique (>= 3)
        max_gates: maximum number of gates to consider (if this is too small,
            the LP will be infeasible)

        This will have the following variables:
        - ("G", v, n_gates): the number of functions with exactly `n_gates` gates,
            in sets of cliques that have _exactly_ `v` vertices.
        - ("V", v): expected number of gates to detect sets of cliques with
            exactly `v` vertices.
        - ("U", v): expected number of gates to detect sets of cliques with
            up to `v` vertices.
        - ("E", v, n_cliques): expected number of gates to detect sets of cliques with
            exactly `v` vertices and exactly `n_cliques` cliques.
        - ("C", n_cliques): expected number of gates for functions with
            n_cliques cliques
        """
        self.n = n
        self.k = k
        if k < 3:
            raise ValueError("k must be >= 3")
        # Number of possible cliques
        self.num_possible_cliques = comb(n, k)
        # Set max_gates (if not provided)
        if max_gates is None:
            raise ValueError("currently, max_gates must be provided")
        self.max_gates = max_gates

        # Number of functions for each number of vertices, with exactly some number
        # of vertices. (These correspond to the `V` variables.)
        hg_counter = hypergraph_counter.HypergraphCounter(
            self.n,
            self.k,
        )
        # the counts of BUGGYCLIQUE functions
        self.function_counts = hg_counter.count_hypergraphs_exact_vertices()

        # The variables for the LP.
        self.variables = []
        for v in range(self.k, self.n + 1):
            # Averages over number of vertices and cliques.
            self.variables += [("V", v), ("U", v)]
            # Counts for each number of gates.
            for g in range(1, self.max_gates + 1):
                self.variables += [("G", v, g)]
            # Averages by number of cliques.
            for n_cliques in range(comb(v, self.k) + 1):
                self.variables += [("E", v, n_cliques)]
        for n_cliques in range(0, self.num_possible_cliques + 1):
            self.variables += [("C", n_cliques)]
        # wrapper for LP solver
        self.lp = pulp_helper.PulpHelper(self.variables)
        # basis for gates
        self.basis = gate_basis.TwoInputNandBasis()

    def add_averaging_constraints(self):
        """Adds constraints on the average number of gates for each group."""
        # Connect function counts, and their expected value
        for v in range(self.k, self.n + 1):
            n_functions = sum(self.function_counts[v])
            # get expected number of gates, based on counts
            coefs = [(("G", v, g), g) for g in range(1, self.max_gates + 1)]
            self.lp.add_constraint(
                [(("V", v), -n_functions)] + coefs,
                "=",
                0,
            )
            # Constrain the number of functions in this group
            self.lp.add_constraint(
                [(("G", v, g), 1) for g in range(1, self.max_gates + 1)],
                "=",
                n_functions,
            )

    def add_marginal_constraints(self):
        """Add constraints for marginals of `E`, with respect to number of vertices
        and number of cliques."""
        # Marginals by number of cliques.
        for n_cliques in range(self.num_possible_cliques + 1):
            # Find range of number of vertices which have sets with <= n_cliques cliques.
            coefs = [
                (
                    ("E", v_prime, n_cliques),
                    self.function_counts[v_prime][n_cliques],
                )
                for v_prime in range(self.k, self.n + 1)
                if n_cliques < self.function_counts[v_prime].shape[0]
            ]
            total_counts = sum([a1[1] for a1 in coefs])
            self.lp.add_constraint([(("C", n_cliques), -total_counts)] + coefs, "=", 0)
        # Marginals by number of vertices.
        for v in range(self.k, self.n + 1):
            counts = self.function_counts[v]
            coefs = [
                (("E", v, n_cliques), counts[n_cliques])
                for n_cliques in range(len(counts))
            ]
            total_counts = sum([a1[1] for a1 in coefs])
            self.lp.add_constraint([(("V", v), -total_counts)] + coefs, "=", 0)
        # ??? add trivial lower bounds on these?

    def add_cumulative_constraints(self):
        """Add constraints connecting `V` and `U`.

        These reflect that `V` is "exactly some number of vertices",
        while `U` is "up to this number of vertices" (inclusive).
        """
        for v in range(self.k, self.n + 1):
            coefs = [
                (("V", v1), sum(self.function_counts[v1]))
                for v1 in range(self.k, v + 1)
            ]
            total_counts = sum([a1[1] for a1 in coefs])
            self.lp.add_constraint([(("U", v), -total_counts)] + coefs, "=", 0)

    def add_counting_bounds(self):
        """Adds counting bounds, for a given number of gates."""
        # First, compute number of possible functions
        # for each number of gates.
        num_possible_functions = self.basis.num_functions(
            comb(self.n, 2), self.max_gates
        )
        # Get the "V" variables, which represent sets of cliques
        # with exactly `v` vertices.
        v_vars = [x for x in self.variables if x[0] == "V"]
        # For each number of gates, bound the number of functions with
        # that many gates.
        for g in range(1, self.max_gates + 1):
            self.lp.add_constraint(
                [(("G", v, g), 1) for (_, v) in v_vars],
                "<=",
                num_possible_functions[g],
            )

    def add_zeroing_bound(self):
        """Adds bound from zeroing out one vertex."""
        # Add bound from zeroing one vertex (a sort of "step case").
        for v in range(self.k + 1, self.n):
            # Probability of hitting at least one clique by zeroing a vertex.
            p_at_least_one_hit = 1 - 2 ** (-comb(v - 1, self.k - 1))
            # If we "hit" a clique, we "zonk" at least one NAND gate.
            self.lp.add_constraint(
                [(("U", v - 1), 1), (("U", v), -1)],
                ">=",
                p_at_least_one_hit,
            )

    def add_slope_bound(self):
        """Bounds change in number of gates, with one additional clique."""
        # Upper bound on difference in number of gates between two functions
        # which differ by one clique. Note that this depends on the basis.
        max_n_gates_diff = self.basis.xor_of_and_upper_bound(comb(self.k, 2))
        for n_cliques in range(1, self.num_possible_cliques + 1):
            self.lp.add_constraint(
                [(("C", n_cliques), 1), (("C", n_cliques - 1), -1)],
                ">=",
                -max_n_gates_diff,
            )
            self.lp.add_constraint(
                [(("C", n_cliques), 1), (("C", n_cliques - 1), -1)],
                "<=",
                max_n_gates_diff,
            )

    def get_all_bounds(self):
        """Gets bounds for each possible number of cliques.

        Returns:
            pandas.DataFrame: bounds for each possible number of cliques
        """
        # We compute the bound for finding CLIQUE-PARITY, including
        # all the cliques.
        coefs = {("C", self.num_possible_cliques): 1}
        r = self.lp.solve_with_objective(coefs)
        if not r:
            return None
        n_cliques = np.arange(self.num_possible_cliques + 1)
        # pdb.set_trace()
        bounds = np.array([r[("C", nc)] for nc in n_cliques])
        return pandas.DataFrame(
            {
                "Num. cliques": n_cliques,
                "Min. gates": bounds,
            }
        )


def get_bounds(n, k, use_zeroing, use_slope, max_gates=None):
    """Gets bounds with some set of constraints.

    Args:
        n, k: problem size
        use_zeroing: whether to use zeroing bound
        use_slope: whether to use slope bound
        max_gates: maximum number of gates
    """
    # ??? track resource usage?
    sys.stderr.write(f"[bounding with n={n}, k={k}, max_gates={max_gates}]\n")
    bound = ParityBound(n, k, max_gates=max_gates)
    bound.add_averaging_constraints()
    bound.add_marginal_constraints()
    bound.add_cumulative_constraints()
    bound.add_counting_bounds()
    if use_zeroing:
        bound.add_zeroing_bound()
    if use_slope:
        bound.add_slope_bound()
    b = bound.get_all_bounds()
    b["label"] = f"zeroing={use_zeroing}, slope={use_slope}"
    return b


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Bounds on PARITY-CLIQUE.")
    parser.add_argument("n", type=int, help="Number of vertices")
    parser.add_argument("k", type=int, help="Size of cliques")
    parser.add_argument(
        "--max-gates", type=int, default=10, help="Maximum number of gates"
    )
    parser.add_argument(
        "--result-file",
        help="Write result to indicated file (by default, writes to stdout)",
    )
    return parser.parse_args()


def main():
    """Main execution entry point."""
    args = parse_args()
    bounds = [
        get_bounds(
            args.n,
            args.k,
            use_zeroing,
            use_slope,
            max_gates=args.max_gates,
        )
        for use_zeroing in [False, True]
        for use_slope in [False, True]
    ]

    bounds = pandas.concat(bounds)
    out_file = args.result_file if args.result_file else "/dev/stdout"
    with open(out_file, "wt", encoding="utf-8") as f:
        bounds.to_csv(f, index=False)


if __name__ == "__main__":
    main()
