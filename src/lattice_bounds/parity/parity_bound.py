#!/usr/bin/env python3
"""Attempt at a lower bound for CLIQUE-PARITY."""

import argparse
import fractions
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
    """Attempt at bound for highest "layers" of BUGGYCLIQUE."""

    def __init__(self, n, k, max_gates=None):
        """Constructor gets graph info, and sets up variable names.

        n: number of vertices in the graph
        k: number of vertices in a clique (>= 3)
        max_gates: maximum number of gates to consider (if this is too small,
            the LP will be infeasible)

        This will have the following variables:
        - ("G", v, n_cliques, n_gates): the number of functions with exactly `n_gates` gates,
            in sets of cliques that have _exactly_ `v` vertices and _exactly_ `n_cliques` cliques.
        - ("V", v, n_cliques): expected number of gates to detect sets of cliques with
            exactly `v` vertices and exactly `n_cliques` cliques.
        - ("U", v, n_cliques): expected number of gates to detect sets of cliques with
            up to `v` vertices and exactly `n_cliques` cliques.
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
            for n_cliques in range(comb(v, self.k) + 1):
                # Averages over number of vertices and cliques.
                self.variables += [("V", v, n_cliques), ("U", v, n_cliques)]
                # Counts for each number of gates.
                for g in range(1, self.max_gates + 1):
                    self.variables += [("G", v, n_cliques, g)]
        # wrapper for LP solver
        self.lp = pulp_helper.PulpHelper(self.variables)
        # basis for gates
        self.basis = gate_basis.TwoInputNandBasis()

    def add_averaging_constraints(self):
        """Adds constraints on the average number of gates for each group."""
        for v in range(self.k, self.n + 1):
            n_functions = self.function_counts[v]
            for n_cliques in range(comb(v, self.k) + 1):
                # get expected number of gates, based on counts
                coefs = [
                    (("G", v, n_cliques, g), g) for g in range(1, self.max_gates + 1)
                ]
                self.lp.add_constraint(
                    [(("V", v, n_cliques), -n_functions[n_cliques])] + coefs,
                    "=",
                    0,
                )
                # Constrain the number of functions in this group
                self.lp.add_constraint(
                    [(("G", v, n_cliques, g), 1) for g in range(1, self.max_gates + 1)],
                    "=",
                    n_functions[n_cliques],
                )

    def add_cumulative_constraints(self):
        """Add constraints connecting `V` and `U`.

        These reflect that `V` is "exactly some number of vertices",
        while `U` is "up to this number of vertices" (inclusive).
        """
        for v in range(self.k, self.n + 1):
            for n_cliques in range(comb(v, self.k) + 1):
                coefs = [
                    (("V", v1, n_cliques), self.function_counts[v1][n_cliques])
                    for v1 in range(self.k, v + 1)
                    if n_cliques < len(self.function_counts[v1])
                ]
                total_counts = sum([a1[1] for a1 in coefs])
                self.lp.add_constraint(
                    [(("U", v, n_cliques), -total_counts)] + coefs, "=", 0
                )

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
                [(("G", v, n_cliques, g), 1) for (_, v, n_cliques) in v_vars],
                "<=",
                num_possible_functions[g],
            )

    def add_zeroing_bound(self):
        """Adds bound from zeroing out one vertex."""
        # As a "base case", we hard-wire the bound for BUGGYCLIQUE when k=n.
        # For the unbounded-fan-in NAND gate basis, this is 2.
        # For 2-input NAND gates, this could be higher; but we only
        # use the bound of 2, since in general, zeroing will only give a
        # bound of (just under) one fewer NAND gate. (With a handful
        # of cliques, we could do better; but this advantage would
        # vanish as we add more cliques. So for simplicity, we don't
        # do this.)
        for v in range(self.k, self.n + 1):
            self.lp.add_constraint([(("V", v, 1), 1)], ">=", 2)
        # Add bound from zeroing one vertex (a sort of "step case").
        for v in range(self.k + 1, self.n + 1):
            # loop through number of cliques in that graph
            for n_cliques_before in range(1, comb(v, self.k) + 1):
                # Maximum number of cliques we might hit, assuming that we start
                # with a graph which only uses v vertices.
                max_cliques_hit = comb(v - 1, self.k - 1)
                # Bounds on number of cliques zeroed.
                # The min is at least 1 (assuming we can "re-roll"), and no
                # more than the difference between the number of cliques in the
                # larger set, and the number left over.
                min_zeroed = max(1, n_cliques_before - comb(v - 1, self.k))
                # The max is limited by the current number of vertices (and how
                # many cliques hit one vertex), and the number of cliques.
                max_zeroed = min(max_cliques_hit, n_cliques_before)
                # the range of possible number of cliques zeroed ...
                n_cliques_hit = np.arange(min_zeroed, max_zeroed + 1)
                # ... and the number left over
                n_cliques_after = n_cliques_before - n_cliques_hit

                # The probability of some number of cliques being hit
                # (again, assuming that we start with only v vertices are "in use" by
                # the hyperedges)
                def p_hit(x):
                    return hyperg_frac(
                        comb(v, self.k), n_cliques_before, max_cliques_hit, x
                    )

                # The probability of at least one clique being hit
                p_at_least_one_hit = 1 - p_hit(0)
                print(f"p_at_least_one_hit={p_at_least_one_hit}")
                # Coefficients for the difference in the number of gates,
                # before and after zeroing out a vertex.
                A = [(("U", v, n_cliques_before), 1)]
                A += [
                    (("U", v - 1, n_cliques_after[j]), -p_hit(j))
                    for j in range(len(n_cliques_after))
                ]
                # If we "hit" a clique, we "zonk" at least one NAND gate.
                self.lp.add_constraint(A, ">=", p_at_least_one_hit)

    def add_gap_bound(self):
        """Adds bound based on the 'gap' between large and small sets of cliques.

        FIXME: try adding the average of the top L and bottom L layers?
        """
        N = self.num_possible_cliques
        for i in range(1, N // 2):
            # We have:
            # C_{n-i} <= C_n + C_i + 1 XOR
            # C_n >= C_{n-i} - C_i - 1 XOR
            # C_n - C_{n-i} + C_i >= -1 XOR
            self.lp.add_constraint(
                [
                    (("U", self.n, N), 1),
                    (("U", self.n, N - i), -1),
                    (("U", self.n, i), 1),
                ],
                ">=",
                -self.basis.xor_upper_bound(),
            )

    def add_upper_bound(self):
        """Adds upper bound, relative to the 'grand mean'."""
        # ??? not sure this will be worthwhile
        pass

    def get_all_bounds(self):
        """Gets bounds for each possible number of cliques.

        Returns:
            pandas.DataFrame: bounds for each possible number of cliques
        """
        # We compute the bound for finding CLIQUE-PARITY, including
        # all the cliques.
        coefs = {("U", self.n, self.num_possible_cliques): 1}
        r = self.lp.solve_with_objective(coefs)
        if not r:
            return None
        n_cliques = np.arange(self.num_possible_cliques + 1)
        bounds = np.array([r[("U", self.n, nc)] for nc in n_cliques])
        return pandas.DataFrame(
            {
                "Num. cliques": n_cliques,
                "Min. gates": bounds,
            }
        )


def get_bounds(n, k, use_zeroing, use_upper, max_gates=None):
    """Gets bounds with some set of constraints.

    Args:
        n, k: problem size
        use_zeroing: whether to use zeroing bound
        use_upper: whether to use upper bound
        max_gates: maximum number of gates
    """
    # ??? track resource usage?
    sys.stderr.write(f"[bounding with n={n}, k={k}, max_gates={max_gates}]\n")
    bound = ParityBound(n, k, max_gates=max_gates)
    bound.add_averaging_constraints()
    bound.add_cumulative_constraints()
    bound.add_counting_bounds()
    bound.add_gap_bound()
    if use_zeroing:
        bound.add_zeroing_bound()
    if use_upper:
        bound.add_upper_bound()
    b = bound.get_all_bounds()
    b["label"] = f"zeroing={use_zeroing}, upper={use_upper}"
    return b


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bounds using 'picky' clique detection."
    )
    parser.add_argument("n", type=int, help="Number of vertices")
    parser.add_argument("k", type=int, help="Size of cliques")
    parser.add_argument(
        "--max-gates", type=int, default=10, help="Maximum number of gates"
    )
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
            use_zeroing,
            False,  # use_upper
            max_gates=args.max_gates,
        )
        for use_zeroing in [False, True]
    ]

    bounds = pandas.concat(bounds)
    out_file = args.result_file if args.result_file else "/dev/stdout"
    with open(out_file, "wt", encoding="utf-8") as f:
        bounds.to_csv(f, index=False)


if __name__ == "__main__":
    main()
