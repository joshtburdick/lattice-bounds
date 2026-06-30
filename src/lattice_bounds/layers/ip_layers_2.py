#!/usr/bin/env python3
"""Bound for detecting large sets of cliques."""

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
class LayerBound:
    """Attempt at bound for highest "layers" of BUGGYCLIQUE."""

    def __init__(self, n, k, max_gates=None):
        """Constructor gets graph info, and sets up variable names.

        n: number of vertices in the graph
        k: number of vertices in a clique (>= 3)
        max_gates: maximum number of gates to consider (if this is too small,
            the LP will be infeasible)

        This will have the following variables:
        - ("G", v, n_gates): the number of functions with exactly `n_gates` gates,
            in sets of cliques that have _exactly_ `v` vertices.
        - ("U", v): expected number of gates to detect sets of cliques with
            up to `v` vertices
        - ("V", v): expected number of gates to detect sets of cliques with
            exactly `v` vertices
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
            # Counts of number of functions with a given number of gates.
            for g in range(1, self.max_gates + 1):
                self.variables += [("G", v, g)]
            # Add the "V" and "U" variables for this number of vertices.
            self.variables += [("U", v), ("V", v)]
            # Averages over number of vertices and cliques.
            for n_cliques in range(comb(v, self.k) + 1):
                self.variables += [("U", v, n_cliques)]

        # wrapper for LP solver
        self.lp = pulp_helper.PulpHelper(self.variables)
        # basis for gates
        # FIXME(jtb): make this a parameter to constructor
        self.basis = gate_basis.TwoInputNandBasis()

    def add_averaging_constraints(self):
        """Adds constraints on the average number of gates for each group."""
        for v in range(self.k, self.n + 1):
            n_functions = self.function_counts[v].sum()
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

    def add_cumulative_constraints(self):
        """Add constraints connecting `V` and `U`.

        These reflect that `V` is "exactly some number of vertices",
        while `U` is "up to this number of vertices" (inclusive).
        """
        for v in range(self.k, self.n + 1):
            coefs = [
                (("V", v1), self.function_counts[v1].sum())
                for v1 in range(self.k, v + 1)
            ]
            total_counts = sum([a1[1] for a1 in coefs])
            self.lp.add_constraint([(("U", v), -total_counts)] + coefs, "=", 0)

    def add_marginal_constraints(self):
        """Connects U vars, marginalizing out the number of cliques.

        There is one of these constraints for each number of vertices.
        """
        for v in range(self.k, self.n + 1):
            n_functions = self.function_counts[v]
            coefs = [
                (("U", v, n_cliques), n_functions[n_cliques])
                for n_cliques in range(comb(v, self.k) + 1)
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
        # Add bound on detecting 0 cliques
        for v in range(self.k, self.n + 1):
            self.lp.add_constraint([(("U", v, 0), 1)], "=", 1)

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
            self.lp.add_constraint([(("U", v, 1), 1)], ">=", 2)
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
                # (again, assuming that we start with only v vertices
                # "in use" by the hyperedges)
                def p_hit(x):
                    return hyperg_frac(
                        comb(v, self.k), n_cliques_before, max_cliques_hit, x
                    )

                # The probability of at least one clique being hit
                p_at_least_one_hit = 1 - p_hit(0)
                # Coefficients for the difference in the number of gates,
                # before and after zeroing out a vertex.
                A = [(("U", v, n_cliques_before), 1)]
                A += [
                    (("U", v - 1, n_cliques_after[j]), -p_hit(n_cliques_hit[j]))
                    for j in range(len(n_cliques_after))
                ]
                # If we "hit" a clique, we "zonk" at least one NAND gate.
                self.lp.add_constraint(A, ">=", p_at_least_one_hit)

    def add_upper_bound(self):
        """Adds upper bound."""
        inputs_per_clique = comb(self.k, 2)
        # Given a circuit which detects some set of cliques, this is
        # the number of additional gates needed to detect one more clique.
        # FIXME (jtb): add this to the basis classes.
        gates_per_clique = 2 * inputs_per_clique
        for v in range(self.k, self.n + 1):
            for n_cliques in range(1, comb(v, self.k) + 1):
                self.lp.add_constraint(
                    [(("U", v, n_cliques), 1), (("U", v, n_cliques - 1), -1)],
                    "<=",
                    gates_per_clique,
                )

    def get_all_bounds(self, layer_size):
        """Gets bounds for each possible number of cliques.

        Args:
            layer_size: number of 'large' layers to bound

        Returns:
            pandas.DataFrame: bounds for each possible number of cliques
        """
        # Get weights for 'large' layers.
        n_functions_in_layer = {
            n_cliques: comb(self.num_possible_cliques, n_cliques)
            for n_cliques in range(
                self.num_possible_cliques - layer_size, self.num_possible_cliques + 1
            )
        }
        total_functions_in_layers = sum(n_functions_in_layer.values())
        coefs = {
            ("U", self.n, nc): (weight / total_functions_in_layers)
            for nc, weight in n_functions_in_layer.items()
        }
        # Solve, minimizing number of gates to detect a set of cliques
        # in the "large" layers.
        r = self.lp.solve_with_objective(coefs)
        if not r:
            return None
        # Get bounds for "expected number of gates" for functions in
        # each layer. To simplify plotting, we include the two endpoints
        # of each layer.
        n_cliques = np.arange(self.num_possible_cliques + 1)
        bounds = np.array([r[("U", self.n, nc)] for nc in n_cliques])
        return pandas.DataFrame(
            {
                "Num. cliques": n_cliques,
                "Min. gates": bounds,
            }
        )


def get_bounds(n, k, layer_size, use_zeroing, use_upper, max_gates=None):
    """Gets bounds with some set of constraints.

    Args:
        n, k: problem size
        layer_size: number of 'large' layers to bound
        use_zeroing: whether to use zeroing bound
        use_upper: whether to use upper bound
        max_gates: maximum number of gates
    """
    # ??? track resource usage?
    sys.stderr.write(f"[bounding with n={n}, k={k}, max_gates={max_gates}]\n")
    bound = LayerBound(n, k, max_gates=max_gates)
    bound.add_averaging_constraints()
    bound.add_cumulative_constraints()
    bound.add_counting_bounds()
    bound.add_marginal_constraints()
    if use_zeroing:
        bound.add_zeroing_bound()
    if use_upper:
        bound.add_upper_bound()
    b = bound.get_all_bounds(layer_size)
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
        "layer_size", type=int, help="Number of 'large' layers to bound"
    )
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
            args.layer_size,
            use_zeroing,
            use_upper,
            max_gates=args.max_gates,
        )
        for use_zeroing in [False, True]
        for use_upper in [False, True]
    ]

    bounds = pandas.concat(bounds)
    out_file = args.result_file if args.result_file else "/dev/stdout"
    with open(out_file, "wt", encoding="utf-8") as f:
        bounds.to_csv(f, index=False)


if __name__ == "__main__":
    main()
