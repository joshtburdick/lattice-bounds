#!/usr/bin/env python3
# Bound based on subsets, for _very_ tiny graphs.

import argparse
import itertools
import sys

import numpy as np
import pandas
import scipy.special
import scipy.stats

from lattice_bounds import hypergraph_counter
from lattice_bounds import pulp_helper


class SubsetBound:
    """Bound based on subsets of gates."""

    def __init__(self, n, k, max_gates):
        """Constructor gets graph info, and sets up variable names.

        n: number of vertices in the graph
        k: number of vertices in a clique (>= 3)
        max_gates: the number of gates to consider being present

        For now, we only are considering 'zeroing out' vertices;
        therefore, we label sets of cliques by the *set of vertices*.

        This will have the following 0-1 indicator variables:
        - ("V", a, gate_id): 1 iff `gate_id` is in the circuit
            for the set of cliques `a`.
        - ("E", a, b, gate_id): for each distinct a and b such that
            k <= |a| = |b| < n, this is 1 iff `gate_id`
            is in the circuit for `a` but not for `b`.
        """
        self.n = n
        self.k = k
        if k < 3:
            raise ValueError("k must be >= 3")
        # Number of possible cliques
        self.num_possible_cliques = comb(n, k)
        self.max_gates = max_gates

        # Vertices are numbered from 0 .. n-1.
        # The "vertices" of this graph are the subsets of {0..n-1}
        # with at least `k` elements
        self.big_V = [
            frozenset(c) for c in more_itertools.powerset(range(self.n)) if len(v) >= k
        ]
        self.num_big_v = len(self.big_V)

        # The "sideways edge" (a, b) ensures that there's some gate in
        # the circuit for a that's not in b.
        self.big_e = []
        for num_vertices in range(self.k, self.n):
            vertex_sets = [v for v in self.big_V if len(v) == num_vertices]
            for a, b in itertools.combinations(vertex_sets, 2):
                self.big_e += [(a, b), (b, a)]
        self.num_big_e = len(self.big_E)

        # The variables for the LP.
        self.variables = []
        for a in self.big_V:
            for gate_id in range(self.max_gates):
                self.variables += [("V", a, gate_id)]
        for a, b in self.big_E:
            for gate_id in range(self.max_gates):
                self.variables += [("E", a, b, gate_id)]

        # wrapper for LP solver
        self.lp = pulp_helper.PulpHelper(self.variables)

    def add_downwards_edge_constraints(self):
        """Adds constraints that zeroing a vertex zonks at least one gate."""
        for a in self.big_V:
            for v in a:
                b = a - frozenset(v)
                # constraint that a is a superset of b
                for g in range(self.max_gates):
                    self.lp.add_constraint(
                        [
                            (("V", a, g), 1),
                            (("V", b, g), -1),
                        ],
                        "<=",
                        0,
                    )
                # constraint that a has at least one more gate than b
                self.lp.add_constraint(
                    [(("v", a, g), 1) for g in range(self.max_gates)]
                    + [(("v", b, g), -1) for g in range(self.max_gates)],
                    ">=",
                    1,
                )

    def add_sideways_edge_constraints(self):
        """These define the edge sets, relative to vertex sets.

        These enforce that different sets (of a given size) must
        contain *something* different from each other.
        """
        for a, b in self.big_E:
            for gate_id in range(self.max_gates):
                # These two constraints, in combination, mean that,
                # for each gate, edge a->b is only 1 when vertex a
                # is 1 but vertex b is 0.
                self.lp.add_constraint(
                    [
                        (("E", a, b, gate_id), -1),
                        (("V", a, gate_id), -1),
                    ],
                    "<=",
                    0,
                )
                self.lp.add_constraint(
                    [
                        (("E", a, b, gate_id), -1),
                        (("V", b, gate_id), 1),
                    ],
                    "<=",
                    0,
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
