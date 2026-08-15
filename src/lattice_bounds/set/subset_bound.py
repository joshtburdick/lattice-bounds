#!/usr/bin/env python3
# Bound based on subsets, for _very_ tiny graphs.

import argparse
import itertools
import more_itertools
import pdb
import sys

import numpy as np
import pandas
from scipy.special import comb
import scipy.stats

from lattice_bounds import hypergraph_counter
from lattice_bounds import pulp_helper


class SubsetBound:
    """Bound based on subsets of gates."""

    def __init__(self, n, k, max_gates=None):
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
        self.big_v = [
            frozenset(v) for v in more_itertools.powerset(range(self.n)) if len(v) >= k
        ]
        self.num_big_v = len(self.big_v)

        # The "sideways edge" (a, b) ensures that there's some gate in
        # the circuit for a that's not in b.
        self.big_e = []
        for num_vertices in range(self.k, self.n):
            vertex_sets = [v for v in self.big_v if len(v) == num_vertices]
            for a, b in itertools.combinations(vertex_sets, 2):
                self.big_e += [(a, b), (b, a)]
        self.num_big_e = len(self.big_e)

        # The variables for the LP.
        self.variables = []
        for a in self.big_v:
            for gate_id in range(self.max_gates):
                self.variables += [("V", a, gate_id)]
        for a, b in self.big_e:
            for gate_id in range(self.max_gates):
                self.variables += [("E", a, b, gate_id)]

        # wrapper for LP solver
        self.lp = pulp_helper.PulpHelper(self.variables)

        # bound all variables in [0, 1] (there may be a
        # special function to do this, but for now, we
        # skip that)
        for v in self.variables:
            self.lp.add_constraint([(v, 1)], ">=", 0)
            self.lp.add_constraint([(v, 1)], "<=", 1)

    def add_downward_edge_constraints(self):
        """Adds constraints that zeroing a vertex zonks at least one gate."""
        for a in self.big_v:
            # The circuit for each clique has at least one gate.
            if len(a) == self.k:
                self.lp.add_constraint(
                    [(("V", a, g), 1) for g in range(self.max_gates)],
                    ">=",
                    1,
                )
                continue
            # If a has > k vertices, zeroing it out must zonk at least one gate,
            for v in a:
                b = a - frozenset([v])
                # constraint that a is a superset of b
                for g in range(self.max_gates):
                    self.lp.add_constraint(
                        [
                            (("V", a, g), 1),
                            (("V", b, g), -1),
                        ],
                        ">=",
                        0,
                    )
                # constraint that a has at least one more gate than b
                self.lp.add_constraint(
                    [(("V", a, g), 1) for g in range(self.max_gates)]
                    + [(("V", b, g), -1) for g in range(self.max_gates)],
                    ">=",
                    1,
                )

    def add_sideways_edge_constraints(self):
        """These define the edge sets, relative to vertex sets.

        These enforce that different sets (of a given size) must
        contain *something* different from each other.
        """
        for a, b in self.big_e:
            for gate_id in range(self.max_gates):
                # These two constraints, in combination, mean that,
                # for each gate, edge a->b is only 1 when vertex a
                # is 1 but vertex b is 0.
                self.lp.add_constraint(
                    [
                        (("E", a, b, gate_id), 1),
                        (("V", a, gate_id), -1),
                    ],
                    "<=",
                    0,
                )
                self.lp.add_constraint(
                    [
                        (("E", a, b, gate_id), 1),
                        (("V", b, gate_id), 1),
                    ],
                    "<=",
                    1,
                )
                # Also, at least one of these must be >= 1.
                self.lp.add_constraint(
                    [(("E", a, b, g), 1) for g in range(self.max_gates)],
                    ">=",
                    1,
                )

    def get_all_bounds(self):
        """Gets bounds for each possible number of cliques.

        Returns:
            pandas.DataFrame: bounds for each possible number of cliques
        """
        vertices = frozenset(range(self.n))
        # Objective function: how many of the gates are in the set
        # with all of the vertices?
        coefs = {("V", vertices, g): 1 for g in range(self.max_gates)}
        r = self.lp.solve_with_objective(coefs)
        bound = -1
        if r:
            bound = r["__objective__"]
        return pandas.DataFrame(
            {
                "n": [self.n],
                "k": [self.k],
                "max_gates": [self.max_gates],
                "min_gates_bound": [bound],
            }
        )


def get_bounds(n, k, use_downward, use_sideways, max_gates=None):
    """Gets bounds with some set of constraints.

    Args:
        n, k: problem size
        use_downward: whether to use downward-edge constraints
        use_sideways: whether to use sideways-edge constraints
        max_gates: maximum number of gates
    """
    # ??? track resource usage?
    sys.stderr.write(f"[bounding with n={n}, k={k}, max_gates={max_gates}]\n")
    bound = SubsetBound(n, k, max_gates=max_gates)
    if use_downward:
        bound.add_downward_edge_constraints()
    if use_sideways:
        bound.add_sideways_edge_constraints()
    b = bound.get_all_bounds()
    b["label"] = f"downward={use_downward}, sideways={use_sideways}"
    return b


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Bounds on PARITY-CLIQUE.")
    parser.add_argument("n", type=int, help="Number of vertices")
    parser.add_argument("k", type=int, help="Size of cliques")
    parser.add_argument("--max-gates", type=int, help="Maximum number of gates")
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
            use_downward,
            use_sideways,
            max_gates=args.max_gates,
        )
        for use_downward in [False, True]
        for use_sideways in [False, True]
    ]

    bounds = pandas.concat(bounds)
    out_file = args.result_file if args.result_file else "/dev/stdout"
    with open(out_file, "wt", encoding="utf-8") as f:
        bounds.to_csv(f, index=False)


if __name__ == "__main__":
    main()
