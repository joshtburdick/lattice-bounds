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

import pysat
from pysat import formula, solvers


class SubsetBound:
    """Bound based on subsets of gates."""

    def __init__(self, n, k, max_gates):
        """Constructor gets graph info, and sets up variable names.

        n: number of vertices in the graph
        k: number of vertices in a clique (>= 3)
        max_gates: the number of gates to consider being present

        For now, we only are considering 'zeroing out' vertices;
        therefore, we label sets of cliques by the *set of vertices*.

        This will have the following boolean variables:
        - ("V", a, gate_id): true iff `gate_id` is in the circuit
            for the set of cliques `a`.
        - ("E", a, b, gate_id): for each distinct a and b such that
            k <= |a| = |b| < n, this is true iff `gate_id`
            is in the circuit for `a` but not for `b`.
            ??? are "E" variables needed?
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

        # Enumerate the variables (this may not be necessary)
        self.variable_names = []
        for a in self.big_v:
            for gate_id in range(self.max_gates):
                self.variable_names += [(a, gate_id)]
        self.var_to_object = {x: formula.Atom(str(x)) for x in self.variable_names}

        self.solver = solvers.Solver(name="g3")

    def add_different_gates_constraint(self, a, b):
        """Adds a constraint that `b` contains at least one gate not in `a`."""

        # Each of these clauses indicates that a particular gate is in `b` but not `a`.
        clause = formula.Atom(False)
        for g in range(self.max_gates):
            clause = formula.Or(
                clause,
                formula.And(
                    formula.Neg(self.var_to_object[(a, g)]),
                    self.var_to_object[(b, g)],
                ),
            )
        self.solver.append_formula(clause)

    def add_downward_constraints(self):
        """Adds constraints that smaller sets of vertices correspond to smaller circuits."""
        for a in self.big_v:
            # The circuit for each individual clique has at least one gate.
            if len(a) == self.k:
                clause = formula.Atom(False)
                for g in range(self.max_gates):
                    clause = formula.Or(
                        clause,
                        self.var_to_object[(a, g)],
                    )
                self.solver.append_formula(clause)
                continue
            # If a has > k vertices, then sets with one fewer vertex must be
            # contained in a.
            for v in a:
                b = a - frozenset([v])
                for g in range(self.max_gates):
                    # Any individual gate which is in `a` is also in `b`.
                    self.solver.append_formula(
                        formula.Or(
                            self.var_to_object[(a, g)],
                            formula.Neg(self.var_to_object[(b, g)]),
                        )
                    )
                # There's at least one gate in `a` that's not in `b`.
                self.add_different_gates_constraint(b, a)

    def add_sideways_edge_constraints(self):
        """These define the edge sets, relative to vertex sets.

        These enforce that different sets (of a given size) must
        contain *something* different from each other.
        """
        # Loop through pairs of vertex sets of the same size
        for k in range(self.k, self.n):
            vertex_sets = [a for a in self.big_v if len(a) == k]
            for i in range(len(vertex_sets)):
                for j in range(len(vertex_sets)):
                    if i == j:
                        continue
                    a, b = vertex_sets[i], vertex_sets[j]
                    self.add_different_gates_constraint(a, b)

    def add_sorting_constraints(self):
        """Add constraints that enforce some ordering on the gates.

        This shouldn't change whether there's a solution, but it might
        speed up solving.
        """
        raise NotImplementedError("Sorting constraints not implemented yet.")

    def check_possible(self):
        """Check if it's possible to have this number of gates."""
        return self.solver.solve()


def check_for_lower_bound(n, k, use_downward, use_sideways):
    """Checks for a lower bound of k."""
    print(
        f"Checking for lower bound for n={n}, k={k}, downward={use_downward}, sideways={use_sideways}"
    )
    min_possible_gates = None
    # We know that the number of non-output gates is no more than $\binom{n}{k}$.
    for num_gates in range(1, comb(n, k, exact=True) + 1):
        print(num_gates, end=" ")
        sys.stdout.flush()
        bound = SubsetBound(n, k, max_gates=num_gates)
        if use_downward:
            bound.add_downward_constraints()
        if use_sideways:
            bound.add_sideways_edge_constraints()
        if bound.check_possible():
            min_possible_gates = num_gates
            break
    print(
        f"\nn={n}, k={k}, downward={use_downward}, sideways={use_sideways} -> num. non-output gates >=  {min_possible_gates}\n"
    )
    return min_possible_gates


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Bounds on PARITY-CLIQUE.")
    parser.add_argument("n", type=int, help="Number of vertices")
    parser.add_argument("k", type=int, help="Size of cliques")
    return parser.parse_args()


def main():
    """Main execution entry point."""
    args = parse_args()
    bounds = [
        check_for_lower_bound(
            args.n,
            args.k,
            use_downward,
            use_sideways,
        )
        for use_downward in [False, True]
        for use_sideways in [False, True]
    ]


if __name__ == "__main__":
    main()
