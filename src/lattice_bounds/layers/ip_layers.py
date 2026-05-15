#!/usr/bin/env python3
# LP bound for clique parity, with layers.
# FIXME
# - The "zeroing" bound should use
#   "sets with _up to_ some number of vertices";
#   add variables for that, and change the bound
#   to be slightly < 1
# - Change levels to be "low" and "high"?
# - Allow switching basis?

import argparse
import fractions
import itertools
import math
import pdb
import sys

sys.path.append("..")  # XXX
sys.path.append("../..")  # XXX

import numpy as np
import pandas
import scipy.special
import scipy.stats

import gate_basis
import flexible_lp_helper
import pulp_helper
import scip_helper

# import exact_simplex_helper
import simplex_algorithm_helper
import hypergraph_counter


# Wrapper for comb(), with exact arithmetic.
def comb(n, k):
    return scipy.special.comb(n, k, exact=True)


def group_by_key(items, key_func):
    """This groups items by a key function.

    items: iterable of items to group
    key_func: function to compute key for each item

    Returns a dictionary mapping keys to lists of items.
    """
    result = {}
    for item in items:
        key = key_func(item)
        if key not in result:
            result[key] = []
        result[key].append(item)
    return result


def flatten(nested_list):
    return list(itertools.chain.from_iterable(nested_list))


class IpLayers:
    """Attempt at bound for "layers" of the clique problem."""

    def __init__(self, n, k, max_gates, num_layers):
        """Constructor gets graph info, and sets up variable names.

        n: number of vertices in the graph
        k: number of vertices in a clique (>= 3)
        max_gates: maximum number of gates to consider (if this is too small,
            the LP will be infeasible)
        num_layers: If this is an integer, the number of layers to use.
            For now, this should be odd.
        """
        self.n = n
        self.k = k
        self.max_gates = max_gates
        self.num_layers = num_layers

        if k < 3:
            raise ValueError("k must be >= 3")

        # Number of possible cliques
        self.num_possible_cliques = comb(n, k)

        # The layers, as tuples (a, b), which represent the interval [a, b).
        self.layers = self.make_layers(num_layers)

        # Number of functions for each group (num. vertices, layer).
        self.counts_by_group = self.get_counts_by_group()

        # The variables for the LP.
        vars = []
        # Loop trough the (num. vertices, layer) pairs.
        for v, layer in self.counts_by_group:
            # Expected number of gates, for each group.
            vars += [("E", v, layer)]
            # Counts of functions with some number of gates.
            for g in range(self.max_gates + 1):
                vars += [(v, layer, g)]
        # Averages, by number of vertices, and layer
        for v in range(self.k, self.n + 1):
            vars += [("V", v)]
        for l1, l2 in self.layers:
            vars += [("L", l1)]
        # pdb.set_trace()
        # wrapper for LP solver
        self.lp = pulp_helper.PuLP_Helper(vars)
        # basis for gates
        self.basis = gate_basis.TwoInputNandBasis()
        # for debugging: directory in which to save LP problem files
        self.lp_save_dir = None

    def make_layers(self, num_layers):
        """Makes the layers for the LP.

        We divide the possible number of cliques into layers, and add
        constraints on the average number of gates for each layer.
        We require the layers to be symmetric around N/2.
        (The layers won't, in general, be the same height; but this
        tries to make them as similar as possible.)

        FIXME: the "layers balanced about N/2" may not actually be necessary;
        it would be nice to simplify this.

        num_layers: currently must be odd.
        """
        endpoints = (
            np.linspace(0, self.num_possible_cliques + 1, num_layers + 1)
            .astype(int)
            .tolist()
        )
        return list(zip(endpoints[:-1], endpoints[1:]))

    def get_counts_by_group(self):
        """Gets number of functions, for each possible (num. vertices, layer).

        Returns a dictionary mapping (num. vertices, layer) to number of functions.
        Note that many of the groups will be empty, and so this will not include
        all possible (num. vertices, layer) pairs.
        """
        counter = hypergraph_counter.HypergraphCounter(self.n, self.k)
        counts_by_num_vertices = counter.count_hypergraphs_exact_vertices()
        counts_by_group = {}
        for v in range(self.k, self.n + 1):
            for l1, l2 in self.layers:
                if l1 <= len(counts_by_num_vertices[v]):
                    counts_by_group[(v, l1)] = counts_by_num_vertices[v][l1:l2].sum()
        # pdb.set_trace()
        # check that these add up to the total number of functions
        # FIXME for now, we omit the empty function
        assert sum(counts_by_group.values()) == 2**self.num_possible_cliques - 1
        return counts_by_group

    def add_averaging_constraints(self):
        """Adds constraints on the average number of gates for each group."""
        for v, layer in self.counts_by_group:
            A = [((v, layer, g), g) for g in range(self.max_gates + 1)]
            # "expected number of gates" = sum(counts * gates) / sum(counts)
            self.lp.add_constraint(
                [(("E", v, layer), -self.counts_by_group[(v, layer)])] + A, "=", 0
            )
            # add constraints on the number of functions in each group
            self.lp.add_constraint(
                [((v, layer, g), 1) for g in range(self.max_gates + 1)],
                "=",
                self.counts_by_group[(v, layer)],
            )

    def add_marginal_constraints(self):
        """Adds marginal constraints on average (by num. vertices, and layer)"""
        # marginalize over layers, for each number of vertices
        counts_by_v = group_by_key(self.counts_by_group.items(), lambda x: x[0][0])
        for v, counts1 in counts_by_v.items():
            # pdb.set_trace()
            A = [(("E", v, layer), w) for (v, layer), w in counts1]
            total_w = sum([w for _, w in counts1])
            self.lp.add_constraint([(("V", v), -total_w)] + A, "=", 0)
        # marginalize over vertices, for each layer
        counts_by_layer = group_by_key(self.counts_by_group.items(), lambda x: x[0][1])
        for layer, counts1 in counts_by_layer.items():
            A = [(("E", v, layer), w) for (v, layer), w in counts1]
            total_w = sum([w for _, w in counts1])
            self.lp.add_constraint([(("L", layer), -total_w)] + A, "=", 0)

    def add_counting_bounds(self):
        """Adds counting bounds, for a given number of gates."""
        # first, compute number of possible functions
        # for each number of gates
        num_possible_functions = self.basis.num_functions(comb(self.n, 2), max_gates)
        for g in range(self.max_gates + 1):
            self.lp.add_constraint(
                [((v, layer, g), 1) for v, layer in self.counts_by_group],
                "<=",
                num_possible_functions[g],
            )

    def add_zeroing_bound(self):
        """Adds bound from zeroing out one vertex."""
        for v in range(self.k, self.n):
            # "at least one NAND gate was hit"
            self.lp.add_constraint([(("V", v + 1), 1), (("V", v), -1)], ">=", 1)

    def get_all_bounds(self):
        """Gets bounds for each possible number of cliques.

        This is the bounds for each possible number of cliques,
        in the scenario that the number of gates for
        CLIQUE is minimized.
        """
        # solve, minimizing number of gates in "highest layer"
        r = self.lp.solve(("L", self.layers[-1][0]))
        if not r:
            return None
        # Get bounds for "expected number of gates" for functions in
        # each layer. To simplify plotting, we include the two endpoints
        # of each layer.
        n_cliques = flatten([[l1, l2 - 1] for l1, l2 in self.layers])
        bounds = flatten([[r[("L", l1)]] * 2 for l1, l2 in self.layers])
        return pandas.DataFrame(
            {
                "Num. levels": self.num_layers,
                "Num. cliques": n_cliques,
                "Min. gates": bounds,
            }
        )


def get_bounds(n, k, max_gates, num_layers):
    """Gets bounds with some set of constraints.

    For now, just uses all the constraints.
    Args:
        n, k, max_gates: problem size
        num_layers: number of layers
    """
    # ??? track resource usage?
    sys.stderr.write(
        f"[bounding with n={n}, k={k}, max_gates={max_gates}, num_layers={num_layers}]\n"
    )
    bound = IpLayers(n, k, max_gates, num_layers)
    bound.add_averaging_constraints()
    bound.add_marginal_constraints()
    bound.add_counting_bounds()
    bound.add_zeroing_bound()

    b = bound.get_all_bounds()
    return b


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bounds circuit size for detecting subsets of cliques."
    )
    parser.add_argument("n", type=int, help="Number of vertices")
    parser.add_argument("k", type=int, help="Size of clique")
    parser.add_argument(
        "max_gates", type=int, help="Maximum number of gates to consider"
    )
    parser.add_argument(
        "num_layers",
        type=str,
        help="Number of layers, which can be a comma-separated list of integers (e.g. '10' or '10,12,14')",
    )
    parser.add_argument(
        "--dump-lp", help="Dump LP problem statement to a file", action="store_true"
    )
    parser.add_argument(
        "--max-gates-search",
        help="Search for smallest number of gates which doesn't crash",
        action="store_true",
    )
    parser.add_argument(
        "--result-file", help="Write result to indicated file (rather than stdout)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n = args.n
    k = args.k
    num_layers = args.num_layers
    max_gates = args.max_gates

    bounds = [
        get_bounds(n, k, max_gates, int(num_layers))
        for num_layers in args.num_layers.split(",")
    ]
    bounds = pandas.concat(bounds)
    if args.result_file:
        with open(args.result_file, "wt") as f:
            bounds.to_csv(f, index=False)
    else:
        bounds.to_csv(sys.stdout, index=False)
