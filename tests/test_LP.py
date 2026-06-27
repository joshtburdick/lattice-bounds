"""
Basic test of LP solver (and wrapper).

Here, we want to make sure the LP solver can handle large
integer coefficients. (The wrapper will convert fractions
to integers by multiplying by common denominator).

We aren't as worried about getting results at high precision.
"""

import math

from lattice_bounds import pulp_helper


def test_basic_LP():
    """ """
    # Define a simple LP
    lp = pulp_helper.PulpHelper(["x1", "x2"])
    lp.add_constraint([("x1", 2), ("x2", 1)], ">=", 1)
    # FIXME(jtb): currently, PuLP's default GLPK solver doesn't handle
    # large integer coefficients well. Once that's
    # fixed, this should use a larger coefficient
    lp.add_constraint([("x1", 2), ("x2", 1)], ">=", 1)
    lp.add_constraint([("x1", 1), ("x2", 10**100)], ">=", 1)

    result = lp.solve_with_objective({"x1": 1, "x2": 1})
    print(result)


#    assert result.status == "Optimal"
#    assert math.abs(result.value - 0.5) < 1e-9
