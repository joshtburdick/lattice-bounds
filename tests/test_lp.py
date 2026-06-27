"""
Basic test of LP solver (and wrapper).

Here, we want to make sure the LP solver can handle large
integer coefficients. (The wrapper will convert fractions
to integers by multiplying by common denominator).

We aren't as worried about getting results at high precision.
"""

from lattice_bounds import pulp_helper


def test_lp_with_large_coefficients():
    """Tests that the LP solver can handle large integer coefficients."""
    lp = pulp_helper.PulpHelper(["x1", "x2"])
    lp.add_constraint([("x1", 2), ("x2", 1)], ">=", 1)
    # FIXME(jtb): currently, PuLP's default GLPK solver doesn't handle
    # super-large integer coefficients well. (Admittedly, it can deal
    # with 255-digit integers, but that doesn't seem sufficient.)
    # Once that's fixed, this should use a larger coefficient
    # (e.g., 10**10000).
    lp.add_constraint([("x1", 2), ("x2", 1)], ">=", 1)
    lp.add_constraint([("x1", 1), ("x2", 10**10)], ">=", 1)
    result = lp.solve_with_objective({"x1": 1, "x2": 1})

    assert result
    assert abs(result["__objective__"] - 0.5) < 1e-9
