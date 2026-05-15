#!/usr/bin/env python3
# Convenient interface (hopefully) for use with PuLP.

import fractions
import math
import os
import pdb
import re
import sys
import subprocess
import tempfile

import functools
import numpy as np

import pulp


class PuLP_Helper:
    """Wrapper class for PuLP solver, providing convenient variable names."""

    def __init__(self, var_names):
        """Constructor.

        var_names: names of the variables
        """
        # mapping from variable name (which needn't be a string)
        # to PuLP variable object
        self.vars = {}
        # the variables are assumed to all be >= 0
        for v in var_names:
            self.vars[v] = pulp.LpVariable(self.get_parseable_name(v), 0)
        # the problem
        self.prob = pulp.LpProblem("sidsproblem", pulp.LpMinimize)

    def get_parseable_name(self, var_name):
        """Makes a 'parseable' version of a variable name."""
        s = str(var_name)
        s = re.sub("[ ']", "", s)
        s = re.sub("[\\.,\\(\\)]", "_", s)
        return "x_" + s

    def add_constraint(self, A, op, b):
        """Adds one row to the problem.

        A: a list of (x, a) pairs, where:
            a is the coefficient
            x is the variable, which is a key in self.vars
        (XXX arguably "(a, x)" would make sense, but I used "(x, a)"
            elsewhere in the code, so using that)
        op: either '<=', '==', or '>=': the type of constraint
            (??? change "=" to "==" ?)
        b: the corresponding bound
        Side effects: adds the constraint
        """
        if op not in ["<=", "=", ">="]:
            raise ValueError(f"unknown operator: {op}")
        # get least common multiple of denominators of A and b
        coefs = [fractions.Fraction(a) for (_, a) in A]
        denominators = [fractions.Fraction(x).denominator for x in coefs + [b]]
        lcm = functools.reduce(math.lcm, denominators)
        # convert coefficients to format PuLP expects, multiplying
        # by LCM (so that, hopefully, all coefficients are integers)
        A_as_expr = pulp.lpSum([(lcm * a) * self.vars[x] for (x, a) in A if a != 0])
        # also multiply b by the LCM
        if op == "<=":
            self.prob += A_as_expr <= lcm * b
        if op == "=":
            self.prob += A_as_expr == lcm * b
        if op == ">=":
            self.prob += A_as_expr >= lcm * b

    def solve_1(self, var_to_minimize):
        """solves the linear system, for one variable.

        this assumes all variables are >= 0.
        fixme add option to make all variables integers?
        """
        self.prob += self.vars[var_to_minimize]
        # for debugging
        self.prob.writelp("./bound.lp")
        r = self.prob.solve(pulp.glpk(options=["--exact"]))
        # problem had a solution
        if r == 1:
            return self.vars[var_to_minimize].varValue
        # problem was infeasible, or something else went wrong
        else:
            return None

    def solve(self, var_to_minimize, bounds=None):
        """Solves the linear system.

        objective: what to minimize (as a Numpy vector)
        var_to_minimize: name of the variable to minimize
        bounds: bounds on individual variables
        Returns: a dict, indexed by variable name, of
            all the variables, at the lower bound
        """
        self.prob += self.vars[var_to_minimize]
        self.prob.writeLP("./bound.lp")
        r = self.prob.solve(pulp.GLPK("glpsol", options=["--exact"]))
        # r = self.prob.solve(pulp.GLPK())
        # problem had a solution
        print(f"Result r = {r}")
        if r == 1:
            opt = {x: self.vars[x].varValue for x in self.vars}
            return opt
        # problem was infeasible, or something else went wrong
        else:
            return None

    def solve_with_objective(self, objective):
        """Solves the linear system, with a multivariable objective

        objective: linear function to minimize (as a dict, indexed
            by variable name, and coefficients as values)
        Returns: a dict, indexed by variable name, of
            all the variables, at the lower bound.
            (Also includes the objective value, as "__objective__".)
        """
        objective_function = 0
        for x, a in objective.items():
            objective_function += a * self.vars[x]
        self.prob += objective_function

        self.prob.writeLP("./bound.lp")
        # r = self.prob.solve(pulp.GLPK("glpsol",
        #                              options=['--exact']))
        # trying without the "exact" option
        r = self.prob.solve(pulp.GLPK("glpsol"))

        print(f"Result r = {r}")

        # did problem have a solution?
        if r == 1:
            opt = {x: self.vars[x].varValue for x in self.vars}
            opt["__objective__"] = pulp.pulp.value(self.prob.objective)
            return opt
        # problem was infeasible, or something else went wrong
        else:
            return None
