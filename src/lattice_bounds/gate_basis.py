#!/usr/bin/env python3
"""
Number of gates required for various things,
for different sorts of gates.
"""

import numpy as np

# note that comb() returns a float by default;
# for loop bounds, it needs the "exact=True" option,
# so that it returns an int
import scipy.special


def function_counts(num_inputs, max_fan_in=2):
    """Gets counts of functions implementable with some number of gates.

    num_inputs: number of inputs to the circuit
    max_fan_in: maximum number of inputs (None for "unbounded")
    """
    num_gates = 0
    # we assume that with zero gates, we can either
    # "pass through" an input, or implement constants 0 and 1
    num_functions = num_inputs + 2
    # number of inputs to the next gate
    num_gate_inputs = num_inputs + 2
    while True:
        yield (num_gates, num_functions)
        # "pool" of inputs to new gates are the inputs,
        # outputs of previous gates, or 0 or 1
        num_gate_inputs = num_inputs + num_gates + 2
        num_gates += 1
        if max_fan_in:
            num_functions *= scipy.special.comb(num_gate_inputs, max_fan_in, exact=True)
        else:
            num_functions *= 2**num_gate_inputs


def get_expected_num_gates(num_inputs, num_functions, max_fan_in=2):
    """Lower bound on E[ # gates ] to compute some # functions.

    Note that we don't round this up, even though the expected number
    of gates may be slightly higher.
    num_inputs: number of inputs to the circuit
    num_functions: number of functions (as a numpy array)
    max_fan_in: maximum number of inputs (None for "unbounded")
    Returns: a numpy array of the same shape as num_functions,
    such that the i'th element is the expected number of gates
    needed to compute num_functions[i] functions.
    """
    # First, count the number of functions implementable with
    # each number of gates, up until the largest number of functions
    # we need to consider.
    max_functions = num_functions.max()
    total_functions = 0
    num_functions_implementable = []
    for _, num_functions_1 in function_counts(num_inputs, max_fan_in):
        total_functions += num_functions_1
        num_functions_implementable.append(num_functions_1)
        if total_functions > max_functions:
            break
    # Now, for each number of functions, compute the expected number of gates
    # needed to implement it.
    num_functions_implementable = np.array(num_functions_implementable)
    cumu_num_functions_implementable = np.cumsum(num_functions_implementable)
    num_gates = np.arange(len(num_functions_implementable))
    expected_num_gates = (
        num_gates * num_functions_implementable / cumu_num_functions_implementable
    )
    idx = np.searchsorted(cumu_num_functions_implementable, num_functions, side="right")
    return expected_num_gates[idx]


class UnboundedFanInNandBasis:
    """Bounds on number of unbounded-fan-in NAND gates."""

    def __init__(self):
        pass

    def num_functions(self, num_inputs, max_gates):
        """Number of functions implementable with some number of gates.

        This is an upper bound (for various reasons), but should still
            allow lower-bounding the number of gates.
        num_inputs: the number of inputs to the function
        max_gates: the maximum number of gates to include
        Returns: a numpy vector f with shape (max_gates+1,), such that
            f[i] is "number of functions using exactly i gates"
        """
        f = np.full(max_gates + 1, None)
        # We assume that "nothing can be computed with zero gates".
        f[0] = 0
        # Note that this allows a gate with no inputs. This "empty NAND"
        # outputs the constant 0 (since the "empty AND" outputs the constant 1).
        f[1] = 2**num_inputs
        for g in range(2, max_gates + 1):
            f[g] = (2 ** (num_inputs + g - 1)) * f[g - 1]
        return f

    def expected_num_gates(self, num_inputs, num_functions):
        """Lower bound on E[ # gates ] to compute some # functions.

        Note that this won't, in general, be an integer.
        num_inputs: the number of inputs to the circuit
        num_functions: the number of functions (this can be a numpy array)
        Returns: a numpy array of the same shape as num_functions,
        such that the i'th element is the expected number of gates
        needed to compute num_functions[i] functions.
        """
        return get_expected_num_gates(num_inputs, num_functions, max_fan_in=None)

    def expected_gates_1(self, num_inputs, lg_num_functions):
        """Lower bound on E[ # gates ] to compute some # functions.

        This is similar to expected_gates(), but it's a bit faster.
        Note that this won't, in general, be an integer.
        num_inputs: the number of inputs to the circuit
        lg_num_functions: the number of functions (on log_2 scale);
            this can be an np.array
        """
        b = num_inputs - 0.5
        # the "-1" here is because this is the average, not the max.
        g = np.sqrt(2 * lg_num_functions + b * b) - b - 1
        return g

    def and_upper_bound(self):
        """Upper bound on computing AND of two functions."""
        return 2

    def or_upper_bound(self):
        """Upper bound on computing OR of two functions.

        If A and B are the sets of cliques, then this is
        relative to the sum of the sizes of the two
        corresponding circuits.
        """
        # For unbounded fan-in, I think we save one gate (because we can
        # combine all the wires going into each output gate, feed them into
        # one of the output gates, and discard the other)
        return -1

    def not_upper_bound(self):
        """Upper bound on computing NOT."""
        return 1

    def or_of_and_upper_bound(self, _num_and, num_or):
        """Upper bound of ORing some ANDs together.

        This computes the AND of `_num_and` inputs, `num_or` times, and
        then ORs the results together.
        """
        # Because of unbounded fan-in, we can use NAND gates to compute
        # the ANDs, and then compute the OR using a final NAND gate.
        return num_or + 1

    def xor_of_and_upper_bound(self, _num_and, num_xor):
        """Upper bound of XORing some ANDs together.

        This computes the AND of `_num_and` inputs, `num_xor` times, and
        then XORs the results together.
        """
        # Here, we can use two NAND gates per AND. The second NAND gate
        # will be an inverter, which we can use one of in computing XOR.
        # I think this means we need an additional three NAND gates per XOR.
        # ??? can we do better?
        return 2 * num_xor + 3 * num_xor - 1

    def xor_upper_bound(self):
        """Upper bound on computing XOR of two functions."""
        # unbounded fan-in doesn't help here
        return 4

    def zonked_gates(self):
        """Number of gates zonked by feeding in a zero to one vertex."""
        # ??? is this needed?
        return 1.0


class TwoInputNandBasis:
    """Bounds on number of two-input NAND gates.

    ??? include basis with two-input "any function" gates?
    """

    def __init__(self):
        """Constructor."""

    def num_functions(self, num_inputs, max_gates):
        """Number of functions implementable with some number of gates.

        Again, this is an upper bound.
        This is almost the counting described in
            Aaronson, Scott. P ?= NP, page 44,
            but is slightly different.

        num_inputs: the number of inputs to the function
        max_gates: the maximum number of gates to include
        Returns: a numpy vector f with shape (max_gates+1,), such that
            f[i] is "number of functions using exactly i gates"
        """
        f = np.full(max_gates + 1, None)
        # With zero gates, we assume we can implement the constant 0.
        f[0] = 1
        for g in range(1, max_gates + 1):
            # each additional gate can use:
            # - an input
            # - a previous gate
            # - the constant 1 (thus changing a two-input NAND to a NOT)
            f[g] = (
                scipy.special.comb(num_inputs + (g - 1) + 1, 2, exact=True) * f[g - 1]
            )
        return f

    def expected_num_gates(self, num_inputs, num_functions):
        """Lower bound on E[ # gates ] to compute some # functions.

        Note that this will be an integer (although the expected value
        may be slightly higher.)
        num_inputs: the number of inputs to the circuit
        num_functions: the number of functions (this can be a numpy array)
        Returns: a numpy array of the same shape as num_functions,
        such that the i'th element is the expected number of gates
        needed to compute num_functions[i] functions.
        """
        return get_expected_num_gates(num_inputs, num_functions, max_fan_in=2)

    def and_upper_bound(self):
        """Upper bound on computing AND of two functions."""
        # we can use a NAND, and then invert the output
        return 2

    def or_upper_bound(self):
        """Upper bound on computing OR of two functions.

        Returns: upper bound on number of gates to compute OR of two circuits,
            relative to their sum.
        """
        # invert the two inputs, then use a NAND
        return 3

    def not_upper_bound(self):
        """Upper bound on negating a function."""
        return 1

    def or_of_and_upper_bound(self, num_and, num_or):
        """Upper bound of ORing some ANDs together.

        This computes the AND of `num_and` inputs, `num_or` times, and
        then ORs the results together.
        """
        gates_for_and = 2 * (num_and - 1)
        # ??? I think that, since the AND circuits all end with inverters,
        # we can just drop the final inverters, and use NAND gates to compute OR ?
        return num_or * gates_for_and + (num_or - 1)

    def xor_of_and_upper_bound(self, num_and, num_xor):
        """Upper bound of XORing some ANDs together.

        This computes the AND of `num_and` inputs, `num_xor` times, and
        then XORs the results together.
        """
        gates_for_and = 2 * (num_and - 1)
        # ??? Again, since the AND circuits all end with inverters,
        # we can use the final inverters as part of the XOR circuit ?
        return num_xor * gates_for_and + 2 * (num_xor - 1)

    def xor_upper_bound(self):
        """Upper bound on computing XOR of two functions."""
        return 4

    def zonked_gates(self):
        """Number of gates zonked by feeding in a zero to one vertex."""
        # ??? is this needed?
        return 1.0
