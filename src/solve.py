import copy
import math
from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, lpDot, getSolver, LpSolver
from .indices import VariableIndex, Constraint, Solution, System
from .notesystem import NoteSystem


def solve(system: System, solver: LpSolver | None = None) -> Solution | None:
    """Solves the ILP modelled by the NoteSystem. Returns True if the system is solvable (according to the solver)"""
    vs, ineq, eq = system
    prob = LpProblem("", LpMinimize)
    variables = {x: LpVariable(str(x), cat="Binary") for x in vs}
    for a, x, b in ineq:
        xv = [variables[s] for s in x]
        prob += lpDot(a, xv) <= b

    for c, x, d in eq:
        xv = [variables[s] for s in x]
        prob += lpDot(c, xv) == d

    status = prob.solve(solver)
    if status != 1:
        return

    # Log the current solution
    sol: Solution = []
    for var in variables:
        val = variables[var].varValue
        if val is not None and not var.aux:
            if not (math.isclose(val, 0.0) or math.isclose(val, 1.0)):
                raise ValueError(f"Variable {var} has a value of {val}, but it should be binary (0 or 1).")
            if math.isclose(val, 1.0):
                sol.append(var)
    return sol


class NoteSystemSolver:
    """Solves the ILP modelled by the NoteSystem"""

    def __init__(self, ns: NoteSystem, solver: str = "PULP_CBC_CMD"):
        self._ns = copy.deepcopy(ns)
        self._found_solutions: list[Solution] = []
        self._solver = getSolver(solver, msg=False)

    def clear(self):
        """Clears the list of found solutions."""
        self._found_solutions.clear()

    def get_solutions(self) -> list[Solution]:
        """Returns the list of found solutions."""
        return copy.deepcopy(self._found_solutions)

    def solve(self):
        """Finds a new solution, if any. Appends it to the list of found solutions, and return it.
        Returns None if no solution is found."""
        # Add constraints to the current system that excludes specific solutions
        vs, ineq, eq = self._ns.get_system()
        new_constraints: list[Constraint] = []
        for s in self._found_solutions:
            ones = s
            zeros = [v for v in vs if v not in ones]
            a = [1] * len(ones) + [-1] * len(zeros)
            x = ones + zeros
            b = len(ones) - 1
            new_constraints.append((a, x, b))
        system = (vs, ineq + new_constraints, eq)
        sol = solve(system, self._solver)
        if sol is None:
            return None
        self._found_solutions.append(sol)
        return sol
