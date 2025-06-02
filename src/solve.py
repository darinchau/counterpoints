import copy
import math
from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, lpDot, getSolver, LpSolver
from .indices import VariableIndex, Constraint, Solution, System
from .notesystem import NoteSystem


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
        Returns None if no solution is found. This method will solve the entire system at once"""
        # Add constraints to the current system that excludes specific solutions
        system = self._ns.get_system()
        system = _prohibit_solution(system, self._found_solutions)
        sol = _solve(system, self._solver)
        if sol is None:
            return None
        assert set(sol[0] + sol[1]) == set(system[0]), "The solution does not contain all variables in the system."
        self._found_solutions.append(sol)
        return sol

    def solve_iterative(self):
        """Finds a new solution, if any. Appends it to the list of found solutions, and return it.
        Returns None if no solution is found. This method will solve the system bar by bar, backtracking
        when necessary. For longer systems, this method is probably more efficient than `solve()`."""
        system = self._ns.get_system()
        system = _prohibit_solution(system, self._found_solutions)
        vs, ineq, eq = system
        max_bar_number = max(v.bar_number for v in vs)
        min_bar_number = min(v.bar_number for v in vs)
        if min_bar_number < 0:
            raise ValueError(f"Variables cannot be indexed by bar number: found negative bar number {min_bar_number}. ")
        bar_vs = [[] for _ in range(max_bar_number + 1)]
        bar_eqs = [[] for _ in range(max_bar_number + 1)]
        bar_ineqs = [[] for _ in range(max_bar_number + 1)]
        for v in vs:
            bar_vs[v.bar_number].append(v)
        for c in eq:
            bar_eqs[_constraint_bar_number(c)].append(c)
        for c in ineq:
            bar_ineqs[_constraint_bar_number(c)].append(c)
        solutions: list[Solution] = []
        sol = _solve_bar_solution(solutions, bar_vs, bar_ineqs, bar_eqs, self._solver)
        if sol is None:
            return None
        assert set(sol[0] + sol[1]) == set(vs), "The solution does not contain all variables in the system."
        self._found_solutions.append(sol)
        return sol


def _constraint_bar_number(c: Constraint) -> int:
    """Return the bar number that this constraint restricts to. This is the largest bar number
    of any variable that appears in the constraint"""
    return max(v.bar_number for v in c[1]) if c[1] else -1


def _restrict_system(system: System, solution: Solution) -> System:
    """Restricts the system to the given solution. Returns a new system with the restrictions applied."""
    _, ineq, eq = system
    new_ineq: list[Constraint] = []
    new_eq: list[Constraint] = []
    varset: set[VariableIndex] = set()
    solution_vars = set(solution[0] + solution[1])
    for c in ineq:
        a, x, b = c
        if not all(v in solution_vars for v in x):
            new_ineq.append(c)
            varset.update(x)
    for c in eq:
        a, x, b = c
        if not all(v in solution_vars for v in x):
            new_eq.append(c)
            varset.update(x)
    n_restricted = 0
    for v in solution[0]:
        if v in varset:
            new_eq.append(([1], [v], 0))
            n_restricted += 1
    for v in solution[1]:
        if v in varset:
            new_eq.append(([1], [v], 1))
            n_restricted += 1
    return System(list(varset), new_ineq, new_eq)


def _prohibit_solution(system: System, solutions: list[Solution]) -> System:
    vs, ineq, eq = system
    new_constraints: list[Constraint] = []
    for zeros, ones in solutions:
        a = [1] * len(ones) + [-1] * len(zeros)
        x = ones + zeros
        b = len(ones) - 1
        new_constraints.append((a, x, b))
    system = System(vs, ineq + new_constraints, eq)
    return system


def _solve_bar_solution(
    solutions: list[Solution],
    bar_vs: list[list[VariableIndex]],
    bar_ineqs: list[list[Constraint]],
    bar_eqs: list[list[Constraint]],
    solver: LpSolver | None = None
) -> Solution | None:
    """Solves the ILP modelled by the NoteSystem for a specific bar."""
    def join_solution(sols: list[Solution]) -> Solution:
        """Joins the solutions into a single solution."""
        s0: set[VariableIndex] = set()
        s1: set[VariableIndex] = set()
        for (sol0, sol1) in sols:
            s0.update(sol0)
            s1.update(sol1)
        return list(s0), list(s1)
    assert len(bar_vs) == len(bar_ineqs) == len(bar_eqs), "The number of bars does not match the number of groups of inequalities and equations."
    if len(solutions) == len(bar_vs):
        return join_solution(solutions)
    bar_i = len(solutions)
    bad_solutions: list[Solution] = []
    variables: list[VariableIndex] = []
    for v in bar_vs[:bar_i + 1]:
        variables.extend(v)
    join_s = join_solution(solutions)
    system = System(variables, bar_ineqs[bar_i], bar_eqs[bar_i])
    system = _restrict_system(system, join_s)
    while True:
        sys_no_bad_sols = _prohibit_solution(system, bad_solutions)
        sol = _solve(sys_no_bad_sols, solver)
        if sol is None:
            return None
        solutions.append(sol)
        sol_ = _solve_bar_solution(solutions, bar_vs, bar_ineqs, bar_eqs, solver)
        if sol_ is not None:
            return sol_
        solutions.pop()
        bad_solutions.append(sol)


def _solve(system: System, solver: LpSolver | None = None) -> Solution | None:
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
    sol: Solution = ([], [])
    for var in variables:
        val = variables[var].varValue
        if val is None:
            return None

        if math.isclose(val, 1.0):
            sol[1].append(var)
        elif math.isclose(val, 0.0):
            sol[0].append(var)
        else:
            raise ValueError(f"Variable {var} has a value of {val}, but it should be binary (0 or 1).")
    return sol
