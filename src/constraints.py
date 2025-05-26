from abc import abstractmethod, ABC
from .notesystem import PooledNote as Note


class ConstraintSystem(ABC):
    """An abstract class representing a system of constraints. Child class will implement the abstract functions
    to model specific types of constraints and possibly solve them or approximate them.

    Calling each of the method should add the constraint to the system and return None.

    The .solve method should solve the constraints and return a solution, which is a list of notes that ought to be active

    What are the types of constraints we need to model really?
    -- Cantus Firmus --
    1. Only one note over a certain set can be active
    2. Subsequent notes must form one of the given intervals with the previous note
        - Subsequent notes must move up (down) from the previous note
        - Subsequent notes must form a specific interval
    """
    @abstractmethod
    def only_one_active_note(self, notes: list[Note]) -> None:
        ...

    @abstractmethod
    def subsequent_notes_intervals(self, notes1: list[Note], notes2: list[Note], intervals: list[int]) -> None:
        ...

    @abstractmethod
    def solve(self) -> list[int]:
        """Solve the constraints and return a solution, which is a list of ID of notes that ought to be active."""
        ...


class ILP(ConstraintSystem):
    def __init__(self):
        # A constraint is Ax \le b, where the vector a, x indices and b are modelled
        self._constraints: list[tuple[list[int], list[int], int]] = []
        self._variables: set[int] = set()

    def add_constraint(self, a: list[int], x: list[int], b: int) -> None:
        """Add a constraint of the form Ax <= b."""
        if len(a) != len(x):
            raise ValueError("Length of a and x must match")
        if not all(var in self._variables for var in x):
            # Raise an error instead of updating variables to prevent errors from silently passing
            raise ValueError("All variables in x must be already defined")
        self._constraints.append((a, x, b))

    def add_equal_constraint(self, a: list[int], x: list[int], b: int) -> None:
        """Add a constraint of the form Ax = b."""
        self.add_constraint(a, x, b)
        self.add_constraint([-coeff for coeff in a], x, -b)

    def only_one_active_note(self, notes: list[Note]) -> None:
        """Add a constraint that only one of the given notes can be active."""
        if not notes:
            return
        note_ids = [note._id for note in notes]
        self._variables.update(note_ids)
        self.add_constraint([1] * len(notes), note_ids, 1)

    def subsequent_notes_intervals(self, notes1: list[Note], notes2: list[Note], intervals: list[int]) -> None:
        """Add a constraint that subsequent notes must form one of the given intervals with the previous note.
        Assume note 2 is the set of possible notes that follow note 1. It will be the case where the active note from
        note two form a valid interval with the active note from note one."""
        if not notes1 or not notes2 or not intervals:
            return
        coeffs = {n._id: 0 for n in notes1 + notes2}
        self.only_one_active_note(notes1)
        self.only_one_active_note(notes2)
        # sum_{n1, n2} midi_number_n2 * x2 - midi_number_n1 * x1 = sum_{i} y_i interval_i
        # sum_{i} y_i = 1
        # Since only one of x1 and x2 can be active, all the non-active note components should go to 0
        # y_i are auxiliary variables
        # TODO are the more effective ways of modelling this?
        for n1 in notes1:
            for n2 in notes2:
                coeffs[n2._id] += n2.midi_number
                coeffs[n1._id] -= n1.midi_number
        min_t = min(min(self._variables), 0)
        auxvars = [min_t - i - 1 for i in range(len(intervals))]
        self._variables.update(auxvars)
        for i, interval in enumerate(intervals):
            coeffs[auxvars[i]] = -interval
        xs = sorted(coeffs.keys())
        as_ = [coeffs[x] for x in xs]
        self.add_constraint(as_, xs, 0)
        self.add_constraint([1] * len(auxvars), auxvars, 1)

    def solve(self) -> list[int]:
        """Solve the constraints and return a solution, which is a list of IDs of notes that ought to be active."""
        # This is a stub. In a real implementation, this would use an ILP solver to find the solution.
        # For now, we will just return an empty list.
        return []


class MaxSat(ConstraintSystem):
    def __init__(self):
        self._constraints: list[tuple[list[int], int]] = []  # (variables, weight)
        self._variables: set[int] = set()

    def add_constraint(self, variables: list[int], weight: int) -> None:
        """Add a constraint with a given weight."""
        if not variables:
            return
        self._variables.update(variables)
        self._constraints.append((variables, weight))

    def only_one_active_note(self, notes: list[Note]) -> None:
        """Add a constraint that only one of the given notes can be active."""
        if not notes:
            return
        note_ids = [note._id for note in notes]
        self.add_constraint(note_ids, 1)

    def subsequent_notes_intervals(self, notes1: list[Note], notes2: list[Note], intervals: list[int]) -> None:
        """Add a constraint that subsequent notes must form one of the given intervals with the previous note."""
        if not notes1 or not notes2 or not intervals:
            return
        coeffs = {n._id: 0 for n in notes1 + notes2}
        self.only_one_active_note(notes1)
        self.only_one_active_note(notes2)
        for n1 in notes1:
            for n2 in notes2:
                coeffs[n2._id] += n2.midi_number
                coeffs[n1._id] -= n1.midi_number
        xs = sorted(coeffs.keys())
        as_ = [coeffs[x] for x in xs]
        self.add_constraint(as_, 1)

    def solve(self) -> list[int]:
        """Solve the constraints and return a solution, which is a list of IDs of notes that ought to be active."""
        # This is a stub. In a real implementation, this would use a MaxSat solver to find the solution.
        # For now, we will just return an empty list.
        return []
