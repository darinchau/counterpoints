from abc import abstractmethod, ABC
from .notesystem import PooledNote as Note
from sympy import Symbol


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

    def __init__(self):
        self._variables: set[int] = set()

    @property
    def variables(self):
        return frozenset(self._variables)

    def add_variables(self, variables: list[int]):
        if not variables:
            return
        self._variables.update(variables)

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
