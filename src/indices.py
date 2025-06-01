from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from functools import total_ordering, cached_property
from .note import midi_number_from_index_octave, Note
from .consts import _LIMIT_DENOMINATOR


@dataclass(frozen=True, unsafe_hash=True, eq=True)
@total_ordering
class VariableIndex:
    name: str
    # A unique name for which bar/voice/whatever it came from. Useful purely for indexing and bookkeeping

    duration: int
    # Duration as an integer in that how many of these notes are
    # there in a bar of 4/4. Like queavers notes will be 8

    offset: int
    # Offset as an index of the ith note of this duration

    index: int
    # LOF index of the note

    octave: int
    # Octave of the note. -1 <= octave <= 9 indicates a valid note and other values indicate something else.
    # Let's reserve 10-19 for us and if someone is using this library for something
    # else they can use other numbers for something else

    aux: bool = False

    @property
    def is_tie(self) -> bool:
        """Check if this variable is a tie variable."""
        return self.octave == 10 and self.index == 1

    @property
    def is_rest(self) -> bool:
        """Check if this variable is a rest variable."""
        return self.octave == 10 and self.index == 0

    @property
    def is_note(self) -> bool:
        """Check if this variable is a note variable."""
        return -1 <= self.octave <= 9 and not self.aux

    @classmethod
    def make_tie(cls, name: str, duration: int, offset: int) -> VariableIndex:
        """Create a tie variable."""
        return cls(name, duration, offset, 1, 10)

    @classmethod
    def make_rest(cls, name: str, duration: int, offset: int) -> VariableIndex:
        """Create a rest variable."""
        return cls(name, duration, offset, 0, 10)

    def get_tie(self) -> VariableIndex:
        """Get the tie variable for this note."""
        if self.is_tie:
            return self
        return VariableIndex.make_tie(self.name, self.duration, self.offset)

    def get_rest(self) -> VariableIndex:
        """Get the rest variable for this note."""
        if self.is_rest:
            return self
        return VariableIndex.make_rest(self.name, self.duration, self.offset)

    def get_note(self) -> Note:
        return Note(
            self.index,
            self.octave,
            Fraction(4, self.duration),
            Fraction(4 * self.offset, self.duration),
            64
        )

    @cached_property
    def start(self):
        """Calculate the start position of the note in a bar."""
        return Fraction(self.offset, self.duration).limit_denominator(_LIMIT_DENOMINATOR)

    @cached_property
    def end(self):
        """Calculate the end position of the note in a bar."""
        return Fraction(self.offset + 1, self.duration).limit_denominator(_LIMIT_DENOMINATOR)

    @cached_property
    def midi_number(self) -> int:
        return midi_number_from_index_octave(self.index, self.octave)

    def __repr__(self) -> str:
        """String representation of the VariableIndex."""
        from .note import duration_to_str, Note
        var_type = "aux"
        if self.is_tie:
            var_type = "tie"
        elif self.is_rest:
            var_type = "rest"
        elif -1 <= self.octave <= 9:
            n = Note(self.index, self.octave, Fraction(1), Fraction(1), 1)
            var_type = f"{n.pitch_name}{n.octave}"
        return f"VariableIndex(bar {self.name}, {self.offset}-th {duration_to_str(Fraction(4, self.duration))} note, {var_type})"

    def __lt__(self, other: VariableIndex) -> bool:
        sp = (self.start, self.end, self.midi_number, self.index, self.name)
        op = (other.start, other.end, other.midi_number, other.index, other.name)
        return sp < op


Constraint = tuple[list[int], list[VariableIndex], int]  # Represents the constraints of type either Ax <= b or Cx = d
System = tuple[list[VariableIndex], list[Constraint], list[Constraint]]  # Represents the variables, system of equations and inequalities to solve
Solution = list[VariableIndex]  # Represent the list of variables that ought to be 1. Does not include aux variables
