from __future__ import annotations
import typing
import re
from dataclasses import dataclass
from fractions import Fraction

StepName = typing.Literal["C", "D", "E", "F", "G", "A", "B"]
_PITCH_NAME_REGEX = re.compile(r"([CDEFGAB])(#+|b+)?(-?[0-9]+)")
PIANO_A0 = 21               # MIDI number for A0
PIANO_C8 = 108              # MIDI number for C8
_LIMIT_DENOMINATOR = 47     # Maximum number for tuplets. Can lift this if needed


@dataclass(frozen=True, unsafe_hash=True, eq=True, slots=True, order=True)
class VariableIndex:
    name: str           # A unique name for which bar/voice/whatever it came from
    duration: int       # Duration as an integer in that how many of these notes are there in a bar of 4/4. Like queavers notes will be 8
    offset: int         # Offset as an index of the ith note of this duration
    index: int          # LOF index of the note
    octave: int         # Octave of the note. 9 > octave >= -1 indicates a valid note and other values indicate something else. Let's reserve 10-19 for us and if someone is using this library for something else they can use it for something else

    @property
    def is_tie(self) -> bool:
        """Check if this variable is a tie variable."""
        return self.octave == 10 and self.index == 1

    @property
    def is_rest(self) -> bool:
        """Check if this variable is a rest variable."""
        return self.octave == 10 and self.index == 0

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

    @property
    def start(self) -> float:
        """Calculate the start position of the note in a bar."""
        return self.offset / self.duration

    @property
    def end(self) -> float:
        """Calculate the end position of the note in a bar."""
        return (self.offset + 1) / self.duration

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


Constraint = tuple[list[int], list[VariableIndex], int]  # Represents the constraints of type either Ax <= b or Cx = d
VoiceRange = range

SOPRANO = VoiceRange(60, 85)
ALTO = VoiceRange(53, 78)
TENOR = VoiceRange(48, 73)
BASS = VoiceRange(40, 65)

KeyName = typing.Literal[
    'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B',
    'C#', 'D#', 'F#', 'G#', 'A#',
]
ModeName = typing.Literal[
    'Major'  # Add minor key support later since I assume we have to do something weird with harmonic/melodic minor
]
