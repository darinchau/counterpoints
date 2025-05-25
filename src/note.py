from __future__ import annotations
import math
import re
from dataclasses import dataclass, asdict
from functools import reduce
from typing import Literal
from fractions import Fraction

StepName = Literal["C", "D", "E", "F", "G", "A", "B"]
_PITCH_NAME_REGEX = re.compile(r"([CDEFGAB])(#+|b+)?(-?[0-9]+)")
PIANO_A0 = 21               # MIDI number for A0
PIANO_C8 = 108              # MIDI number for C8
_LIMIT_DENOMINATOR = 47     # Maximum number for tuplets. Can lift this if needed


@dataclass(frozen=True)
class Note:
    """A piano note is a representation of a note on the piano, with a note name and an octave
    The convention being middle C is C4. The lowest note is A0 and the highest note is C8.

    If the note is in real time, then the duration and offset is timed with respect to quarter length,
    otherwise it is timed with respect to real-time seconds.

    Attributes:
        index (int): The index of the note in the LOF (Line of Fifths) scale.
        octave (int): The octave of the note, where middle C is C4.
        duration (Fraction): The duration of the note in quarter length
        offset (Fraction): The offset of the note in quarter length
        velocity (int): The velocity of the note, where 0 is silent and 127 is the loudest."""
    index: int
    octave: int
    duration: Fraction
    offset: Fraction
    velocity: int

    def __post_init__(self):
        # Sanity Check
        assert PIANO_A0 <= self.midi_number <= PIANO_C8, f"Note must be between A0 and C8, but found {self.midi_number}"
        assert self.duration >= 0, f"Duration must be greater than or equal to 0, but found {self.duration}"
        assert self.offset >= 0, f"Offset must be greater than or equal to 0, but found {self.offset}"
        assert 0 <= self.velocity < 128, f"Velocity must be between 0 and 127, but found {self.velocity}"

    def __repr__(self):
        return f"Note({self.note_name})"

    @property
    def pitch_name(self) -> str:
        """Returns a note name of the pitch. e.g. A, C#, etc."""
        alter = self.alter
        if alter == 0:
            return self.step
        elif alter == 2:
            return f"{self.step}x"
        elif alter > 0:
            return f"{self.step}{'#' * alter}"
        else:
            return f"{self.step}{'b' * -alter}"

    @property
    def note_name(self):
        """The note name of the note. e.g. A4, C#5, etc."""
        return f"{self.pitch_name}{self.octave}[{self.duration_name}]"

    @property
    def duration_name(self):
        """Returns the duration of the note in a compact notation

        'w' for whole note, 'h' for half note, 'q' for quarter note,
        'r' for eighth note, 's' for sixteenth note

        Each apostrophe represents twice or half the length, so 'w'' is a breve (double whole note),
        or 's'' is a thirty-second note.

        '.' for dotted notes

        If the duration is a tuplet, the tuplet number is included as a suffix.

        If the note is a tied note, it is joined with '+'
        """
        return duration_to_str(self.duration)

    @property
    def step(self) -> StepName:
        """Returns the diatonic step of the note"""
        idx = self.index % 7
        return ("C", "G", "D", "A", "E", "B", "F")[idx]

    @property
    def step_number(self) -> int:
        """Returns the diatonic step number of the note, where C is 0, D is 1, etc."""
        idx = self.index % 7
        return (0, 4, 1, 5, 2, 6, 3)[idx]

    @property
    def alter(self):
        """Returns the alteration of the note aka number of sharps. Flats are represented as negative numbers."""
        return (self.index + 1) // 7

    @property
    def pitch_number(self):
        """Returns the chromatic pitch number of the note. C is 0, D is 2, etc. There are edge cases like B# returning 12 or Cb returning -1"""
        return ([0, 2, 4, 5, 7, 9, 11][self.step_number] + self.alter)

    @property
    def midi_number(self):
        """The chromatic pitch number of the note, using the convention that A4=440Hz converts to 69
        This is also the MIDI number of the note."""
        return self.pitch_number + 12 * self.octave + 12

    def transpose(self, interval: int, compound: int = 0) -> Note:
        """Transposes the note by a given interval. The interval is given by the relative LOF index.
        So unison is 0, perfect fifths is 1, major 3rds is 4, etc.
        Assuming transposing up. If you want to transpose down, say a perfect fifth,
        then transpose up a perfect fourth and compound by -1."""
        new_index = self.index + interval
        # Make a draft note to detect octave changes
        draft_step_number = (0, 4, 1, 5, 2, 6, 3)[new_index % 7]
        draft_alter = (self.index + 1) // 7
        draft_pitch_number = ([0, 2, 4, 5, 7, 9, 11][draft_step_number] + draft_alter)
        new_octave = self.octave + compound
        if (draft_pitch_number % 12) < (self.pitch_number % 12):
            new_octave += 1
        return Note(
            index=new_index,
            octave=new_octave,
            duration=self.duration,
            offset=self.offset,
            velocity=self.velocity
        )

    @classmethod
    def from_midi_number(cls, midi_number: int, duration: float | Fraction = 0., offset: float | Fraction = 0., velocity: int = 64) -> Note:
        """Creates a Note from a MIDI number. A4 maps to 69. If accidentals are needed, assumes the note is sharp."""
        octave = (midi_number // 12) - 1
        pitch = [0, 7, 2, 9, 4, -1, 6, 1, 8, 3, 10, 5][midi_number % 12]
        if not isinstance(duration, Fraction):
            duration = Fraction(duration).limit_denominator(_LIMIT_DENOMINATOR)
        if not isinstance(offset, Fraction):
            offset = Fraction(offset).limit_denominator(_LIMIT_DENOMINATOR)
        return cls(
            index=pitch,
            octave=octave,
            duration=duration,
            offset=offset,
            velocity=velocity
        )


def is_power_of_2(x: int):
    """Checks if a number is a power of 2"""
    return x > 0 and (x & (x - 1)) == 0


def highest_pow2(n: int) -> int:
    p = int(math.log(n, 2))
    return int(pow(2, p))


def dur_prefix(deg: int):
    if 0 <= deg < 3:
        return 'qhw'[deg]
    if deg >= 3:
        return 'w' + "'" * (deg - 2)
    if deg == -1:
        return 'r'
    if deg == -2:
        return 's'
    return 's' + "'" * (-deg - 2)


def duration_to_str(dur: Fraction | int | float):
    """Returns the duration of the note in a compact notation

    'w' for whole note, 'h' for half note, 'q' for quarter note,
    'r' for eighth note, 's' for sixteenth note

    Each apostrophe represents twice or half the length, so 'w'' is a breve (double whole note),
    or 's'' is a thirty-second note.

    '.' for dotted notes

    If the duration is a tuplet, the tuplet number is included as a suffix.

    If the note is a tied note, it is joined with '+'
    """
    assert dur > 0, f"Duration must be greater than 0, but found {dur}"
    if not isinstance(dur, Fraction):
        dur = Fraction(dur).limit_denominator(_LIMIT_DENOMINATOR)
    x, y = dur.numerator, dur.denominator
    if is_power_of_2(y):
        binary_str = bin(x)[2:]
        c1 = binary_str.count('1')
        c0 = binary_str.count('0')
        if binary_str == '1' * c1 + '0' * c0:
            return dur_prefix(c1 + c0 - 1 - int(math.log2(y))) + '.' * (c1 - 1)
        trail = int(re.sub('1+0*', '', binary_str, count=1), 2)
        greedy = x - trail
        return duration_to_str(Fraction(greedy, y)) + '+' + duration_to_str(Fraction(trail, y))
    return str(y) + duration_to_str(Fraction(x, highest_pow2(y)))


def _step_alter_to_lof_index(step: StepName, alter: int) -> int:
    return {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}[step] + 7 * alter
