from __future__ import annotations
import math
import re
from dataclasses import dataclass, asdict
from functools import reduce, cached_property, lru_cache
from typing import Literal
from fractions import Fraction
from .consts import (
    PIANO_A0, PIANO_C8, _LIMIT_DENOMINATOR, StepName, _PITCH_NAME_REGEX,
)


@dataclass(frozen=True)
class Note:
    """A piano note is a representation of a note on the piano, with a note name and an octave
    The convention being middle C is C4. The lowest note is A0 and the highest note is C8.

    If the note is in real time, then the duration and offset is timed with respect to quarter length,
    otherwise it is timed with respect to real-time seconds.

    A note of velocity 0 is a rest of that duration

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
    def is_rest(self):
        """Returns True if the note is a rest, i.e. has velocity 0."""
        return self.velocity == 0

    @cached_property
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

    @cached_property
    def note_name(self):
        """The note name of the note. e.g. A4, C#5, etc."""
        return f"{self.pitch_name}{self.octave}[{self.duration_name}]"

    @cached_property
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

    @cached_property
    def step(self) -> StepName:
        """Returns the diatonic step of the note"""
        idx = self.index % 7
        return ("C", "G", "D", "A", "E", "B", "F")[idx]

    @cached_property
    def step_number(self) -> int:
        """Returns the diatonic step number of the note, where C is 0, D is 1, etc."""
        idx = self.index % 7
        return (0, 4, 1, 5, 2, 6, 3)[idx]

    @cached_property
    def alter(self):
        """Returns the alteration of the note aka number of sharps. Flats are represented as negative numbers."""
        return (self.index + 1) // 7

    @cached_property
    def pitch_number(self):
        """Returns the chromatic pitch number of the note. C is 0, D is 2, etc. There are edge cases like B# returning 12 or Cb returning -1"""
        return ([0, 2, 4, 5, 7, 9, 11][self.step_number] + self.alter)

    @cached_property
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

    @classmethod
    def from_str(cls, name: str, offset: float | int | Fraction = 0, velocity: int = 64):
        """Assume the name in self.pitch_name format

        "{self.pitch_name}{self.octave}[{self.duration_name}]"""
        if not isinstance(offset, Fraction):
            offset = Fraction(offset).limit_denominator(_LIMIT_DENOMINATOR)

        if "[" in name:
            name, rest = name.split("[")
            rest = rest.rstrip("]")
            duration = duration_str_to_fraction(rest)
        else:
            duration = Fraction(1)

        match = _PITCH_NAME_REGEX.match(name)
        if not match:
            # Add the implied octave = 4
            match = _PITCH_NAME_REGEX.match(name + "4")

        assert match and len(match.groups()) == 3, f"The name {name} is not a valid note name"
        pitch_name, alter, octave = match.groups()
        if alter is None:
            alter = ""
        alter = alter.replace("x", "##").replace("-", "b").replace("+", "#")
        sharps = reduce(lambda x, y: x + 1 if y == "#" else x - 1, alter, 0)
        assert pitch_name in ("C", "D", "E", "F", "G", "A", "B"), f"Step must be one of CDEFGAB, but found {pitch_name}"  # to pass the typechecker

        return cls(
            index=_step_alter_to_lof_index(pitch_name, sharps),
            octave=int(octave),
            duration=duration,
            offset=offset,
            velocity=velocity
        )


@lru_cache(maxsize=256)
def midi_number_from_index_octave(index: int, octave: int) -> int:
    """RA convenient function to get the MIDI number from the index and octave
    Should be identical to Note(index, octave, ...).midi_number"""
    step_number = (0, 4, 1, 5, 2, 6, 3)[index % 7]
    alter = (index + 1) // 7
    pitch_number = ([0, 2, 4, 5, 7, 9, 11][step_number] + alter)
    return pitch_number + 12 * octave + 12


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
        dur = Fraction(dur).limit_denominator(47)
    x, y = dur.numerator, dur.denominator
    if not is_power_of_2(y):
        return str(y) + duration_to_str(Fraction(x, highest_pow2(y)))
    binary_str = bin(x)[2:]
    c1 = binary_str.count('1')
    c0 = binary_str.count('0')
    if binary_str == '1' * c1 + '0' * c0:
        return dur_prefix(c1 + c0 - 1 - int(math.log2(y))) + '.' * (c1 - 1)
    trail = int(re.sub('1+0*', '', binary_str, count=1), 2)
    greedy = x - trail
    return duration_to_str(Fraction(greedy, y)) + '+' + duration_to_str(Fraction(trail, y))


_BASE_NOTE = {
    'w': Fraction(4),
    'h': Fraction(2),
    'q': Fraction(1),
    'r': Fraction(1, 2),
    's': Fraction(1, 4)
}


@lru_cache(maxsize=1024)
def duration_str_to_fraction(duration_str: str) -> Fraction:
    """Converts a duration string to a Fraction. Reverse of duration_to_str."""
    duration_str = duration_str.strip()

    match = re.match(r'(\d+)*[qhwrs].*', duration_str)
    assert match, f"Invalid duration string: {duration_str}"
    tuplet_num, = match.groups()
    if tuplet_num is not None:
        return Fraction(highest_pow2(int(tuplet_num)), int(tuplet_num)) * duration_str_to_fraction(duration_str.replace(tuplet_num, '', 1))

    if '+' in duration_str:
        parts = duration_str.split('+')
        return reduce(lambda x, y: x + duration_str_to_fraction(y), parts, Fraction(0))

    match = re.match(r'([qhwrs])(\'*)(\.*)', duration_str)
    assert match, f"Invalid duration string: {duration_str}"

    base_note, apostrophes, dot_num = match.groups()
    base_duration = _BASE_NOTE[base_note]
    mults = []
    dot_num = len(dot_num)

    if apostrophes and base_note == 's':
        mults.extend([Fraction(1, 2) for _ in range(len(apostrophes))])
    elif apostrophes and base_note == 'w':
        mults.extend([Fraction(2) for _ in range(len(apostrophes))])
    elif apostrophes:
        raise ValueError(f"Invalid apostrophes in duration string: {duration_str}")
    if dot_num > 0:
        base = 1 << dot_num
        mults.append(Fraction(base * 2 - 1, base))
    return reduce(lambda x, y: x * y, [base_duration] + mults, Fraction(1))


def _step_alter_to_lof_index(step: StepName, alter: int) -> int:
    return {"C": 0, "D": 2, "E": 4, "F": -1, "G": 1, "A": 3, "B": 5}[step] + 7 * alter


def _test_dur_str():
    tests = [
        (1, 'q'),
        (2, 'h'),
        (4, 'w'),
        (8, "w'"),
        (12, "w'."),
        (16, "w''"),
        (3, 'h.'),
        (1/2, 'r'),
        (1/4, 's'),
        (1/8, "s'"),
        (1/16, "s''"),
        (1/32, "s'''"),
        (3/2, 'q.'),
        (5, 'w+q'),
        (6, 'w.'),
        (7, 'w..'),
        (9, "w'+q"),
        (1/3, '3r'),
        (1/5, '5s'),
        (1/6, '6s'),
        (1/7, '7s'),
        (2/3, '3q'),
        (4/3, '3h'),
        (3/7, '7r.'),
        (5/3, '3h+r')
    ]

    for dur, expected in tests:
        result = duration_to_str(dur)
        assert result == expected, f"Expected {expected} but got {result} for duration {dur}"
        d = duration_str_to_fraction(expected)
        assert math.isclose(d, dur), f"Expected {dur} but got {d} for duration string {expected}"

    for i in range(1, 47):
        for j in range(1, 47):
            dur = Fraction(i, j)
            result = duration_to_str(dur)
            d = duration_str_to_fraction(result)
            assert math.isclose(d, dur), f"Expected {dur} but got {d} for duration string {result}"
