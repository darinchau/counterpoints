# A note system is like a canvas in which you can add notes and model constraints
from __future__ import annotations
import enum
import typing
from fractions import Fraction
from .note import Note
from dataclasses import dataclass, field, asdict

if typing.TYPE_CHECKING:
    from .constraints import ConstraintSystem


class Rhythm:
    """A rhythm defines a pattern of durations for notes in a voice."""

    def __init__(self, durations: list[list[Fraction]], time_sig: str = "4/4"):
        self._durations = durations
        self._time_sig = time_sig
        self._check_time_sig()

    def _check_time_sig(self):
        num, den = map(int, self._time_sig.split('/'))
        for i, d in enumerate(self._durations):
            if sum(d) != Fraction(num, den):
                raise ValueError(f"Bar {i} ({d}) do not match time signature {self._time_sig}")
        # TODO further check subgroup rules
        pass

    @property
    def time_sig(self):
        """Time signature of the rhythm."""
        return self._time_sig


@dataclass(frozen=True)
class PooledNote(Note):
    """A pooled note is the variable associated with the note. We will use the velocity field to
    denote its activity. So if velocity is 0, the note is inactive, and if it is non-zero, the note is active (to a certain degree)"""
    _id: int = field(init=False, default=-1)
    _parent: NotePool = field(init=False, default=None)  # type: ignore


class NotePool:
    def __init__(self):
        self._notes: list[PooledNote] = []
        self._id_counter = 0

    def get_note(self, note: Note, **kwargs):
        """Get a note from the pool, or create a new one if it doesn't exist."""
        for n in self._notes:
            if n == note:
                return n
        nd = asdict(note)
        nd.update(kwargs)
        new_note = PooledNote(**nd)
        object.__setattr__(new_note, '_id', self._id_counter)
        object.__setattr__(new_note, '_parent', self)
        self._notes.append(new_note)
        self._id_counter += 1
        return new_note


class Voice:
    """A voice is a series of non-overlapping notes"""

    def __init__(self, notes: list[Note]):
        self._notes = sorted(notes, key=lambda x: x.offset)
        self._check_nonoverlap()

    @property
    def low(self):
        return min(self._notes, key=lambda x: x.midi_number)

    @property
    def high(self):
        return max(self._notes, key=lambda x: x.midi_number)

    def _check_nonoverlap(self):
        for i in range(len(self._notes) - 1):
            if self._notes[i].offset + self._notes[i].duration > self._notes[i + 1].offset:
                raise ValueError(f"Notes overlap at index {i}")


def generate_cantus_firmus(possible_note_set: list[Note], cs: ConstraintSystem, bars: int = 12, time_sig: str = "4/4"):
    notepool = NotePool()
    bar_length = 4 * Fraction(*map(int, time_sig.split('/')))
    prev_bar_notes: list[PooledNote] | None = None
    for i in range(bars):
        bar_notes: list[PooledNote] = []
        for n in possible_note_set:
            note = notepool.get_note(n, offset=i * bar_length, duration=bar_length)
            bar_notes.append(note)
        cs.only_one_active_note(...)
