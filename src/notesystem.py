# A note system is like a canvas in which you can add notes and model constraints
# I don't know what's the best structure to organize these all variables
# A back of the envelope calculations says around 30 possible pitches at every possible step
# for every voice. And counting only up to 16th notes we have 31 possible rhythms
# so we have 30 * 31 = 930 possible pitches per note per voice.
#
# We model a bar using a 0/1 variable for each possible pitch and rhythm.
# one more for rests and one more for ties.
# Let's call them x_{p, r, off} for pitch p, rhythm r and offset off.
# Assuming everything in 4/4 for now, we can say the activation of A4 at the
# 3rd quarter note of the bar is x_{69, 1, 2} for pitch 69, rhythm 1 and offset 2.
#
# In reality, we will index p by LOF index and octave separately cuz we thought this might be easier
# to make accurate constraints and stuff in the future

from __future__ import annotations
import enum
import typing
import math
import copy
from abc import ABC, abstractmethod
from fractions import Fraction
from .note import Note, duration_str_to_fraction, midi_number_from_index_octave
from dataclasses import dataclass, field, asdict
from itertools import product
from collections import defaultdict
from .consts import (
    PIANO_A0, PIANO_C8, _LIMIT_DENOMINATOR,
    VoiceRange, KeyName, ModeName
)
from .indices import VariableIndex, Constraint, System


class NoteSystem(ABC):
    """Defines a system that organizes notes and constraints.
    All the variables are assumed to be 0/1 variables and are indexed by integers.

    Subclasses should implement get_constraints() to return a set of all constraints."""

    @abstractmethod
    def get_constraints(self) -> tuple[list[Constraint], list[Constraint]]:
        """Return a tuple of two lists:
        - The first list contains constraints of the form Ax <= b
        - The second list contains constraints of the form Cx = d
        """
        raise NotImplementedError

    def get_system(self) -> System:
        ineq, eq = self.get_constraints()
        assert all(len(a) == len(x) for a, x, b in ineq)
        assert all(len(c) == len(x) for c, x, d in eq)
        # Add back the aux variables that might have came up in the constraints
        all_vars = set([x for _, var, _ in ineq for x in var])
        all_vars.update([x for _, var, _ in eq for x in var])
        return sorted(all_vars), ineq, eq


class Bar(NoteSystem):
    """Represents a bar that encodes variables with all the possible notes and constraints"""

    def __init__(self, bar_name: str = ""):
        self._bar_name = bar_name
        self._voice_constraints: list[VoiceRange] = [
            VoiceRange(PIANO_A0, PIANO_C8)  # trim it to standard piano
        ]
        self._scale_constraints: list[tuple[Fraction, Fraction, set[int]]] = []
        self._permitted_notes = set("qhwrs")
        self._variables: set[VariableIndex] | None = None

    @property
    def bar_name(self) -> str:
        """Returns the name of the bar."""
        return self._bar_name

    def add_voice_constraint(self, voice_range: VoiceRange):
        """Add a voice constraint to the bar."""
        if not isinstance(voice_range, range):
            raise TypeError("Voice range must be a range object.")
        self._voice_constraints.append(voice_range)
        self._variables = None

    def add_scale_constraint(self, key: KeyName, mode: ModeName, start=0, end=4):
        """Add a scale constraint to the bar at the given key and mode from bar offset start to end."""
        scale = get_scale(key, mode)
        if not isinstance(start, Fraction):
            start = Fraction(start).limit_denominator(_LIMIT_DENOMINATOR)
        if not isinstance(end, Fraction):
            end = Fraction(end).limit_denominator(_LIMIT_DENOMINATOR)
        if start >= end or not (0 <= start < 4) or not (0 < end <= 4):
            raise ValueError(f"Start and end must be between 0 and 4, with start < end; got {start} and {end}.")
        self._scale_constraints.append((start, end, set(scale)))
        self._variables = None

    def change_permitted_notes(self, permitted_notes: typing.Iterable[str]):
        """Change the permitted notes in the bar."""
        if not isinstance(permitted_notes, (list, set, tuple)):
            raise TypeError("Permitted notes must be a list, set or tuple.")
        for n in permitted_notes:
            try:
                duration_str_to_fraction(n)
            except ValueError:
                raise ValueError(f"Invalid note duration: {n}.")
        self._permitted_notes = set(permitted_notes)
        self._variables = None

    def group_by_start_end(
        self,
        variables: typing.Iterable[VariableIndex] | None = None
    ) -> dict[tuple[Fraction, Fraction], list[VariableIndex]]:
        # Group variables by start and end. If this is useful then reimplement it to NoteSystem class
        grouped = defaultdict(list)
        if variables is None:
            variables = self.get_variables()
        assert variables is not None
        for var in variables:
            if var.is_tie:
                continue
            grouped[(var.start, var.end)].append(var)
        return grouped

    def get_variables(self) -> set[VariableIndex]:
        # Index: (bar, index, octave, rhythm (note duration), offset)
        # This is like the most inefficient way of doing it
        if self._variables is not None:
            return self._variables
        bar_length = Fraction(4)
        min_scale_deg = min(min(x[2]) for x in self._scale_constraints)
        max_scale_deg = max(max(x[2]) for x in self._scale_constraints)
        variables: set[VariableIndex] = set()
        for dur in self._permitted_notes:
            note_length = duration_str_to_fraction(dur)
            n_notes_in_bar = bar_length / note_length
            if n_notes_in_bar.denominator != 1:
                raise ValueError(
                    f"Bar length {bar_length} cannot be evenly divided by note length {note_length}."
                )
            n_notes_in_bar = n_notes_in_bar.numerator
            for i in range(n_notes_in_bar):
                # Midi cannot get past 128 so let's use this as a limit
                permitted_pitches = set(range(0, 128))
                permitted_indices = set(range(min_scale_deg, max_scale_deg + 1))
                start_ql = 4 * Fraction(i, n_notes_in_bar)
                end_ql = 4 * Fraction(i + 1, n_notes_in_bar)
                # Sanity check
                assert end_ql == start_ql + note_length, f"End of note {end_ql} does not match start {start_ql} + length {note_length}."
                for start, end, scale in self._scale_constraints:
                    if start > end_ql or end > start_ql:
                        # The interval does not intersect with the scale constraint time
                        continue
                    permitted_indices &= scale
                if not permitted_indices:
                    continue
                for voice in self._voice_constraints:
                    permitted_pitches &= set(voice)
                for index, octave in product(permitted_indices, range(-1, 9)):
                    note_midi_number = midi_number_from_index_octave(index, octave)
                    if note_midi_number not in permitted_pitches:
                        continue
                    # N + 2 variables per note: note activation, tie, rest
                    # Using i and n_notes_in_bar as a proxy for the offset and duration
                    variables.add(VariableIndex(
                        name=self._bar_name,
                        duration=n_notes_in_bar,
                        offset=i,
                        index=index,
                        octave=octave
                    ))
                variables.add(VariableIndex.make_rest(self._bar_name, n_notes_in_bar, i))
                if i < n_notes_in_bar - 1:
                    # Only add tie if it's not the last note in the bar
                    # TODO add cross-bar tie constraints somewhere else
                    variables.add(VariableIndex.make_tie(self._bar_name, n_notes_in_bar, i))
        self._variables = variables
        return variables

    def get_constraints(self) -> tuple[list[Constraint], list[Constraint]]:
        """Return constraints of the form Ax <= b and Cx = d."""
        # Only one note can be played at a time in a given voice
        ineq_constraints: list[Constraint] = []
        eq_constraints: list[Constraint] = []
        vset = self.get_variables()
        variables = list(vset)

        # Enough notes must be active to make up the bar (including rests)
        # sum_{p, r, off} r x_{p, r, off} = bar_length
        active_note_length_vars = []
        active_note_length_coeff = []
        lcm_denominator = math.lcm(*[x.duration for x in variables])
        for var in variables:
            if var.is_tie:
                continue
            active_note_length_vars.append(var)
            active_note_length_coeff.append(lcm_denominator // var.duration)
        if active_note_length_vars:
            eq_constraints.append((active_note_length_coeff, active_note_length_vars, lcm_denominator))

        # Non-overlap constraints
        # For each two groups of variables that overlap in time, we add a constraint
        grouped_vars = self.group_by_start_end(variables)
        for (s1, e1), vars1 in grouped_vars.items():
            for (s2, e2), vars2 in grouped_vars.items():
                if s1 > s2 or (s1 == s2 and e1 == e2):
                    continue
                if s2 < e1:
                    total_vars = len(vars1) + len(vars2)
                    ineq_constraints.append(([1] * total_vars, vars1 + vars2, 1))

        # Tied note constraints - if note 1 is active and tied then the next note must be active
        # sum_{r'} x_{p, r', off + r} >= z for all p, r, off
        # Introduce an aux variable z: z >= x_{p, r, off} X x_{tie, r, off} >= x_{p, r, off} + x_{tie, r, off} - 1
        # If both the note and tie is active, then the set of all subsequent notes must have at least one
        # being active; otherwise the constraints are trivial. Rests cannot be tied
        grouped_start_var: dict[Fraction, list[VariableIndex]] = defaultdict(list)
        for (s, e), vs in grouped_vars.items():
            grouped_start_var[s].extend(vs)

        for var in variables:
            if var.is_tie or var.is_rest:
                continue
            tie = var.get_tie()
            if tie not in vset:
                continue
            constrained_vars = []

            constrained_vars = [
                v for v in grouped_start_var[var.end] if
                v.index == var.index and
                v.octave == var.octave
            ]
            if not constrained_vars:
                continue
            z = VariableIndex(f"{self._bar_name}aux1", var.duration, var.offset, var.index, var.octave, aux=True)
            ineq_constraints.append(([1, 1, -1], [var, tie, z], 1))
            ineq_constraints.append(([1] + [-1] * len(constrained_vars), [z] + constrained_vars, 0))

        # The tie variable also cannot be active if no notes are active
        # And specifically the tie and the rest cannot be active at the same time
        for _, vs in grouped_vars.items():
            if not vs:
                continue
            tie = vs[0].get_tie()
            rest = vs[0].get_rest()
            if tie not in vset or rest not in vset:
                continue
            vs = [v for v in vs if not v.is_rest]
            ineq_constraints.append(([1] + [-1] * len(vs), [tie] + vs, 0))
            ineq_constraints.append(([1, 1], [tie, rest], 1))
        return (ineq_constraints, eq_constraints)


class BarGrid(NoteSystem):
    """Represents a grid of bars with vertical alignment (voices) and horizontal alignment (bars)."""

    def __init__(self, name: str, n_bars: int = 64, voice_names: typing.Iterable[str] = ("Soprano", "Alto", "Tenor", "Bass")):
        self._name = name
        self._n_bars = n_bars
        self._voice_names = list(voice_names)
        if len(self._voice_names) < 1:
            raise ValueError("At least one voice name must be provided.")
        self.grid: dict[str, list[Bar]] = {k: [
            Bar(f"{self._name}_{k}_{i}") for i in range(self._n_bars)
        ] for k in self._voice_names}

    @property
    def bars(self):
        """Returns a list of all bars in the grid."""
        return [bar for voice in self.grid.values() for bar in voice]

    def get_system(self) -> System:
        # Check if the structure is consistent
        assert set(self.grid.keys()) == set(self._voice_names), \
            f"Voice names {set(self.grid.keys())} do not match provided voice names {set(self._voice_names)}."

        for voice in self._voice_names:
            if len(self.grid[voice]) != self._n_bars:
                raise ValueError(f"Voice {voice} has {len(self.grid[voice])} bars, expected {self._n_bars}.")
        return super().get_system()

    @property
    def name(self) -> str:
        """Returns the name of the grid."""
        return self._name

    @property
    def n_bars(self) -> int:
        """Returns the number of bars in the grid."""
        return self._n_bars

    @property
    def voice_names(self) -> list[str]:
        """Returns the list of voice names in the grid."""
        return copy.deepcopy(self._voice_names)

    def get_constraints(self) -> tuple[list[Constraint], list[Constraint]]:
        ineq: list[Constraint] = []
        eq: list[Constraint] = []
        for voice in self._voice_names:
            # Get all subconstraints from each bar
            for var in self.grid[voice]:
                bar_ineq, bar_eq = var.get_constraints()
                ineq.extend(bar_ineq)
                eq.extend(bar_eq)
            variables = [
                self.grid[voice][i].get_variables()
                for i in range(self._n_bars)
            ]
            # Tie constraints in between bars
            for i in range(self._n_bars - 1):
                bar1_end = set([
                    v for v in variables[i] if
                    math.isclose(v.end, 1) and
                    not v.is_rest and
                    not v.is_tie
                ])
                bar2_start = set([
                    v for v in variables[i + 1] if
                    math.isclose(v.start, 0) and
                    not v.is_rest and
                    not v.is_tie
                ])
                for var in bar1_end:
                    tie = var.get_tie()
                    constrained_vars = [
                        v for v in bar2_start if
                        v.index == var.index and
                        v.octave == var.octave
                    ]
                    if not constrained_vars:
                        continue
                    z = VariableIndex(f"{var.name}aux1", var.duration, var.offset, var.index, var.octave, aux=True)
                    ineq.append(([1, 1, -1], [var, tie, z], 1))
                    ineq.append(([1] + [-1] * len(constrained_vars), [z] + constrained_vars, 0))
        return ineq, eq


def get_scale(key: KeyName, mode: ModeName) -> list[int]:
    """Returns a scale in the form of a list of pitch classes."""
    key_idx = Note.from_str(f"{key}4").index
    if not (-6 <= key_idx <= 6):
        # This check is arbitrary - can remove in the future
        raise ValueError(f"Key {key} {mode} is out of range. Must be between 6 sharps and 6 flats.")
    if mode == 'Major':
        mode_lof = [0, 2, 4, -1, 1, 3, 5]  # Number of sharps in major key notes
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return [key_idx + offset for offset in mode_lof]
