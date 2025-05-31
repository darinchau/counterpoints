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
from abc import ABC, abstractmethod
from fractions import Fraction
from .note import Note, duration_str_to_fraction, midi_number_from_index_octave
from dataclasses import dataclass, field, asdict
from itertools import product
import typing
from collections import defaultdict
from .consts import (
    PIANO_A0, PIANO_C8, _LIMIT_DENOMINATOR,
    VoiceRange, KeyName, ModeName
)
from .indices import VariableIndex, Constraint


class NoteSystem(ABC):
    """Defines a system that organizes notes and constraints.
    All the variables are assumed to be 0/1 variables and are indexed by integers.

    Implement get_variables() to return a set of all variables from your own
    and get_constraints() to return a set of all constraints."""

    @abstractmethod
    def get_variables(self) -> set[VariableIndex]:
        raise NotImplementedError

    @abstractmethod
    def get_constraints(self) -> tuple[list[Constraint], list[Constraint]]:
        """Return a tuple of two lists:
        - The first list contains constraints of the form Ax <= b
        - The second list contains constraints of the form Cx = d
        """
        raise NotImplementedError

    def get_system(self) -> tuple[list[VariableIndex], list[Constraint], list[Constraint]]:
        all_vars = self.get_variables()
        ineq, eq = self.get_constraints()
        assert all(len(a) == len(x) for a, x, b in ineq)
        assert all(len(c) == len(x) for c, x, d in eq)
        # Add back the aux variables that might have came up in the constraints
        all_vars.update([x for _, var, _ in ineq for x in var])
        all_vars.update([x for _, var, _ in eq for x in var])
        for var, value in vars(self).items():
            if isinstance(value, NoteSystem):
                subclass_vars, subclass_ineq, subclass_eq = value.get_system()
                if not all_vars.isdisjoint(subclass_vars):
                    raise ValueError(
                        f"Variable {var} in {self.__class__.__name__} conflicts with existing variables."
                    )
                all_vars.update(subclass_vars)
                ineq.extend(subclass_ineq)
                eq.extend(subclass_eq)
        return sorted(all_vars), ineq, eq


class Bar(NoteSystem):
    """Represents a bar that encodes variables with all the possible notes and constraints"""

    def __init__(self, bar_name: str = ""):
        self.bar_name = bar_name
        self.voice_constraints: list[VoiceRange] = [
            VoiceRange(PIANO_A0, PIANO_C8)  # trim it to standard piano
        ]
        self.scale_constraints: list[tuple[Fraction, Fraction, set[int]]] = []
        self.permitted_notes = set("qhwrs")

    def add_voice_constraint(self, voice_range: VoiceRange):
        """Add a voice constraint to the bar."""
        if not isinstance(voice_range, range):
            raise TypeError("Voice range must be a range object.")
        self.voice_constraints.append(voice_range)

    def add_scale_constraint(self, key: KeyName, mode: ModeName, start=0, end=4):
        """Add a scale constraint to the bar at the given key and mode from bar offset start to end."""
        scale = get_scale(key, mode)
        if not isinstance(start, Fraction):
            start = Fraction(start).limit_denominator(_LIMIT_DENOMINATOR)
        if not isinstance(end, Fraction):
            end = Fraction(end).limit_denominator(_LIMIT_DENOMINATOR)
        if start >= end or not (0 <= start < 4) or not (0 < end <= 4):
            raise ValueError(f"Start and end must be between 0 and 4, with start < end; got {start} and {end}.")
        self.scale_constraints.append((start, end, set(scale)))

    def group_by_start_end(self, variables: typing.Iterable[VariableIndex] | None = None) -> dict[tuple[float, float], list[VariableIndex]]:
        # Group variables by start and end. If this is useful then reimplement it to NoteSystem class
        grouped = defaultdict(list)
        if variables is None:
            variables = self.get_variables()
        for var in variables:
            if var.is_tie:
                continue
            grouped[(var.start, var.end)].append(var)
        return grouped

    def get_constraints(self) -> tuple[list[Constraint], list[Constraint]]:
        """Return constraints of the form Ax <= b and Cx = d."""
        # Only one note can be played at a time in a given voice
        ineq_constraints: list[Constraint] = []
        eq_constraints: list[Constraint] = []
        variables = sorted(self.get_variables())

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
        for var in variables:
            if var.is_tie or var.is_rest:
                continue
            tie = var.get_tie()
            if tie not in variables:
                continue
            constrained_vars = [
                v for v in variables if
                v.start == var.end and
                v.index == var.index and
                v.octave == var.octave and
                v.name == var.name
            ]
            if not constrained_vars:
                continue
            z = VariableIndex(f"{self.bar_name}aux1", var.duration, var.offset, var.index, var.octave, aux=True)
            ineq_constraints.append(([1, 1, -1], [var, tie, z], 1))
            ineq_constraints.append(([1] + [-1] * len(constrained_vars), [z] + constrained_vars, 0))
        return (ineq_constraints, eq_constraints)

    def get_variables(self) -> set[VariableIndex]:
        # Index: (bar, index, octave, rhythm (note duration), offset)
        # This is like the most inefficient way of doing it
        bar_length = Fraction(4)
        min_scale_deg = min(min(x[2]) for x in self.scale_constraints)
        max_scale_deg = max(max(x[2]) for x in self.scale_constraints)
        variables: set[VariableIndex] = set()
        for dur in self.permitted_notes:
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
                for start, end, scale in self.scale_constraints:
                    if start > end_ql or end > start_ql:
                        # The interval does not intersect with the scale constraint time
                        continue
                    permitted_indices &= scale
                if not permitted_indices:
                    continue
                for voice in self.voice_constraints:
                    permitted_pitches &= set(voice)
                for index, octave in product(permitted_indices, range(-1, 9)):
                    note_midi_number = midi_number_from_index_octave(index, octave)
                    if note_midi_number not in permitted_pitches:
                        continue
                    # N + 2 variables per note: note activation, tie, rest
                    # Using i and n_notes_in_bar as a proxy for the offset and duration
                    variables.add(VariableIndex(
                        name=self.bar_name,
                        duration=n_notes_in_bar,
                        offset=i,
                        index=index,
                        octave=octave
                    ))
                variables.add(VariableIndex.make_rest(self.bar_name, n_notes_in_bar, i))
                variables.add(VariableIndex.make_tie(self.bar_name, n_notes_in_bar, i))
        return variables


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
