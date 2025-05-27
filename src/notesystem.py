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
#
# Basic constraints:
#   - Only one note can be played at a time in a given voice
#   - - sum_{p, r} x_{p, r, off} <= 1 for all off
#   - Enough notes must be active to make up the bar (including rests)
#   - - sum_{p, r, off} r x_{p, r, off} = 4
#   - This also yields all the non-overlapping constraints
#   - - x_{p, r, off} + x_{p', r', off'} <= 1 for all p != p', (off, off + r) overlaps with (off', off' + r')
#   - A note can be tied to its next note when the tie variable is active
#   - or if x_{tie, r, off} = 1, then x{p, r, off} = x_{p, r', off + r} for all p, r, off, r'
#   - - |x{p, r, off} - x_{p, r', off + r}| <= 1 - x_{tie, r, off}

from __future__ import annotations
import enum
import typing
import math
from abc import ABC, abstractmethod
from fractions import Fraction
from .note import Note, duration_str_to_fraction, midi_number_from_index_octave
from dataclasses import dataclass, field, asdict
from itertools import product
from collections import defaultdict
from .consts import (
    PIANO_A0, PIANO_C8, _LIMIT_DENOMINATOR, VariableIndex, Constraint,
    VoiceRange, KeyName, ModeName
)


class NoteSystem(ABC):
    """Defines a system that organizes notes and constraints.
    All the variables are assumed to be 0/1 variables and are indexed by integers.

    Implement get_variables() to return a set of all variables from your own
    and get_constraints() to return a set of all constraints."""

    @abstractmethod
    def get_variables(self) -> set[VariableIndex]:
        raise NotImplementedError

    def get_all_variables(self) -> set[VariableIndex]:
        """Return all variables including those in the subclass"""
        all_variables = self.get_variables()
        for var, value in vars(self).items():
            if isinstance(value, NoteSystem):
                subclass_vars = value.get_all_variables()
                # This says all variables must only be contained in one constraint system
                if not all_variables.isdisjoint(subclass_vars):
                    raise ValueError(
                        f"Variable {var} in {self.__class__.__name__} conflicts with existing variables."
                    )
                all_variables.update(subclass_vars)
        return all_variables

    @abstractmethod
    def get_constraints(self) -> tuple[list[Constraint], list[Constraint]]:
        """Return a tuple of two lists:
        - The first list contains constraints of the form Ax <= b
        - The second list contains constraints of the form Cx = d
        """
        raise NotImplementedError

    def get_all_constraints(self) -> tuple[list[Constraint], list[Constraint]]:
        all_vars = self.get_all_variables()
        all_constraints = self.get_constraints()
        # This check says all constraints in self must only contain variables from self or subclasses
        # So this puts a limitation on how we organize constraints
        assert all(x in all_vars for constraint in all_constraints[0] for x in constraint[1])
        assert all(x in all_vars for constraint in all_constraints[1] for x in constraint[1])
        assert all(len(a) == len(x) for a, x, b in all_constraints[0])
        assert all(len(c) == len(x) for c, x, d in all_constraints[1])
        for var, value in vars(self).items():
            if isinstance(value, NoteSystem):
                subclass_cons = value.get_all_constraints()
                all_constraints[0].extend(subclass_cons[0])
                all_constraints[1].extend(subclass_cons[1])
        return all_constraints


class Bar(NoteSystem):
    """Represents a bar that encodes variables with all the possible notes and constraints"""

    def __init__(self, bar_number: int):
        self.bar_number = bar_number
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

    def get_constraints(self) -> tuple[list[Constraint], list[Constraint]]:
        """Return constraints of the form Ax <= b and Cx = d."""
        def is_tie(v: VariableIndex):
            return len(v) == 4 and v[3] == 1

        # Only one note can be played at a time in a given voice
        ineq_constraints: list[Constraint] = []
        eq_constraints: list[Constraint] = []
        variables = frozenset(self.get_variables())  # Just to avoid me accidentally modifying the set

        # at most one note per rhythm per offset
        # sum_{p, r} x_{p, r, off} <= 1 for all off
        one_note_offset_vars: dict[tuple[int, int], list[VariableIndex]] = defaultdict(list)
        for var in variables:
            if is_tie(var):
                # Exclude the tie variable
                continue
            one_note_offset_vars[(var[1], var[2])].append(var)
        for (n_notes_in_bar, offset), vars_ in one_note_offset_vars.items():
            if not vars_:
                continue
            # Create a constraint for this offset
            constraint = [1] * len(vars_)
            ineq_constraints.append((constraint, vars_, 1))
            # print(f"Adding constraint for {n_notes_in_bar}th notes at offset {offset}")

        # Enough notes must be active to make up the bar (including rests)
        # sum_{p, r, off} r x_{p, r, off} = 4
        active_note_length_vars = []
        active_note_length_coeff = []
        lcm_denominator = math.lcm(*[x[1] for x in variables])
        for var in variables:
            if is_tie(var):
                continue
            active_note_length_vars.append(var)
            active_note_length_coeff.append(lcm_denominator // var[1])
        if active_note_length_vars:
            # This is the only equality constraint
            eq_constraints.append((active_note_length_coeff, active_note_length_vars, lcm_denominator))
            # print(f"Adding equality constraint for active note lengths: {lcm_denominator}")

        # Non-overlapping constraints
        # x_{p, r, off} + x_{p', r', off'} <= 1 for all p != p, (off, off + r) overlaps with (off', off' + r')
        for var1 in variables:
            if is_tie(var1):
                continue
            for var2 in variables:
                if var1 >= var2:
                    continue
                if is_tie(var2):
                    continue
                start1, end1 = var1[2] / var1[1], (var1[2] + 1) / var1[1]
                start2, end2 = var2[2] / var2[1], (var2[2] + 1) / var2[1]
                if start1 > start2:
                    # Ensure that start1 <= start2
                    start1, end1, start2, end2 = start2, end2, start1, end1
                    var1, var2 = var2, var1
                if start2 < end1:
                    # Add a constraint: at most one of them can be active
                    ineq_constraints.append(([1, 1], [var1, var2], 1))
                    # print(f"Adding non-overlapping constraint for {var1} and {var2}")
                elif start2 == end1 and (
                    len(var1) == 5 and len(var2) == 5 and var1[3] == var2[3] and var1[4] == var2[4]
                ):
                    # A note can be tied to its next note when the tie variable is active
                    # or if x_{tie, r, off} = 1, then x{p, r, off} = x_{p, r', off + r} for all p, r, off, r'
                    tie1 = (self.bar_number, var1[1], var1[2], 1)  # Tie variable for var1
                    if tie1 not in variables:
                        continue
                    ineq_constraints.append(([1, -1, 1], [var1, var2, tie1], 1))
                    ineq_constraints.append(([1, -1, 1], [var2, var1, tie1], 1))
                    # print(f"Adding tie constraint between notes {var1} and {var2}")

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
                    variables.add((self.bar_number, n_notes_in_bar, i, index, octave))
                variables.add((self.bar_number, n_notes_in_bar, i, 0))  # For the rest
                variables.add((self.bar_number, n_notes_in_bar, i, 1))  # For the tie
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
