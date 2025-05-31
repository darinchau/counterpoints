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
    'Major'  # TODO Add minor key support later since I assume we have to do something weird with harmonic/melodic minor
]
