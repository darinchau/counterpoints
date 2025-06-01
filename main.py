from src.notesystem import get_scale, BarGrid
from src.solve import NoteSystemSolver
from src.consts import SOPRANO, ALTO, TENOR, BASS

b = BarGrid("Fugue in Eb Major", n_bars=2)
for voice, bars in b.grid.items():
    voice_range = {
        "Soprano": SOPRANO,
        "Alto": ALTO,
        "Tenor": TENOR,
        "Bass": BASS,
    }[voice]
    for bar in bars:
        bar.add_scale_constraint("Eb", "Major")
        bar.add_voice_constraint(voice_range)

vs, ineq, eq = b.get_system()
s = NoteSystemSolver(b)
