"""Caption format shared by training-data preparation and generation.

A caption is   "<scene> [ISOPleasant: <value>]"   e.g.
    "park soundscape with flowing water in the foreground [ISOPleasant: 0.50]"

The scene text comes from the masker type and from the sign of the SMR
(negative SMR = masker louder than the soundscape = "foreground").
Training and generation must use exactly this format, so both import it from here.
"""

MASKER_WORDS = {
    "bird": "birdsong",
    "construction": "construction noise",
    "traffic": "traffic noise",
    "water": "flowing water",
    "wind": "wind",
}


def scene_text(masker_word: str, foreground: bool) -> str:
    place = "in the foreground" if foreground else "in the background"
    return f"park soundscape with {masker_word} {place}"


def format_pleasantness(value: float) -> str:
    s = f"{value:.2f}"
    return "0.00" if s == "-0.00" else s


def make_caption(masker_word: str, foreground: bool, pleasantness: float) -> str:
    return f"{scene_text(masker_word, foreground)} [ISOPleasant: {format_pleasantness(pleasantness)}]"
