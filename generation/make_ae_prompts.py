#!/usr/bin/env python3
"""Prompts for the base-model heatmap (SoundAQnet Fig. 6 style): 15 audio-event classes x 4 plain prompts.

Plain prompts, no pleasantness word, no tag, no punctuation (the file name is built from the prompt text).
Writes next to this script
  prompts_base_ae.txt        id|group|item for generate_prompt_range.py --prompts-file
  ae_label_primary.csv       prompt_id + 15 columns lab_<class>: 1 only for the class the prompt was written for
  ae_label_broad.csv         the same plus the classes that obviously come with it (my choice, see ALSO; edit freely):
                             e.g. a bird prompt also counts as Animal, Natural sounds, Outside/rural/natural
The labels are the INTENDED content. Whether the model produced it is not checked here.

    python generation/make_ae_prompts.py
"""
import csv
import re
from pathlib import Path

EVENTS = ['Silence', 'Human sounds', 'Wind', 'Water', 'Natural sounds', 'Traffic', 'Sounds of things', 'Vehicle',
          'Bird', 'Outside, rural or natural', 'Environment and background', 'Speech', 'Music', 'Noise', 'Animal']
OUT, NAT = 'Outside, rural or natural', 'Natural sounds'

# class -> list of (prompt, extra classes)
P = {
 'Silence': [("an empty room with almost no sound", []), ("silence in a quiet room at night", []),
             ("very faint room tone", []), ("a quiet empty hall", [])],
 'Human sounds': [("footsteps of people walking on a pavement", []), ("people laughing and clapping", []),
                  ("a crowd moving through a public square", []), ("children playing in a playground", [])],
 'Wind': [("wind blowing through trees", [NAT, OUT]), ("strong wind howling around a house", [NAT]),
          ("a gentle breeze in tall grass", [NAT, OUT]), ("wind blowing over an open field", [NAT, OUT])],
 'Water': [("waves breaking on a beach", [NAT, OUT]), ("a stream flowing over stones", [NAT, OUT]),
           ("rain falling on leaves", [NAT, OUT]), ("a fountain splashing in a park", [OUT])],
 'Natural sounds': [("thunder rolling over hills", [OUT]), ("leaves rustling in a forest", [OUT]),
                    ("a quiet forest ambience", [OUT]), ("a rocky coast with the sea", [OUT, 'Water'])],
 'Traffic': [("heavy traffic on a busy road", ['Vehicle']), ("cars passing on a street", ['Vehicle']),
             ("a traffic jam with car horns", ['Vehicle']), ("a busy intersection with cars and buses", ['Vehicle'])],
 'Sounds of things': [("a clock ticking in a room", []), ("a door closing and keys jingling", []),
                      ("a washing machine running", []), ("a kettle boiling in a kitchen", [])],
 'Vehicle': [("a motorcycle passing by", []), ("a train passing at a station", []),
             ("a bus engine idling", []), ("an aeroplane flying overhead", [])],
 'Bird': [("birds chirping in a forest", ['Animal', NAT, OUT]), ("a blackbird singing in a garden", ['Animal', NAT, OUT]),
          ("a flock of seagulls at the coast", ['Animal', NAT, OUT]), ("a dawn chorus of birds", ['Animal', NAT, OUT])],
 'Outside, rural or natural': [("a quiet village street in the countryside", []), ("an open meadow with distant sounds", [NAT]),
                               ("a farm yard in the morning", ['Animal']), ("a path through a park", [])],
 'Environment and background': [("background sounds of a city", []), ("the ambience of a large open plan office", []),
                                ("distant background noise of a town", []), ("the general ambience of a shopping street", [])],
 'Speech': [("a man talking", ['Human sounds']), ("two people having a conversation", ['Human sounds']),
            ("a woman speaking in a room", ['Human sounds']), ("a group of people talking in a cafe", ['Human sounds'])],
 'Music': [("a street musician playing guitar", []), ("a piano playing in a room", []),
           ("an orchestra playing", []), ("electronic music playing in a club", [])],
 'Noise': [("loud industrial machinery noise", []), ("a construction site with drilling and hammering", []),
           ("white noise static", []), ("a loud alarm ringing", [])],
 'Animal': [("a dog barking", []), ("cows mooing in a field", [OUT]),
            ("crickets chirping at night", [NAT, OUT]), ("frogs croaking by a pond", [NAT, OUT])],
}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    here = Path(__file__).resolve().parent
    assert list(P) and set(P) == set(EVENTS), set(EVENTS) ^ set(P)
    lines, prim, broad = [], [], []
    for cls in EVENTS:
        for k, (text, extra) in enumerate(P[cls], 1):
            pid = f"ae_{slug(cls)}_{k}"
            lines.append(f"{pid}|{slug(cls)}|{text}")
            prim.append({"prompt_id": pid, **{f"lab_{e}": int(e == cls) for e in EVENTS}})
            broad.append({"prompt_id": pid, **{f"lab_{e}": int(e == cls or e in extra) for e in EVENTS}})
    (here / "prompts_base_ae.txt").write_text(
        "# 15 audio-event classes x 4 plain prompts (generation/make_ae_prompts.py). id|group|item\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    for name, rows in (("ae_label_primary.csv", prim), ("ae_label_broad.csv", broad)):
        with open(here / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(lines)} prompts written to {here}")


if __name__ == "__main__":
    main()
