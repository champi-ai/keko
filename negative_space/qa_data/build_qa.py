"""Build the bundled factual-QA benchmark used by the negative-space experiment.

Run from the repo root to (re)generate ``negative_space/qa_data/qa_bundled.jsonl``::

    uv run python -m negative_space.qa_data.build_qa

Every item is self-authored (no external dataset is copied). Design rules, which
exist to defend against the confounds listed in ``negative_space/README.md``:

* **Matched negative controls.** Fictional-entity questions reuse *exactly* the
  same templates as the real questions (``template`` field), with invented names of
  similar length, so a probe cannot separate them by surface form alone.
* **Aleatoric vs epistemic.** Genuinely ambiguous questions (several defensible
  answers, or missing context) are tagged ``kind="ambiguous"``. They are excluded
  from the primary known/unknown analysis and reserved for the planned
  clarification promoter (N6).
* **Grouping for leakage control.** ``group`` identifies the entity; the evaluation
  splits by group so the same entity never appears in both train and test.

Answers list accepted aliases. Arithmetic items are generated deterministically
from a fixed seed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).with_name("qa_bundled.jsonl")

# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
CAPITALS = [
    ("France", ["Paris"]), ("Japan", ["Tokyo"]), ("Italy", ["Rome"]),
    ("Germany", ["Berlin"]), ("Spain", ["Madrid"]), ("Canada", ["Ottawa"]),
    ("Australia", ["Canberra"]), ("Brazil", ["Brasilia", "Brasília"]),
    ("Egypt", ["Cairo"]), ("Kenya", ["Nairobi"]), ("Argentina", ["Buenos Aires"]),
    ("Peru", ["Lima"]), ("Chile", ["Santiago"]), ("Norway", ["Oslo"]),
    ("Sweden", ["Stockholm"]), ("Finland", ["Helsinki"]), ("Poland", ["Warsaw"]),
    ("Portugal", ["Lisbon"]), ("Greece", ["Athens"]), ("Turkey", ["Ankara"]),
    ("Thailand", ["Bangkok"]), ("Vietnam", ["Hanoi"]), ("Indonesia", ["Jakarta"]),
    ("the Philippines", ["Manila"]), ("Nigeria", ["Abuja"]), ("Morocco", ["Rabat"]),
    ("Pakistan", ["Islamabad"]), ("New Zealand", ["Wellington"]),
    ("Switzerland", ["Bern", "Berne"]), ("Colombia", ["Bogota", "Bogotá"]),
    ("Burkina Faso", ["Ouagadougou"]), ("Bhutan", ["Thimphu"]),
    ("Kyrgyzstan", ["Bishkek"]), ("Suriname", ["Paramaribo"]), ("Eritrea", ["Asmara"]),
    ("Vanuatu", ["Port Vila"]), ("Malawi", ["Lilongwe"]),
    ("Moldova", ["Chisinau", "Chișinău", "Kishinev"]), ("Laos", ["Vientiane"]),
    ("Mongolia", ["Ulaanbaatar", "Ulan Bator"]), ("Tajikistan", ["Dushanbe"]),
    ("Mauritania", ["Nouakchott"]), ("Palau", ["Ngerulmud"]),
    ("Kazakhstan", ["Astana"]), ("Myanmar", ["Naypyidaw", "Nay Pyi Taw", "Naypyitaw"]),
    ("Tanzania", ["Dodoma"]), ("Benin", ["Porto-Novo", "Porto Novo"]),
]
FICTIONAL_COUNTRIES = [
    "Veltoria", "Zembria", "Kastovia", "Oruvania", "Pellaria", "Drestonia",
    "Tarsenia", "Quellmark", "Ostrevia", "Bramoria", "Luskavia", "Vendaria",
    "Solmeria", "Karthane", "Morivale", "Estravia",
]

CITY_COUNTRY = [
    ("Timbuktu", ["Mali"]), ("Krakow", ["Poland"]), ("Marrakesh", ["Morocco"]),
    ("Cusco", ["Peru"]), ("Mandalay", ["Myanmar", "Burma"]), ("Porto", ["Portugal"]),
    ("Chiang Mai", ["Thailand"]), ("Salzburg", ["Austria"]), ("Gdansk", ["Poland"]),
    ("Tabriz", ["Iran"]), ("Samarkand", ["Uzbekistan"]), ("Valparaiso", ["Chile"]),
    ("Bergen", ["Norway"]), ("Mombasa", ["Kenya"]), ("Leipzig", ["Germany"]),
    ("Kyoto", ["Japan"]), ("Seville", ["Spain"]), ("Lyon", ["France"]),
    ("Da Nang", ["Vietnam"]), ("Arequipa", ["Peru"]),
]
FICTIONAL_CITIES = [
    "Kavrenhol", "Drossel Bay", "Ulmrath", "Pentavo", "Serravik", "Qorvath",
    "Maldenport", "Istravel", "Brunnhal", "Tolvareth",
]

# --------------------------------------------------------------------------- #
# Science
# --------------------------------------------------------------------------- #
SYMBOLS = [
    ("gold", ["Au"]), ("silver", ["Ag"]), ("iron", ["Fe"]), ("sodium", ["Na"]),
    ("potassium", ["K"]), ("lead", ["Pb"]), ("tin", ["Sn"]), ("mercury", ["Hg"]),
    ("tungsten", ["W"]), ("copper", ["Cu"]), ("helium", ["He"]), ("neon", ["Ne"]),
    ("zinc", ["Zn"]), ("antimony", ["Sb"]), ("osmium", ["Os"]), ("rhodium", ["Rh"]),
    ("selenium", ["Se"]), ("bismuth", ["Bi"]),
]
ATOMIC_NUMBERS = [
    ("hydrogen", 1), ("helium", 2), ("carbon", 6), ("nitrogen", 7), ("oxygen", 8),
    ("neon", 10), ("sodium", 11), ("chlorine", 17), ("calcium", 20), ("iron", 26),
    ("copper", 29), ("silver", 47), ("gold", 79), ("uranium", 92),
]
FICTIONAL_ELEMENTS = [
    "velorium", "trastonium", "kelvarium", "draxium", "ombrellium", "siltharium",
    "pravonium", "quenthium", "zorbium", "melkarium",
]
SCIENCE_MISC = [
    ("What is the boiling point of water at sea level in degrees Celsius?", ["100"]),
    ("How many bones are in the adult human body?", ["206"]),
    ("Which planet is known as the Red Planet?", ["Mars"]),
    ("Which gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "CO2"]),
    ("What is the largest planet in the Solar System?", ["Jupiter"]),
    ("What is the hardest naturally occurring mineral?", ["diamond"]),
    ("Which organ pumps blood through the human body?", ["heart"]),
    ("How many chromosomes does a typical human cell have?", ["46"]),
    ("Which subatomic particle carries a negative electric charge?", ["electron"]),
    ("Which organelle is known as the powerhouse of the cell?", ["mitochondria", "mitochondrion"]),
    ("What is the chemical formula of table salt?", ["NaCl"]),
    ("Which vitamin does human skin produce when exposed to sunlight?", ["vitamin D"]),
    ("Who proposed the theory of general relativity?", ["Einstein"]),
    ("Who discovered penicillin?", ["Fleming"]),
    ("What is the SI unit of force?", ["newton"]),
    ("What is the SI unit of electrical resistance?", ["ohm"]),
    ("Which planet is closest to the Sun?", ["Mercury"]),
    ("What is the chemical formula of water?", ["H2O"]),
    ("Which blood cells carry oxygen around the body?", ["red blood cells", "red", "erythrocytes"]),
    ("What is the freezing point of water in degrees Fahrenheit?", ["32"]),
]

# --------------------------------------------------------------------------- #
# Dates (template: "In what year ...")
# --------------------------------------------------------------------------- #
DATES = [
    ("did the Berlin Wall fall", 1989), ("did World War II end", 1945),
    ("did Apollo 11 land on the Moon", 1969), ("did the Titanic sink", 1912),
    ("did the French Revolution begin", 1789),
    ("was the United States Declaration of Independence signed", 1776),
    ("did Columbus first reach the Americas", 1492), ("did World War I begin", 1914),
    ("was the Soviet Union dissolved", 1991), ("was the Magna Carta sealed", 1215),
    ("was the Battle of Hastings fought", 1066), ("was the first iPhone released", 2007),
    ("did the Chernobyl disaster happen", 1986),
    ("was Nelson Mandela released from prison", 1990),
    ("were euro banknotes and coins introduced", 2002),
    ("was the Treaty of Versailles signed", 1919),
    ("did Constantinople fall to the Ottomans", 1453),
    ("did the Wright brothers make their first powered flight", 1903),
    ("was Sputnik 1 launched", 1957), ("did the Russian October Revolution take place", 1917),
    ("did the American Civil War end", 1865), ("did the Spanish Armada sail against England", 1588),
    ("was the Peace of Westphalia signed", 1648), ("was the Treaty of Tordesillas signed", 1494),
    ("was the Battle of Waterloo fought", 1815), ("did Krakatoa erupt catastrophically", 1883),
    ("were the first Nobel Prizes awarded", 1901), ("did the Suez Canal open", 1869),
    ("did the Panama Canal open", 1914), ("were the first modern Olympic Games held", 1896),
    ("did India gain independence from Britain", 1947),
    ("was the Hubble Space Telescope launched", 1990),
    ("did the eruption of Mount Vesuvius destroy Pompeii", 79),
    ("did the Great Fire of London happen", 1666),
]
FICTIONAL_DATES = [
    "was the Treaty of Velmora signed", "was the Battle of Karsk Hollow fought",
    "did the Great Dresmoor Flood happen", "did the Ostrevian Revolution begin",
    "was the Pellarian Accord signed", "did the Siege of Marrowgate end",
    "was the Tarsenian Republic founded", "did the Zembrian Civil War end",
    "was the Kavrenhol Canal opened", "did the Quellmark Uprising take place",
    "was the Bramorian Charter sealed", "did the eruption of Mount Oskarr happen",
]

# --------------------------------------------------------------------------- #
# Unanswerable (epistemically impossible; no ground truth exists or can exist yet)
# Written with the same templates as real questions where possible.
# --------------------------------------------------------------------------- #
UNANSWERABLE = [
    ("dates", "in_what_year", "In what year will the next magnitude 9 earthquake strike Japan?"),
    ("dates", "in_what_year", "In what year will the first human be born on Mars?"),
    ("dates", "in_what_year", "In what year will the last glacier in the Alps melt?"),
    ("dates", "in_what_year", "In what year did the first person ever tell a joke?"),
    ("dates", "in_what_year", "In what year was the first word spoken by a human?"),
    ("geography", "capital", "What is the capital of the country that will be founded in 2150?"),
    ("geography", "city_country", "Which country will the largest city of the year 2300 be in?"),
    ("science", "misc", "How many grains of sand are on Bondi Beach right now?"),
    ("science", "misc", "What is the exact number of hairs on Albert Einstein's head at his death?"),
    ("science", "misc", "How many ants are alive on Earth at this exact moment?"),
    ("science", "misc", "What will the global average temperature be on 1 January 2200?"),
    ("arithmetic", "misc", "What number am I thinking of right now?"),
    ("arithmetic", "misc", "What is the last digit of the largest prime number?"),
    ("dates", "in_what_year", "In what year will the Sun's next record sunspot appear?"),
    ("geography", "misc", "What is the name of the first city that humans will build on the Moon?"),
    ("science", "misc", "What is the name of the next chemical element that will be discovered?"),
]

# --------------------------------------------------------------------------- #
# Ambiguous (aleatoric: several defensible answers or missing context)
# --------------------------------------------------------------------------- #
AMBIGUOUS = [
    ("geography", "capital", "What is the capital of Georgia?", ["Tbilisi", "Atlanta"]),
    ("geography", "capital", "What is the capital of Bolivia?", ["Sucre", "La Paz"]),
    ("geography", "capital", "What is the capital of South Africa?", ["Pretoria", "Cape Town", "Bloemfontein"]),
    ("geography", "misc", "What is the largest city?", []),
    ("geography", "misc", "How far is it to the station?", []),
    ("dates", "in_what_year", "In what year did the war end?", []),
    ("dates", "misc", "Who is the current president?", []),
    ("dates", "misc", "Who won the match last night?", []),
    ("science", "misc", "What is Mercury?", ["planet", "element", "god"]),
    ("science", "misc", "What is the boiling point?", []),
    ("science", "misc", "What is the atomic number?", []),
    ("arithmetic", "misc", "What is the square root?", []),
    ("arithmetic", "misc", "What is half of it?", []),
    ("arithmetic", "misc", "What is the answer to the second question?", []),
    ("geography", "misc", "Which Springfield is the biggest?", []),
    ("science", "misc", "How long does it take to boil?", []),
]


def _arith(rng: random.Random) -> list[dict]:
    items = []
    specs = [
        ("add2", 15, lambda: (rng.randint(10, 49), rng.randint(10, 49)), "+"),
        ("add3", 12, lambda: (rng.randint(100, 899), rng.randint(100, 899)), "+"),
        ("sub2", 12, lambda: (rng.randint(50, 99), rng.randint(10, 49)), "-"),
        ("mul1", 12, lambda: (rng.randint(12, 49), rng.randint(3, 9)), "*"),
        ("mul2", 12, lambda: (rng.randint(13, 79), rng.randint(13, 79)), "*"),
    ]
    for tag, n, draw, op in specs:
        seen = set()
        while len([i for i in items if i["template"] == f"arith_{tag}"]) < n:
            a, b = draw()
            if (a, b) in seen:
                continue
            seen.add((a, b))
            val = a + b if op == "+" else a - b if op == "-" else a * b
            sym = {"+": "+", "-": "-", "*": "×"}[op]
            items.append(dict(
                question=f"What is {a} {sym} {b}?", answers=[str(val)],
                domain="arithmetic", kind="answerable", template=f"arith_{tag}",
                group=f"arith:{a}{op}{b}", answer_type="number",
            ))
    return items


def build() -> list[dict]:
    rng = random.Random(20260930)
    items: list[dict] = []
    add = items.append

    for c, ans in CAPITALS:
        add(dict(question=f"What is the capital of {c}?", answers=ans, domain="geography",
                 kind="answerable", template="capital", group=f"country:{c}"))
    for c in FICTIONAL_COUNTRIES:
        add(dict(question=f"What is the capital of {c}?", answers=[], domain="geography",
                 kind="fictional", template="capital", group=f"country:{c}"))
    for city, ans in CITY_COUNTRY:
        add(dict(question=f"Which country is the city of {city} in?", answers=ans,
                 domain="geography", kind="answerable", template="city_country",
                 group=f"city:{city}"))
    for city in FICTIONAL_CITIES:
        add(dict(question=f"Which country is the city of {city} in?", answers=[],
                 domain="geography", kind="fictional", template="city_country",
                 group=f"city:{city}"))

    for el, sym in SYMBOLS:
        add(dict(question=f"What is the chemical symbol for {el}?", answers=sym,
                 domain="science", kind="answerable", template="chem_symbol",
                 group=f"element:{el}", answer_type="symbol"))
    for el, z in ATOMIC_NUMBERS:
        add(dict(question=f"What is the atomic number of {el}?", answers=[str(z)],
                 domain="science", kind="answerable", template="atomic_number",
                 group=f"element:{el}", answer_type="number"))
    for i, el in enumerate(FICTIONAL_ELEMENTS):
        tmpl = "chem_symbol" if i % 2 == 0 else "atomic_number"
        q = (f"What is the chemical symbol for {el}?" if tmpl == "chem_symbol"
             else f"What is the atomic number of {el}?")
        add(dict(question=q, answers=[], domain="science", kind="fictional",
                 template=tmpl, group=f"element:{el}"))
    for i, (q, ans) in enumerate(SCIENCE_MISC):
        add(dict(question=q, answers=ans, domain="science", kind="answerable",
                 template="science_misc", group=f"science_misc:{i}",
                 answer_type="number" if ans[0].isdigit() else "text"))

    for ev, year in DATES:
        add(dict(question=f"In what year {ev}?", answers=[str(year)], domain="dates",
                 kind="answerable", template="in_what_year", group=f"event:{ev}",
                 answer_type="number"))
    for ev in FICTIONAL_DATES:
        add(dict(question=f"In what year {ev}?", answers=[], domain="dates",
                 kind="fictional", template="in_what_year", group=f"event:{ev}"))

    items.extend(_arith(rng))

    for i, (dom, tmpl, q) in enumerate(UNANSWERABLE):
        add(dict(question=q, answers=[], domain=dom, kind="unanswerable", template=tmpl,
                 group=f"unanswerable:{i}"))
    for i, (dom, tmpl, q, ans) in enumerate(AMBIGUOUS):
        add(dict(question=q, answers=ans, domain=dom, kind="ambiguous", template=tmpl,
                 group=f"ambiguous:{i}"))

    for i, it in enumerate(items):
        it.setdefault("answer_type", "text")
        it["id"] = f"qa{i:04d}"
        it["source"] = "bundled"
    return items


def main() -> None:
    items = build()
    with OUT.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(items)} items to {OUT}")
    print(Counter((it["domain"], it["kind"]) for it in items))


if __name__ == "__main__":
    main()
