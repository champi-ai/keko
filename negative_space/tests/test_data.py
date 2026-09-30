from collections import defaultdict

from negative_space.config import DataConfig
from negative_space.data import (answer_key, dedup, is_abstention, is_correct, load_bundled,
                                 load_items, normalize_answer, stratified_subset)


def test_normalize():
    assert normalize_answer("The capital is Brasília.") == "capital is brasilia"
    assert normalize_answer("It has TWO parts, 1,000 total") == "it has 2 parts 1000 total"


def test_number_matching():
    item = dict(answers=["42"], answer_type="number", domain="arithmetic")
    assert is_correct("17 + 25 = 42", item)
    assert not is_correct("17 + 25 = 41", item)
    assert is_correct("42", item)
    year = dict(answers=["1989"], answer_type="number", domain="dates")
    assert is_correct("The Berlin Wall fell in 1989.", year)
    assert not is_correct("It fell in 1991.", year)


def test_symbol_and_text_matching():
    k = dict(answers=["K"], answer_type="symbol")
    assert is_correct("The chemical symbol for potassium is K.", k)
    assert not is_correct("It is Kr.", k)
    t = dict(answers=["Ulaanbaatar", "Ulan Bator"], answer_type="text")
    assert is_correct("The capital of Mongolia is Ulan Bator.", t)
    assert not is_correct("Ulan", t)


def test_abstention():
    assert is_abstention("I don't know.")
    assert is_abstention("I’m not sure about that")
    assert is_abstention("Veltoria is a fictional country.")
    assert is_abstention("")
    assert not is_abstention("Paris")
    item = dict(answers=["Paris"], answer_type="text")
    assert not is_correct("I don't know, maybe Paris", item)


def test_answer_key_groups_rephrasings():
    item = dict(answers=["Paris"], answer_type="text")
    assert answer_key("Paris.", item) == answer_key("The capital is Paris", item) == "<CORRECT>"
    assert answer_key("I don't know", item) == "<ABSTAIN>"
    assert answer_key("Lyon", item) != answer_key("Marseille", item)


def test_bundled_controls_are_template_matched():
    items = load_bundled("negative_space/qa_data/qa_bundled.jsonl")
    assert len(items) >= 250
    by_t = defaultdict(set)
    for it in items:
        by_t[it["template"]].add(it["kind"])
        if it["kind"] == "answerable":
            assert it["answers"]
    for t in ("capital", "city_country", "chem_symbol", "atomic_number", "in_what_year"):
        assert {"answerable", "fictional"} <= by_t[t], t
    assert len({it["id"] for it in items}) == len(items)


def test_dedup_and_subset():
    items = [dict(id=str(i), question=q, domain="d", kind="answerable") for i, q in
             enumerate(["What is the capital of France?", "What is the capital of France ?",
                        "What is the capital of Spain?"])]
    assert len(dedup(items, 0.9)) == 2
    full = load_items(DataConfig(n_keko_honest=5), seed=0)
    sub = stratified_subset(full, 60, seed=0)
    assert len(sub) <= 60 and len({(i["domain"], i["kind"]) for i in sub}) >= 8
