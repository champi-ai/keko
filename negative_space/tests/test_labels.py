from negative_space.extract import label_item

ANS = dict(kind="answerable", answers=["Paris"], answer_type="text")


def test_known_fragile_unknown():
    assert label_item(ANS, "Paris", ["Paris", "It is Paris", "Paris."])["label"] == "known"
    r = label_item(ANS, "Paris", ["Paris", "Lyon", "Paris"])
    assert r["label"] == "fragile" and not r["error"]
    r = label_item(ANS, "Lyon", ["Paris", "Paris", "Paris"])
    assert r["label"] == "unknown" and r["error"] and r["hallucination"]
    r = label_item(ANS, "I don't know", [])
    assert r["label"] == "unknown" and r["error"] and not r["hallucination"]


def test_negative_controls():
    fic = dict(kind="fictional", answers=[], answer_type="text")
    r = label_item(fic, "The capital of Veltoria is Veltor City.", ["Veltor", "I don't know"])
    assert r["label"] == "unknown" and r["error"] and r["hallucination"]
    r = label_item(fic, "I don't know.", ["I don't know"])
    assert r["label"] == "unknown" and not r["error"]


def test_ambiguous_excluded_and_consistency():
    amb = dict(kind="ambiguous", answers=["Tbilisi", "Atlanta"], answer_type="text")
    assert label_item(amb, "Atlanta", ["Tbilisi"])["label"] == "ambiguous"
    r = label_item(ANS, "Paris", ["Paris", "Lyon", "Rome"])
    assert abs(r["consistency"] - 1 / 3) < 1e-9 and r["n_clusters"] == 3
