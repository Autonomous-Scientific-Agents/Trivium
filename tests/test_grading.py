from trivium.evaluation.grading import match_answer, normalize_answer


def test_normalize():
    assert normalize_answer("The Answer: MACE.") == "mace"


def test_numeric_tolerance_and_units():
    assert match_answer("0.12 eV", ["0.12 eV"], "numeric")
    assert match_answer("0.1201", ["0.12 eV"], "numeric")
    assert match_answer("5,000,000 conformations", ["5000000 conformations"], "numeric")
    assert not match_answer("0.21 eV", ["0.12 eV"], "numeric")


def test_boolean_and_choice():
    assert match_answer("true", ["True"], "boolean")
    assert match_answer("False.", ["False"], "boolean")
    assert not match_answer("true", ["False"], "boolean")
    assert match_answer("B", ["B", "Institution Y"], "choice")
    assert match_answer("B) Institution Y", ["B", "Institution Y"], "choice")
    assert match_answer("Institution Y", ["B", "Institution Y"], "choice")
    assert not match_answer("C", ["B", "Institution Y"], "choice")


def test_short_answer_containment_is_bounded():
    assert match_answer("It was the PBE functional", ["PBE"])
    assert not match_answer("PBE " + "and many other words " * 10, ["PBE"])
    assert not match_answer("unknown", ["PBE"])
