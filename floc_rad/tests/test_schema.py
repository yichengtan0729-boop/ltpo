from flocrad.schema import apply_edits, canonical_edits, parse_report, realize_state


def test_parse_and_round_trip():
    state = parse_report("Small left pleural effusion. No pneumothorax.")
    assert state["pleural_effusion"].presence == "present"
    assert state["pleural_effusion"].laterality == "left"
    assert state["pleural_effusion"].severity == "mild"
    assert state["pneumothorax"].presence == "absent"
    rendered = realize_state(state)
    reparsed = parse_report(rendered)
    assert reparsed["pleural_effusion"].laterality == "left"
    assert reparsed["pneumothorax"].presence == "absent"


def test_canonical_edits_are_executable():
    source = parse_report("Moderate right pleural effusion. No lung opacity.")
    target = parse_report("Small left pleural effusion. Left lower lung opacity.")
    edits = canonical_edits(source, target)
    corrected = apply_edits(source, edits)
    assert corrected["pleural_effusion"].laterality == "left"
    assert corrected["pleural_effusion"].severity == "mild"
    assert corrected["lung_opacity"].presence == "present"
    assert corrected["lung_opacity"].laterality == "left"
    assert corrected["lung_opacity"].location == "lower"
