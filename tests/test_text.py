from clausula.text import normalize, split_sentences


def test_normalize_matches_training_format():
    assert normalize("We've changed the Terms. You don't agree?") == "we 've changed the terms . you do n't agree ?"


def test_split_sentences_drops_headings_and_bullets():
    text = "1. Acceptance\nBy using the app, you agree to these Terms. We may change them at any time.\n- You must be 16 or older to sign up."
    assert split_sentences(text) == [
        "By using the app, you agree to these Terms.",
        "We may change them at any time.",
        "You must be 16 or older to sign up.",
    ]
