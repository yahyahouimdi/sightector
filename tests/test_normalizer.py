import pytest

from sightector.text import find_offensive_terms, normalize


@pytest.mark.parametrize("raw, expected", [
    ("You are a fu*ck idiot", "you are a fuck idiot"),
    ("iiidiot", "idiot"),
    ("f u c k you", "fuck you"),
    ("F.U.C.K off", "fuck off"),
    ("f**k you", "fuck you"),
    ("b*tch please", "bitch please"),
    ("you're an ID10T", "you're an idiot"),
    ("u r a m0r0n", "u r a moron"),
    ("\u0455tupid", "stupid"),                        # Cyrillic look-alike
    ("\U0001d41f\U0001d42e\U0001d41c\U0001d424 you", "fuck you"),  # math bold
    ("what the fuck  ing hell", "what the fucking hell"),
    ("ass hole", "asshole"),
    ("stfu", "shut the fuck up"),
    ("k i l l yourself", "kill yourself"),
])
def test_deobfuscation(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("clean", [
    "as i said , 4you",
    "class pass",
    "kill the process",
    "i have 3 cats and 2 dogs",
    "a b c d",
])
def test_clean_text_is_not_rewritten(clean):
    assert normalize(clean) == clean


def test_elongation_and_urls():
    assert normalize("sooooo good https://x.com @john") == "soo good url @user"


def test_find_offensive_terms():
    assert find_offensive_terms("you stupid idiot , stupid") == ["stupid", "idiot"]
