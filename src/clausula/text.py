"""Turn raw Terms of Service text into model-ready sentences."""

import re

# The training data (LexGLUE unfair_tos) is lowercased and tokenized:
# "We've changed" -> "we 've changed", "terms." -> "terms .". Real ToS pasted
# by users are not, so we apply the same normalization before predicting.
_PUNCT = re.compile(r"([.,;:!?()\[\]\"“”])")
_CONTRACTION = re.compile(r"(\w)('(?:s|ve|re|ll|d|m))\b")
_NOT = re.compile(r"(\w)n't\b")
_SPACES = re.compile(r"\s+")

# Split after sentence-ending punctuation followed by whitespace, or on line breaks.
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\n+")
_BULLET = re.compile(r"^\s*(?:[-•*·]|\(?\w{1,3}[.)])\s+")

MIN_WORDS = 4


def normalize(sentence: str) -> str:
    s = sentence.lower().replace("’", "'")
    s = _NOT.sub(r"\1 n't", s)
    s = _CONTRACTION.sub(r"\1 \2", s)
    s = _PUNCT.sub(r" \1 ", s)
    return _SPACES.sub(" ", s).strip()


def split_sentences(text: str) -> list[str]:
    """Split a document into sentences, dropping bullets markers and short fragments (headings, numbers)."""
    sentences = []
    for chunk in _SENTENCE_END.split(text):
        chunk = _BULLET.sub("", chunk).strip()
        if len(chunk.split()) >= MIN_WORDS:
            sentences.append(chunk)
    return sentences
