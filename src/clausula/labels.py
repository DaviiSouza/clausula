"""Unfair clause categories from LexGLUE `unfair_tos` (CLAUDETTE corpus).

The order of LABELS matches the dataset's label ids (0..7). A sentence with no
label is considered fair.
"""

LABELS = [
    "Limitation of liability",
    "Unilateral termination",
    "Unilateral change",
    "Content removal",
    "Contract by using",
    "Choice of law",
    "Jurisdiction",
    "Arbitration",
]

# Plain-language explanations shown to end users in the app.
EXPLANATIONS = {
    "Limitation of liability": "The provider limits or excludes its responsibility for damages, losses or service failures.",
    "Unilateral termination": "The provider can suspend or close your account at any time, possibly without a clear reason.",
    "Unilateral change": "The provider can change the terms or the service whenever it wants, often without asking you.",
    "Content removal": "The provider can delete or modify content you uploaded without notice.",
    "Contract by using": "You are bound by the terms simply by using the service, without explicitly agreeing.",
    "Choice of law": "The contract is governed by the laws of a country or state chosen by the provider.",
    "Jurisdiction": "Disputes must be resolved in courts chosen by the provider, possibly far from you.",
    "Arbitration": "Disputes go to private arbitration instead of a court, limiting your legal options.",
}

# Short, plain-language headline for each finding (shown as the title of each point in the app).
HEADLINES = {
    "Limitation of liability": "They limit their responsibility if something goes wrong",
    "Unilateral termination": "They can close your account at any time",
    "Unilateral change": "They can change the terms whenever they want",
    "Content removal": "They can delete your content without notice",
    "Contract by using": "Just using the service means you accept the terms",
    "Choice of law": "A law chosen by them governs the contract",
    "Jurisdiction": "Disputes go to courts they choose",
    "Arbitration": "You give up your right to go to court",
}

# Editorial choice: how much each kind of clause can hurt a regular user.
SEVERITY = {
    "Arbitration": "high",
    "Limitation of liability": "high",
    "Unilateral termination": "high",
    "Unilateral change": "medium",
    "Content removal": "medium",
    "Jurisdiction": "medium",
    "Choice of law": "low",
    "Contract by using": "low",
}
_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def grade(categories) -> str:
    """A (fair) to E (many serious red flags), from the *kinds* of unfair clause found, weighted by severity."""
    score = sum(_WEIGHT[SEVERITY[c]] for c in set(categories))
    for letter, limit in (("A", 0), ("B", 3), ("C", 7), ("D", 11)):
        if score <= limit:
            return letter
    return "E"
