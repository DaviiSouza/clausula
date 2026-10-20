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
