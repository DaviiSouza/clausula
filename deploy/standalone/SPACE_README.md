---
title: Clausula
emoji: ⚖️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Find potentially unfair clauses in Terms of Service
models:
  - DaviiiiS/clausula-legal-bert
datasets:
  - coastalcph/lex_glue
---

# Clausula

Paste a Terms of Service (or upload a PDF) and see the potentially unfair clauses highlighted, with category, confidence and a plain-language explanation.

- Model: [DaviiiiS/clausula-legal-bert](https://huggingface.co/DaviiiiS/clausula-legal-bert)
- Code: [github.com/DaviiSouza/clausula](https://github.com/DaviiSouza/clausula)

Runs on a free CPU: a full ToS takes ~10 seconds. Not legal advice.
