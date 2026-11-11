# Lesson 4 (Part 0): In-context with LangChain — the full-doc baseline

Same strategy as `01` (stuff the whole document into the prompt), but through
LangChain instead of the raw SDK. Its main job in this lab: provide the
**full-document baseline** that every RAG config is compared against.

## Baseline number

With the 12KB handbook (~11 sections), one question costs:

```text
input_tokens: 3008
```

Compare with the RAG configs from Lesson 3 (177–655 tokens): in-context is
**4.5–17× more expensive** for the same corpus. This number goes into the
Lesson 4 comparison table as the `full-doc` row.

## Raw SDK vs LangChain (having written both paths)

| Aspect           | Raw SDK (`01`)                             | LangChain (`04`)                                                      |
| ---------------- | ------------------------------------------ | --------------------------------------------------------------------- |
| Boilerplate      | Client setup, prompt assembly all explicit | One `ChatGoogleGenerativeAI` object, `.invoke()`                      |
| Token usage      | Count manually via API                     | `response.usage_metadata` for free                                    |
| Abstraction cost | None — everything visible                  | Hidden prompt plumbing, extra dependency                              |
| When it pays off | One-off scripts, learning                  | Swappable components (splitters, stores) — used in Lesson 3/4 harness |

The honest takeaway: LangChain is not "better", it moves code from your file
into a library. That is a win when the library's components are the ones you
need (loaders, splitters, vector stores), and dead weight otherwise.

## Notes for the rest of Lesson 4

Parts 1–3 (config object, section-level chunking, hit-rate over a full
question set) are tracked in the plan; the baseline row above is their
reference point.
