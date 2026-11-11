# Lesson 3: Chunking (developer view)

Corpus grew from a 486-char test doc to a 12KB employee handbook
(11 sections, ~4K tokens). All experiments run through a R&D harness
(`03_rag_basic.py`): naive_chunking → embed → metrics → retrieve → answer,
with chunking config passed as parameters (not module constants).

## Experiment: multi-hop question across 3 chunk configs

One multi-hop question ("salary payday + resignation notice period" —
answers live in two distant sections):

| Config   | Chunks | Orphans | Multi-hop answer | input_tokens |
| -------- | ------ | ------- | ---------------- | ------------ |
| 200/50   | 74     | 12      | ✅ both facts    | 177          |
| 500/100  | 36     | 12      | ❌ missed payday | 339          |
| 1000/200 | 11     | 0       | ✅ both facts    | 655          |

## Findings

1. **Retrieval quality is NOT monotonic in chunk size** (✅ → ❌ → ✅).
   The 500/100 failure was a boundary-placement problem: a cut separated
   the payday fact from the context that makes it retrievable. Chunking is
   a measured hyperparameter, not a rule of thumb.
2. **Cost is the only thing that scales linearly** (177 → 339 → 655 tokens,
   3.7×). Even the largest config stays ~6× cheaper than stuffing the full
   document (Lesson 1 baseline).
3. **The orphan metric measures a different disease per config**: at 200/50
   orphans are orphaned section headers; at 500/100 they are boundary
   fragments. A metric can silently change meaning when configs change —
   inspect actual flagged chunks before trusting the number.
4. **Index size dwarfs text size**: 74 chunks × 3072 dims × 4B ≈ 909KB to
   index a 12KB document (~75×). Storage cost is a real production factor.
5. RAG can fail **silently**: at 500/100 the model confidently said the
   payday info "is not in the document" — a retrieval failure, not a
   knowledge gap. Silent failure is why evals (golden sets, hit rate) must
   exist.

## Methodology caveats (to be fixed before drawing final conclusions)

- Results above are from a **single question (n=1)**. A full question set
  × configs (hit rate) is required before claiming any config is "best".
- Section-level chunking (split on `## ` headers, prepend header to chunk
  text vs. store as metadata) is deferred to the next lesson — same size
  as 1000/200 but semantic boundaries, isolating "boundary quality" from
  "chunk size".

## Who decides the chunking strategy?

Chunking is where business understanding meets measurement — the machine
only executes the cutting:

| Task | Who | Nature |
| ---- | --- | ------ |
| Choose strategy (recursive / structural / semantic) | Human, based on document type | Design decision |
| Choose size / overlap | Human, based on how users will ask | Hyperparameter, must be measured |
| Cut thousands of chunks | Machine (splitter) | Operation |
| Re-index when corpus changes | Automated pipeline | Operations |

Document type drives the strategy: Markdown handbooks → section-based;
legal contracts → clause-based with metadata; code → function/class;
tables → keep whole, never split cells.
