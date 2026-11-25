# Research Summary: Retrieval-Augmented Generation

## Retrieval

Retrieval-augmented generation (RAG) grounds a language model's answers in
documents retrieved at question time. The standard pipeline embeds chunks of
the corpus into a vector database, embeds the question with the same model,
and retrieves the nearest chunks by cosine similarity. Retrieval quality
dominates end-to-end quality: if the right passage is not retrieved, no
amount of prompt engineering will fix the answer.

## Chunking

Chunk size trades context against precision. Very large chunks dilute the
embedding signal; very small chunks lose surrounding context. A common
default for personal-scale corpora is around 500-800 tokens with 10-15
percent overlap. Overlap prevents sentences at chunk boundaries from being
cut in half. Metadata (filename, page, section) must be preserved on every
chunk so answers can cite their sources.

## Citations and Trust

A trustworthy RAG answer cites the exact passage it used. The safest
implementation validates model-produced citations against the set of chunks
that were actually retrieved: any citation referencing a chunk that was not
retrieved is discarded, and if no valid citation remains the assistant
abstains rather than guessing. Retrieved text must be treated as untrusted
data, because a malicious document could otherwise inject instructions into
the prompt.

## Limitations observed in practice

Lexical mismatch remains the top failure mode for small corpora: when the
question uses different words than the document, pure vector retrieval can
miss the passage. Hybrid keyword-plus-vector retrieval mitigates this but is
out of scope for the current project version. Another observed failure mode
is over-confident answers on marginally related chunks, which is why a
minimum similarity threshold and an abstention path are recommended.
