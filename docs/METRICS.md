# Metrics

PromptLint focuses on robustness: consistency across models and sampling, plus constraint adherence. It does not measure factual correctness.

## Default measures
- **Constraint adherence**: average constraint satisfaction across all runs.
- **Cross-model consistency**: pairwise similarity between model outputs at the same sampling settings.
- **Cross-temperature stability**: pairwise similarity between outputs of the same model across temperatures.
- **Task alignment**: checks output format via `metadata.expected_format` (or falls back to constraints).
- **Success rate**: ratio of successful provider calls.

## Similarity signals
For consistency/stability, PromptLint combines:
- character sequence similarity
- token Jaccard similarity
- structural similarity (line counts, bullet ratios, JSON presence, code blocks)
- optional embedding cosine similarity (if embeddings are configured)

When embeddings are available, the group similarity uses `0.7 * cosine + 0.3 * text_similarity`.

## Aggregation
Measures are combined using a weighted geometric mean with a stability penalty. Weights are defined by the evaluator and stored in `Measure.details.weight`.
