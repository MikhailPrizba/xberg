//! Regression tests for #263: the enrichment chokepoint must write its results onto the
//! document, not only into the side struct it returns.
//!
//! `enrich` used to return every stage's output solely in `EnrichedResult` and hand back
//! `EnrichedResult::extraction` untouched. `ExtractedDocument` is the only thing downstream
//! consumers ever see — it is what serializes to JSON, what the REST schema and the
//! language bindings expose, and what the post-processors and `split_and_extract` operate
//! on — so detected entities, page classifications and image captions produced by `enrich`
//! were invisible everywhere, and the `LlmUsage` for the classification and captioning
//! calls was discarded outright.
