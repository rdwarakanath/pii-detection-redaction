"""
presidio_bert_combo.py - BERT (via Presidio) + your rule helpers, merged.

pip install presidio-analyzer spacy transformers torch
python -m spacy download en_core_web_sm

Flow:
  1. BERT NER (inside Presidio) finds PERSON on every string      -> pass 1
  2. BERT names feed the NameRegistry (surname propagation)
  3. Rules (pii_helpers) + registry + BERT are merged             -> pass 2
"""
from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from pii_helpers import (Span, NameRegistry, FirstNameChecker, build_name_registry, detect_pii,
                         apply_allowlist, name_parts, DEFAULT_ALLOWLIST)

# --------------------------------------------------------------------------- #
# 1. Presidio analyzer that uses BERT as its NLP engine
# --------------------------------------------------------------------------- #

BERT_MODEL = "dslim/bert-base-NER"  # or "StanfordAIMI/stanford-deidentifier-base" (de-id tuned)


def build_analyzer(model_name: str = BERT_MODEL) -> AnalyzerEngine:
    cfg = {
        "nlp_engine_name": "transformers",
        "models": [{
            "lang_code": "en",
            "model_name": {"spacy": "en_core_web_sm", "transformers": model_name},
        }],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON", "PERSON": "PERSON",
                "LOC": "LOCATION", "ORG": "ORGANIZATION", "MISC": "O",
            },
            # ORG/LOC would redact company names and cities: leave to rules/allowlist
            "labels_to_ignore": ["O", "MISC", "ORG", "LOC"],
            "aggregation_strategy": "simple",
            "stride": 16,
            "alignment_mode": "expand",
        },
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=cfg).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])


def bert_spans(analyzer: AnalyzerEngine, text: str, threshold: float = 0.4) -> list[Span]:
    # Only ask Presidio for PERSON: your rules are stronger for structured IDs.
    res = analyzer.analyze(text=text, language="en", entities=["PERSON"], score_threshold=threshold)
    return [Span(r.start, r.end, r.entity_type, round(r.score, 2), "bert") for r in res]


# --------------------------------------------------------------------------- #
# 2. Merge policy
# --------------------------------------------------------------------------- #

def merge_spans(rule_spans: list[Span], bert: list[Span], min_score: float = 0.5) -> list[Span]:
    """
    - Structured entities (email, phone, SSN...) from rules always win over BERT.
    - PERSON from rules and BERT that overlap are UNIONED (fixes boundary
      disagreements, e.g. rules='Achebe', BERT='Miriam T. Achebe').
    - A BERT-only PERSON is kept if score >= min_score (fail closed).
    """
    structured = [s for s in rule_spans if s.entity != "PERSON"]
    persons = [s for s in rule_spans if s.entity == "PERSON"] + bert

    # union overlapping PERSON spans
    persons.sort(key=lambda s: s.start)
    merged: list[Span] = []
    for s in persons:
        if merged and s.start <= merged[-1].end:
            p = merged[-1]
            merged[-1] = Span(p.start, max(p.end, s.end), "PERSON",
                              max(p.score, s.score) + (0.05 if p.source != s.source else 0),
                              f"{p.source}+{s.source}")
        else:
            merged.append(s)

    # structured spans beat PERSON spans where they overlap
    out = list(structured)
    for p in merged:
        if p.score >= min_score and all(p.end <= q.start or p.start >= q.end for q in structured):
            out.append(Span(p.start, p.end, p.entity, min(p.score, 1.0), p.source))
    # structured vs structured: highest score wins
    out.sort(key=lambda s: (-s.score, -(s.end - s.start)))
    keep: list[Span] = []
    for s in out:
        if all(s.end <= k.start or s.start >= k.end for k in keep):
            keep.append(s)
    return sorted(keep, key=lambda s: s.start)


# --------------------------------------------------------------------------- #
# 3. Two-pass document pipeline
# --------------------------------------------------------------------------- #

def run_document(texts: list[str], analyzer: AnalyzerEngine | None = None,
                 min_score: float = 0.5) -> list[list[Span]]:
    analyzer = analyzer or build_analyzer()

    # pass 1: BERT on everything, learn people
    bert_all = [bert_spans(analyzer, t) for t in texts]
    registry: NameRegistry = build_name_registry(texts)          # emails + rule names
    for t, bs in zip(texts, bert_all):                            # + BERT names
        for sp in bs:
            toks = t[sp.start:sp.end].split()
            if len(toks) >= 2:
                registry.add(toks[-1], toks[0][0])

    # pass 2: rules + registry, merged with BERT
    results = []
    for t, bs in zip(texts, bert_all):
        rules = detect_pii(t, registry, min_score=min_score, allowlist=())
        merged = merge_spans(rules, bs, min_score)
        results.append(apply_allowlist(t, merged, DEFAULT_ALLOWLIST))
    return results


if __name__ == "__main__":
    demo = ["Policy owner: Miriam T. Achebe (EMP-38104), m.achebe@cadencefg.com, +1 (212) 555-0147",
            "Contact Helena Vorstermann on +44 20 7555 0163; Achebe will confirm."]
    for t, spans in zip(demo, run_document(demo)):
        print([(t[s.start:s.end], s.entity, s.source) for s in spans])
