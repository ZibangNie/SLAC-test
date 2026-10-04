"""Analyze the fixed three-document source sample, without model inference.

Full native fields are used to detect repeated definitions, and only paragraph
mentions are retained for the final link inventory. The original paragraph-only
scope is also reported. Controls use identical whole source units; their metadata
are byte matched when possible, not tokenizer- or cost-matched.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from SLAC.retrieval.decision.conditional import (ConditionalState, SourceRelation, Unit, build_request)
from SLAC.retrieval.decision.explicit_definition import extract_explicit_definitions

BASE = ROOT / "artifacts/research-foundation/offline-20261004/native-definition-sample-01"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def qualified_id(doc_id: str, unit_id: str) -> str:
    # Sidecar block labels repeat across documents. JSON pair encoding is
    # unambiguous even if either original component contains a separator.
    return canonical([doc_id, unit_id]).decode()


def main() -> None:
    source_path = BASE / "selected_sources.json"
    sample = json.loads(source_path.read_bytes())
    paragraphs = tuple(Unit(qualified_id(row["doc_id"], row["id"]), row["text"], row["order"], row["doc_id"])
                       for row in sample["paragraph_units"])
    paragraph_ids = {u.id for u in paragraphs}
    # Source order is native block order, not pack selection order.
    full = tuple(Unit(qualified_id(b["doc_id"], b["native_unit_id"]), b["raw_native_text"], b["block_order"], b["doc_id"])
                 for b in sample["blocks"] if b["retrievable"] and b["raw_native_text"].strip())
    by_id = {unit.id: unit for unit in full}
    document_order = {d["doc_id"]: i for i, d in enumerate(sample["documents"])}
    para_result = extract_explicit_definitions(paragraphs)
    full_result = extract_explicit_definitions(full)
    links = sorted((link for link in full_result.links if link.mention.unit_id in paragraph_ids),
                   key=lambda link: (document_order[link.mention.doc_id], link.mention.order,
                                     link.mention.start, link.definition.anchor.order))
    unique = {}
    for link in links:
        link.verify(by_id)
        key = (link.mention.unit_id, link.definition.anchor.unit_id, link.definition.acronym)
        unique.setdefault(key, link)

    controls = []
    for link in list(unique.values())[:12]:
        relation = link.verify(by_id)
        dep = by_id[link.mention.unit_id]
        definition = by_id[link.definition.anchor.unit_id]
        # Fixed nearest-source-order control; never inspect a model decision.
        eligible = sorted((u for u in paragraphs if u.doc_id == dep.doc_id
                           and u.id not in {dep.id, definition.id}
                           and u.text not in {dep.text, definition.text}
                           and link.definition.acronym not in u.text),
                          key=lambda u: (abs(u.order - definition.order), u.order, u.id))
        row = {"link": asdict(link), "status": "no_control", "control_id": None}
        for distractor in eligible:
            false_relation = SourceRelation(dep.id, distractor.id, "definition",
                                            relation.anchor_start, relation.anchor_end)
            state = ConditionalState("What does " + link.definition.acronym + " stand for in this document?",
                                     (dep, distractor), definition, (relation,))
            settings = dict(endpoint_id="offline-no-transport", model_id="offline-unexecuted",
                            expected_response_model="offline-unexecuted",
                            token_counter=lambda text: len(text.encode()), max_tokens=65536,
                            max_payload_bytes=131072, counter_version="utf8-byte-counter-not-model-tokens-v1")
            try:
                plain = build_request(state, arm="plain_conditional", **settings)
                true = build_request(state, arm="relation_conditioned", **settings)
                wrong = build_request(ConditionalState(state.query, state.current_pack, state.candidate,
                                                       (false_relation,)),
                                      arm="relation_conditioned", **settings)
            except ValueError:
                row["status"] = "payload_or_evidence_contract_rejected"
                continue
            true_payload, wrong_payload, plain_payload = true.payload(), wrong.payload(), plain.payload()
            true_meta = true_payload["state"].pop("relations")
            wrong_meta = wrong_payload["state"].pop("relations")
            if true_payload != wrong_payload or true_payload != plain_payload:
                raise AssertionError("control altered visible source or question")
            if len(canonical(true_meta)) != len(canonical(wrong_meta)):
                row["status"] = "no_equal_metadata_byte_control"
                continue
            if len({plain.cache_key, true.cache_key, wrong.cache_key}) != 3:
                raise AssertionError("distinct relation arms collided")
            if not plain.rendered_evidence == true.rendered_evidence == wrong.rendered_evidence:
                raise AssertionError("control changed evidence rendering")
            row.update(status="built_unexecuted", control_id=distractor.id,
                       evidence_utf8_bytes=true.evidence_tokens,
                       relation_metadata_bytes=len(canonical(true_meta)),
                       plain_request=plain.payload(), true_request=true.payload(), wrong_request=wrong.payload(),
                       request_keys=[plain.cache_key, true.cache_key, wrong.cache_key])
            break
        controls.append(row)

    counts = []
    for doc in sample["documents"]:
        doc_id = doc["doc_id"]
        matching = [link for link in links if link.mention.doc_id == doc_id]
        edges = [link for link in unique.values() if link.mention.doc_id == doc_id]
        counts.append({"sample_ordinal": document_order[doc_id] + 1,
                       "full_scope_definition_occurrences": sum(d.anchor.doc_id == doc_id for d in full_result.definitions),
                       "full_scope_ambiguous_acronyms": sum(a.doc_id == doc_id for a in full_result.ambiguities),
                       "paragraph_scope_links": sum(l.mention.doc_id == doc_id for l in para_result.links),
                       "full_scope_paragraph_mention_links": len(matching),
                       "unique_unit_acronym_edges": len(edges),
                       "nonadjacent_native_block_edges": sum(l.mention.order - l.definition.anchor.order > 1 for l in edges)})
    signature = lambda link: (link.mention.unit_id, link.mention.start, link.definition.anchor.unit_id)
    para_signatures = {signature(l) for l in para_result.links}
    full_signatures = {signature(l) for l in links}
    aggregate = {"schema": "slac-native-definition-feasibility-v1", "documents": 3,
                 "native_fields": sample["parsed_block_records"], "paragraph_units": len(paragraphs),
                 "per_document": counts,
                 "paragraph_scope_links_removed_by_full_scope": len(para_signatures - full_signatures),
                 "full_scope_links_added_from_nonparagraph_definitions": len(full_signatures - para_signatures),
                 "manual_audit_planned_links": min(12, len(links)),
                 "control_unique_edges_attempted": len(controls),
                 "control_status": dict(sorted(Counter(row["status"] for row in controls).items())),
                 "source_texts_equal_across_arms": True, "metadata_match_unit": "UTF-8 bytes, not model tokens",
                 "model_inference": 0, "api_calls": 0, "qa_or_answer_files_read": 0,
                 "semantic_accuracy_or_rag_gain_measured": False,
                 "sample_sha256": digest(source_path.read_bytes()),
                 "source_sha256": sample["source_sha256"],
                 "implementation_sha256": {name: digest((ROOT / name).read_bytes()) for name in (
                     "SLAC/retrieval/decision/explicit_definition.py", "docs/research/run_native_definition_feasibility.py",
                     "docs/research/sample_native_definition_sources.py")}}
    private = {"aggregate": aggregate, "paragraph_scope": asdict(para_result),
               "full_scope": asdict(full_result), "retained_links": [asdict(l) for l in links],
               "controls": controls, "manual_audit_links": [asdict(l) for l in links[:12]]}
    with (BASE / "extraction.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(private, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
