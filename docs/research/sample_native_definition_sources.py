"""Copy a fixed, bounded source-only development sample; no QA or network.

The sampling plan is written before this program runs. Original sidecar bytes
are hash checked mechanically, but only the first three document records and
their (at most 400) source blocks are parsed. Output stays in ignored artifacts.
This verifies internal sidecar consistency, not the original archive anew.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/research-foundation/qasper-native-chunks-01"
OUTPUT = ROOT / "artifacts/research-foundation/offline-20261004/native-definition-sample-01"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    plan_path = OUTPUT / "sample_plan.json"
    plan = json.loads(plan_path.read_bytes())
    if plan["max_documents"] != 3 or plan["max_block_records"] != 400:
        raise ValueError("the fixed sample limit changed")
    for name in ("native_documents.jsonl", "native_blocks.jsonl"):
        if digest((SOURCE / name).read_bytes()) != plan["source_sha256"][name]:
            raise ValueError("source sidecar changed since sample selection")

    documents = []
    with (SOURCE / "native_documents.jsonl").open(encoding="utf-8") as stream:
        for _ in range(3):
            line = stream.readline(2_000_001)
            if not line or len(line) > 2_000_000:
                raise ValueError("missing or oversized selected document")
            documents.append(json.loads(line))
    by_id = {row["doc_id"]: row for row in documents}
    if len(by_id) != 3:
        raise ValueError("sample document IDs are not distinct")
    total = sum(row["native_field_count"] for row in documents)
    if not 3 <= total <= 400:
        raise ValueError("selected source fields exceed the frozen cap")
    blocks = []
    with (SOURCE / "native_blocks.jsonl").open(encoding="utf-8") as stream:
        for _ in range(total):
            line = stream.readline(200_001)
            if not line or len(line) > 200_000:
                raise ValueError("missing or oversized selected block")
            row = json.loads(line)
            if row["doc_id"] not in by_id:
                raise ValueError("source order differs; do not scan for substitutes")
            blocks.append(row)

    expected_order, paragraphs, summary = [], [], []
    for doc in documents:
        selected = [b for b in blocks if b["doc_id"] == doc["doc_id"]]
        expected_order.extend([doc["doc_id"]] * doc["native_field_count"])
        if len(selected) != doc["native_field_count"]:
            raise ValueError("incomplete selected document")
        native = doc["native_document_text"]
        canonical = doc["canonical_text"]
        if (digest(native.encode()) != doc["native_document_text_sha256"]
                or digest(canonical.encode()) != doc["canonical_text_sha256"]):
            raise ValueError("selected document text hash mismatch")
        if native != "\n\n".join(b["raw_native_text"] for b in selected):
            raise ValueError("selected native fields do not reconstruct document")
        used = set()
        paragraph_count = changed_count = 0
        for index, block in enumerate(selected):
            text = block["raw_native_text"]
            start, end = block["native_char_span"]
            if (block["block_order"] != index or not 0 <= start <= end <= len(native)
                    or native[start:end] != text
                    or digest(text.encode()) != block["raw_native_text_sha256"]):
                raise ValueError("native block identity/span/hash mismatch")
            if block["retrievable"]:
                unit_id = block["native_unit_id"]
                if not unit_id or unit_id in used:
                    raise ValueError("duplicate or missing native unit identity")
                used.add(unit_id)
                left, right = block["canonical_char_span"]
                if not 0 <= left < right <= len(canonical):
                    raise ValueError("canonical source span outside document")
                retrieval_text = canonical[left:right]
                if (digest(retrieval_text.encode()) != block["retrieval_text_sha256"]
                        or block["raw_equals_retrieval_text"] != (text == retrieval_text)):
                    raise ValueError("retrieval/native text binding mismatch")
            if block["kind"] == "paragraph" and text.strip():
                if not block["retrievable"]:
                    raise ValueError("nonempty paragraph lacks source identity")
                paragraphs.append({"id": block["native_unit_id"], "doc_id": doc["doc_id"],
                                   "order": index, "text": text})
                paragraph_count += 1
                changed_count += int(not block["raw_equals_retrieval_text"])
        if len(used) != doc["native_unit_count"]:
            raise ValueError("selected native-unit count mismatch")
        summary.append({"sample_ordinal": len(summary) + 1, "fields": len(selected),
                        "paragraph_units": paragraph_count,
                        "paragraphs_different_from_retrieval_text": changed_count})
    if [b["doc_id"] for b in blocks] != expected_order:
        raise ValueError("selected document groups are not contiguous in source order")

    sample = {"sample_plan_sha256": digest(plan_path.read_bytes()),
              "source_sha256": plan["source_sha256"], "documents": documents,
              "blocks": blocks, "paragraph_units": paragraphs, "summary": summary,
              "raw_archive_rechecked": False, "qa_or_answer_files_read": 0,
              "api_calls": 0, "parsed_document_records": 3,
              "parsed_block_records": len(blocks)}
    with (OUTPUT / "selected_sources.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(sample, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"sample_sha256": digest((OUTPUT / "selected_sources.json").read_bytes()),
                      "summary": summary, "qa_or_answer_files_read": 0, "api_calls": 0}))


if __name__ == "__main__":
    main()
