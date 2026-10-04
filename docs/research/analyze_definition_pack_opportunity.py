"""Bounded natural-query definition-gap diagnostic; no quality labels or API."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/research-foundation/offline-20261004"
OUT = BASE / "definition-pack-opportunity-01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(units, selected):
    return "\n\n".join(f"[{uid}]\n{units[uid]['text']}"
                       for uid in sorted(selected, key=lambda uid: units[uid]["order"]))


def choose(units, ranking, count, budget=1024, k=3):
    if len(set(ranking)) != len(ranking) or not set(ranking) <= units.keys():
        raise ValueError("ranking contains duplicate or unknown source IDs")
    selected, seen = [], set()
    for uid in ranking:
        text = units[uid]["native_text"]
        if text in seen or count(selected + [uid]) > budget:
            continue
        selected.append(uid)
        seen.add(text)
        if len(selected) == k:
            break
    return sorted(selected, key=lambda uid: units[uid]["order"])


def gaps(units, selected, edges, candidates, seeds, query, count, budget=1024, k=3):
    """Group selected-mention gaps by missing definition unit, not occurrence.

    Replacements retain EVERY selected mention triggering that target. These
    are feasible alternatives, not choices or claims that dropped text is useless.
    """
    selected_set = set(selected)
    texts = {units[u]["native_text"] for u in selected}
    missing = {}
    for edge in edges:
        dep, target, acronym = edge
        if dep not in units or target not in units:
            raise ValueError("relation endpoint missing")
        if dep in selected_set and target not in selected_set and units[target]["native_text"] not in texts:
            group = missing.setdefault(target, {"dependents": set(), "acronyms": set()})
            group["dependents"].add(dep)
            group["acronyms"].add(acronym)
    rows = []
    for target, group in sorted(missing.items(), key=lambda item: units[item[0]]["order"]):
        appended = selected + [target]
        append_tokens = count(appended)
        alternatives = []
        for removed in selected:
            if removed in group["dependents"]:
                continue
            replacement = [u for u in selected if u != removed] + [target]
            tokens = count(replacement)
            if len(replacement) <= k and tokens <= budget:
                alternatives.append({"removed_id": removed, "selected_ids": sorted(replacement, key=lambda u: units[u]["order"]),
                                     "evidence_tokens": tokens})
        rows.append({"target_id": target, "dependent_ids": sorted(group["dependents"]),
                     "acronyms": sorted(group["acronyms"]),
                     "exact_acronym_in_query": any(re.search(r"(?<![A-Za-z0-9_])" + re.escape(a) + r"(?![A-Za-z0-9_])", query)
                                                    for a in group["acronyms"]),
                     "in_original_candidates": target in candidates, "in_original_dense_seeds": target in seeds,
                     "adjacent_to_any_selected_native_unit": any(abs(units[target]["order"] - units[u]["order"]) == 1 for u in selected),
                     "appended_evidence_tokens": append_tokens, "append_fits_tokens": append_tokens <= budget,
                     "append_fits_k_and_tokens": len(appended) <= k and append_tokens <= budget,
                     "single_replacements": alternatives})
    return rows


def main():
    # Local tokenizer only; no remote calls, model weights or QA evaluation.
    from transformers import AutoTokenizer
    inputs_path = OUT / "selected_inputs.json"
    inputs = json.loads(inputs_path.read_bytes())
    score_path = BASE / "definition-sample-reranker-run-01/scores.json"
    scores = json.loads(score_path.read_bytes())
    config_path = ROOT / "artifacts/research-foundation/qasper-reranker-02/experiment_config.json"
    config = json.loads(config_path.read_bytes())
    tokenizer_path = Path(config["bge_tokenizer"])
    manifest = json.loads((ROOT / "artifacts/research-foundation/qasper-relation-prepared-01/manifest.json").read_bytes())
    tokenizer_hashes = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"):
        p = tokenizer_path / name
        expected = manifest["input_sha256"].get(str(p))
        if expected is None or sha(p) != expected:
            raise ValueError("evidence tokenizer differs from original pilot")
        tokenizer_hashes[name] = expected
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    relation_path = BASE / "native-definition-sample-01/extraction.json"
    relations = json.loads(relation_path.read_bytes())["retained_links"]
    rankings = {(r["doc_id"], r["question_id"]): r["ranked_ids"] for r in scores["rankings"]}
    if set(rankings) != {(q["doc_id"], q["question_id"]) for q in inputs["queries"]}:
        raise ValueError("bounded score coverage differs from selected query sample")
    records = []
    for ordinal, query in enumerate(inputs["queries"], 1):
        doc = query["doc_id"]
        units = {u["unit_id"]: u for u in inputs["documents"][doc]}
        edges = set()
        for link in relations:
            if link["mention"]["doc_id"] != doc:
                continue
            dep_doc, dep = json.loads(link["mention"]["unit_id"])
            pre_doc, pre = json.loads(link["definition"]["anchor"]["unit_id"])
            if dep_doc != doc or pre_doc != doc:
                raise ValueError("source-qualified relation mismatch")
            for uid, anchor in ((dep, link["mention"]), (pre, link["definition"]["anchor"])):
                text = units[uid]["native_text"]
                if hashlib.sha256(text.encode()).hexdigest() != anchor["text_sha256"] or text[anchor["start"]:anchor["end"]] != anchor["text"]:
                    raise ValueError("native witness does not match selected source")
            edges.add((dep, pre, link["definition"]["acronym"]))
        cache = {(): 0}
        def count(selected):
            key = tuple(sorted(selected))
            if key not in cache:
                cache[key] = len(tokenizer.encode(render(units, selected), add_special_tokens=True, truncation=False))
            return cache[key]
        row = {"query_ordinal": ordinal, "doc_id": doc, "question_id": query["question_id"],
               "query": query["query"], "candidate_ids": query["candidate_ids"], "methods": {}}
        method_rankings = {"dense": query["ranked_ids"], "local_bge_reranker": rankings[doc, query["question_id"]]}
        for method, ranking in method_rankings.items():
            if len(ranking) != len(query["candidate_ids"]) or set(ranking) != set(query["candidate_ids"]):
                raise ValueError("same-pool comparison changed candidates")
            selected = choose(units, ranking, count)
            missing = gaps(units, selected, edges, query["candidate_ids"], query["seed_ids"], query["query"], count)
            for gap in missing:
                gap["rank_in_original_pool"] = ranking.index(gap["target_id"]) + 1 if gap["target_id"] in ranking else None
            row["methods"][method] = {"ranked_ids": ranking, "selected_ids": selected,
                                      "evidence_tokens": count(selected),
                                      "pack_sha256": hashlib.sha256(render(units, selected).encode()).hexdigest(), "gaps": missing}
        for method, value in row["methods"].items():
            other = row["methods"]["local_bge_reranker" if method == "dense" else "dense"]
            for gap in value["gaps"]:
                gap["definition_present_in_other_baseline"] = gap["target_id"] in other["selected_ids"]
        row["whole_pack_token_measurements"] = [{"selected_ids": list(k), "tokens": v} for k, v in sorted(cache.items())]
        records.append(row)
    aggregate = {"queries": len(records), "documents": len(inputs["documents"]), "api_calls": 0,
                 "new_local_reranker_pairs": len(scores["pair_scores"]), "quality_labels_read": False,
                 "budget": {"whole_render_BGE_tokens": 1024, "max_units": 3}, "methods": {}}
    for method in ("dense", "local_bge_reranker"):
        method_rows = [r["methods"][method] for r in records]
        all_gaps = [g for r in method_rows for g in r["gaps"]]
        aggregate["methods"][method] = {"packs": len(method_rows), "packs_with_gap": sum(bool(r["gaps"]) for r in method_rows),
          "missing_definition_targets": len(all_gaps), "targets_in_candidates": sum(g["in_original_candidates"] for g in all_gaps),
          "targets_in_dense_seeds": sum(g["in_original_dense_seeds"] for g in all_gaps),
          "targets_adjacent_to_any_selected": sum(g["adjacent_to_any_selected_native_unit"] for g in all_gaps),
          "targets_present_in_other_baseline": sum(g["definition_present_in_other_baseline"] for g in all_gaps),
          "targets_with_acronym_in_query": sum(g["exact_acronym_in_query"] for g in all_gaps),
          "targets_append_fits_tokens": sum(g["append_fits_tokens"] for g in all_gaps),
          "targets_append_fits_k_and_tokens": sum(g["append_fits_k_and_tokens"] for g in all_gaps),
          "targets_with_feasible_single_replacement": sum(bool(g["single_replacements"]) for g in all_gaps),
          "pack_tokens": [r["evidence_tokens"] for r in method_rows]}
    result = {"aggregate": aggregate, "records": records, "evidence_tokenizer_sha256": tokenizer_hashes,
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (inputs_path, score_path, relation_path, Path(__file__))}}
    with (OUT / "opportunities.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
