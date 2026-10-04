"""Synthetic tests for numeric cache joining without model clients."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import run_cached_additions as runner


def endpoint(cache="c", units=None, tokens=0):
    return {"family_id": "family", "doc_id": "doc", "question_id": "question",
            "selected_ids": units or [], "pack_sha256": "pack-" + cache,
            "cache_key": cache, "actual_evidence_tokens": tokens}


def raw(*rows):
    return b"\n".join(json.dumps(row).encode() for row in rows)


def test_only_required_scores_are_projected_and_answer_text_is_discarded():
    empty = endpoint()
    rows = raw({**empty, "official_answer_f1": 0.5, "predicted_answer": "unused"},
               {**empty, "official_answer_f1": 0.5, "method": "alias"},
               {"cache_key": "outside", "official_answer_f1": "not inspected"})
    scores, counts = runner.project_score_rows(rows, {"c": empty})
    assert scores == {"c": 0.5} and counts == {"c": 2}
    assert "unused" not in repr(scores)


@pytest.mark.parametrize("changed", [
    {"question_id": "other"}, {"selected_ids": ["unit"]},
    {"pack_sha256": "other"}, {"actual_evidence_tokens": 7},
    {"official_answer_f1": True}, {"official_answer_f1": float("nan")},
    {"official_answer_f1": 1.01},
])
def test_required_score_cannot_drift(changed):
    frozen = endpoint()
    row = {**frozen, "official_answer_f1": 0.5, **changed}
    with pytest.raises(ValueError):
        runner.project_score_rows(raw(row), {"c": frozen})


def test_conflicting_duplicate_or_missing_endpoint_fails():
    frozen = endpoint()
    with pytest.raises(ValueError, match="conflicting"):
        runner.project_score_rows(raw({**frozen, "official_answer_f1": 0},
                                     {**frozen, "official_answer_f1": 1}), {"c": frozen})
    with pytest.raises(ValueError, match="conflicting"):
        runner.merge_scores([{"c": 0}, {"c": 1}], {"c": frozen})
    with pytest.raises(ValueError, match="missing"):
        runner.merge_scores([{}], {"c": frozen})
    assert runner.merge_scores([{"c": 0}, {"c": 0}], {"c": frozen}) == {"c": 0}


def test_judgments_remain_raw_and_do_not_filter_edges():
    result = runner.project_judgments({"jev": {"t": "no", "outside": "bad"}},
                                    {"t": {"yes": 0.7}}, {"t"})
    assert result == {"t": {"support_label": "no", "raw_yes_score": 0.7}}
    manifest = [{**{k: endpoint()[k] for k in runner.IDENTITY}, "base": endpoint(),
        "larger": endpoint("d", ["u"], 8), "added_unit_id": "u", "added_support_task_id": "t"}]
    edges = runner.scored_edges(manifest, {"c": 0, "d": 1}, result)
    assert len(edges) == 1 and edges[0].support_label == "no"
    assert edges[0].superset.answer_f1 - edges[0].subset.answer_f1 == 1
    with pytest.raises(ValueError, match="missing"):
        runner.scored_edges(manifest, {"c": 0}, result)


def test_missing_or_invalid_judgment_does_not_drop_edge():
    with pytest.raises(ValueError, match="missing"):
        runner.project_judgments({"jev": {}}, {}, {"t"})
    with pytest.raises(ValueError, match="invalid"):
        runner.project_judgments({"jev": {"t": "no"}}, {"t": {"yes": True}}, {"t"})


def test_duplicate_json_fields_and_network_fail_closed():
    with pytest.raises(ValueError, match="duplicate"):
        runner.parse('{"cache_key":"c","cache_key":"d"}')
    with pytest.raises(RuntimeError, match="network disabled"):
        runner.blocked_socket()


def test_output_cannot_escape_artifacts():
    with pytest.raises(ValueError, match="inside"):
        runner.artifact(runner.ARTIFACTS / ".." / "outside")
