"""Synthetic-only checks for the restricted confirmation export.

No real archive, cohort, question, reference, model, or network is opened. The
fixtures deliberately include material which must remain outside question and
public projections. These checks do not certify semantic answer correctness.
"""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs/research"))
SPEC = importlib.util.spec_from_file_location(
    "confirmation_export_synthetic", REPO / "docs/research/export_qasper_confirmation.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

ARCHIVE_SHA = "a" * 64
ANSWER_FIELDS = {"unanswerable", "extractive_spans", "free_form_answer", "yes_no", "evidence"}
QUESTION_FIELDS = {"doc_id", "source_id", "original_family_id", "family_id", "question_id", "question"}
IDENTITY = ("family_id", "doc_id", "question_id")
PRIVATE_MARKERS = ("PRIVATE_DOC", "PRIVATE_SOURCE", "PRIVATE_FAMILY", "PRIVATE_COMPONENT",
                   "QUESTION_TEXT_SENTINEL", "REFERENCE_TEXT_SENTINEL", "WORKER_SECRET")


def answer(**changes):
    return {"unanswerable": False, "extractive_spans": [],
            "free_form_answer": "REFERENCE_TEXT_SENTINEL", "yes_no": None,
            "evidence": ["Paragraph one.", "Paragraph one."], **changes}


def make_fixture(counts=(3, 1, 0)):
    cohort, native, canonical = [], {}, {}
    for di, question_count in enumerate(counts):
        doc, source = f"PRIVATE_DOC_{di}", f"PRIVATE_SOURCE_{di}"
        family, component = f"PRIVATE_FAMILY_{di}", f"PRIVATE_COMPONENT_{di // 2}"
        paper = {"title": "  Native title\t", "abstract": "Cafe\u0301 abstract.\n",
                 "full_text": [{"section_name": " Section 1 ",
                                "paragraphs": ["Paragraph one.", "Paragraph one.", " \n\t"]}],
                 "figures_and_tables": [{"file": "figure.png", "caption": "Figure caption."}], "qas": []}
        for qi in range(question_count):
            paper["qas"].append({"question_id": f"q{qi}",
                                 "question": f"  QUESTION_TEXT_SENTINEL {qi}?\n",
                                 "question_writer": "WORKER_SECRET",
                                 "answers": [{"annotation_id": f"a{qi}", "worker_id": "WORKER_SECRET",
                                              "answer": answer()}]})
        parts, blocks = [], []

        def add(kind, pointer, text):
            if not text.strip():
                return
            # Canonical and native are intentionally different whitespace
            # representations. Both must be retained, never conflated.
            canonical_text = text.strip()
            prefix = "\n\n" if parts else ""
            start = sum(map(len, parts)) + len(prefix)
            parts.extend([prefix, canonical_text])
            blocks.append({"block_id": f"block-{len(blocks)}", "kind": kind,
                           "source_locator": {"json_pointer": pointer},
                           "char_span": [start, start + len(canonical_text)]})

        add("heading", f"/{source}/title", paper["title"])
        add("heading", f"/{source}/abstract", "Abstract")
        add("abstract", f"/{source}/abstract", paper["abstract"])
        for si, section in enumerate(paper["full_text"]):
            add("heading", f"/{source}/full_text/{si}", section["section_name"])
            for pi, text in enumerate(section["paragraphs"]):
                add("paragraph", f"/{source}/full_text/{si}/paragraphs/{pi}", text)
        body = "".join(parts)
        body_sha = hashlib.sha256(body.encode()).hexdigest()
        cohort.append({"source": "qasper", "split": "validation", "doc_id": doc,
                       "source_id": source, "family_id": family, "component_key": component,
                       "normalized_body_sha256": body_sha,
                       "status": "operational_candidate_proposed", "development_exposed": False})
        native[source] = paper
        canonical[doc] = {"source": "qasper", "doc_id": doc, "source_id": source,
                          "family_id": family, "original_split": "validation",
                          "normalized_body_sha256": body_sha, "canonical_text": body,
                          "blocks": blocks,
                          "raw_locator": {"member": "qasper-dev-v0.3.json", "json_pointer": f"/{source}"},
                          "extra": {"raw_archive_sha256": ARCHIVE_SHA}}
    return cohort, native, canonical


def project(data=None, sampled=None):
    data = data or make_fixture()
    cohort, native, canonical = data
    if sampled is None:
        sampled = [row["doc_id"] for row in cohort]
    return m.project_dataset(cohort, native, canonical, ARCHIVE_SHA, sampled)


def lookup(rows):
    return {tuple(row[k] for k in IDENTITY): row for row in rows}


def test_null_section_heading_keeps_paragraphs_and_all_questions():
    data=make_fixture((3,));cohort,native,canonical=data
    source=cohort[0]['source_id'];doc=cohort[0]['doc_id']
    native[source]['full_text'][0]['section_name']=None
    canonical[doc]['blocks']=[b for b in canonical[doc]['blocks']
        if b['source_locator']['json_pointer']!=f'/{source}/full_text/0']
    result=project(data)
    assert len(result['questions'])==3
    assert sum(u['kind']=='paragraph' for u in result['documents'][0]['units'])==2
    assert all(u['native_text'] is not None for u in result['documents'][0]['units'])


def test_null_section_heading_with_canonical_heading_is_rejected():
    data=make_fixture((1,));source=data[0][0]['source_id']
    data[1][source]['full_text'][0]['section_name']=None
    with pytest.raises(ValueError,match='null native heading'):
        project(data)


@pytest.mark.parametrize('field',['title','abstract'])
def test_null_non_heading_native_field_is_rejected(field):
    data=make_fixture((1,));source=data[0][0]['source_id']
    data[1][source][field]=None
    with pytest.raises(ValueError,match='native unit is not text'):
        project(data)


@pytest.mark.parametrize('value',[0,False,[],{}])
def test_nonnull_nonstring_section_heading_is_rejected(value):
    data=make_fixture((1,));source=data[0][0]['source_id']
    data[1][source]['full_text'][0]['section_name']=value
    with pytest.raises(ValueError,match='native unit is not text'):
        project(data)


def test_sample_exact_fixed_hash_membership_and_input_order_independence():
    values = [f"document-{i:02d}" for i in range(20)]
    salt = "SLAC-QASPER-CONFIRMATION-EXPORT-20260927-v1"
    expected = sorted(values, key=lambda value: (
        hashlib.sha256(f"{salt}|documents|{value}".encode()).hexdigest(), value))[:8]
    actual = m.sample_ids(values, 8, "documents")
    assert set(actual) == set(expected)
    assert len(actual) == 8
    assert actual == m.sample_ids(list(reversed(values)), 8, "documents")


@pytest.mark.parametrize("size,limit", [(0, 2), (1, 2), (2, 2), (3, 2), (20, 8)])
def test_sample_count_has_no_replacement_or_top_up(size, limit):
    values = [str(i) for i in range(size)]
    actual = m.sample_ids(values, limit, "questions:synthetic")
    assert len(actual) == len(set(actual)) == min(size, limit)
    assert set(actual) <= set(values)


def test_complete_projection_and_reference_question_join():
    data = make_fixture()
    result = project(data)
    assert set(result) == {"documents", "questions", "question_manifest", "references",
                           "document_inventory", "spot_check", "public_aggregate"}
    assert len(result["documents"]) == len(result["document_inventory"]) == 3
    assert len(result["questions"]) == len(result["references"]) == len(result["question_manifest"]) == 4
    assert set(lookup(result["questions"])) == set(lookup(result["references"])) == set(lookup(result["question_manifest"]))
    assert all(set(row) == QUESTION_FIELDS for row in result["questions"])
    assert all(set(row) == set(IDENTITY) for row in result["question_manifest"])
    assert all("question" not in row for row in result["references"])
    assert {r["doc_id"] for r in result["document_inventory"]} == {r["doc_id"] for r in data[0]}


def test_original_family_retained_and_component_used_for_statistics():
    cohort, native, canonical = make_fixture()
    result = project((cohort, native, canonical))
    original = {r["doc_id"]: r for r in cohort}
    for rows in (result["questions"], result["references"], result["documents"]):
        for row in rows:
            assert row["family_id"] == original[row["doc_id"]]["component_key"]
            assert row["original_family_id"] == original[row["doc_id"]]["family_id"]
    assert len({r["family_id"] for r in result["questions"]}) == 1


def test_all_annotations_and_exact_values_retained_no_worker_fields():
    data = make_fixture((1,))
    annotations = [answer(unanswerable=True, free_form_answer="", evidence=[]),
                   answer(yes_no=False, free_form_answer=""),
                   answer(extractive_spans=[" span A ", "span B"], free_form_answer=""),
                   answer(evidence=["FLOAT SELECTED: Figure caption.", "unmatched", "", "unmatched"],
                          highlighted_evidence=[" optional highlighted \n"]) ]
    data[1][data[0][0]["source_id"]]["qas"][0]["answers"] = [
        {"annotation_id": str(i), "worker_id": "WORKER_SECRET", "answer": a}
        for i, a in enumerate(annotations)]
    result = project(data)
    exported = result["references"][0]["answer_annotations"]
    assert [row["native_answer"] for row in exported] == annotations
    assert all(set(row) == {"native_answer"} for row in exported)
    assert "WORKER_SECRET" not in json.dumps(result)


def test_question_and_native_text_are_not_stripped_or_normalized():
    data = make_fixture((1,))
    result = project(data)
    paper = next(iter(data[1].values()))
    assert result["questions"][0]["question"] == paper["qas"][0]["question"]
    units = result["documents"][0]["units"]
    assert any(u["native_text"] == paper["title"] for u in units)
    assert any(u["native_text"] == paper["abstract"] for u in units)
    canonical = next(iter(data[2].values()))["canonical_text"]
    assert all(canonical[u["start"]:u["end"]] == u["text"] for u in units)
    assert sum(u["native_text"] == "Paragraph one." for u in units) == 2
    assert len({u["unit_id"] for u in units}) == len(units)


def test_changing_answers_cannot_select_questions_or_sample():
    data = make_fixture((4, 3, 0))
    before = project(data)
    changed = deepcopy(data)
    for paper in changed[1].values():
        for qa in paper["qas"]:
            qa["answers"] = [{"answer": answer(unanswerable=True, free_form_answer="", evidence=[])}]
    after = project(changed)
    for key in ("questions", "question_manifest", "documents"):
        assert before[key] == after[key]
    sample_keys = lambda result: [(r["doc_id"], r["question_id"])
                                 for r in result["spot_check"]["sample_questions"]]
    assert sample_keys(before) == sample_keys(after)


def test_public_projection_contains_no_identity_text_or_worker_markers():
    result = project()
    public = json.dumps(result["public_aggregate"], ensure_ascii=False)
    assert all(marker not in public for marker in PRIVATE_MARKERS)
    assert "file:" not in public and "D:/" not in public and "C:/" not in public


def test_same_question_id_across_documents_is_valid():
    result = project(make_fixture((1, 1)))
    assert [r["question_id"] for r in result["questions"]] == ["q0", "q0"]
    assert len(lookup(result["questions"])) == 2


def test_duplicate_question_in_same_document_is_rejected():
    data = make_fixture((1,))
    paper = next(iter(data[1].values()))
    paper["qas"].append(deepcopy(paper["qas"][0]))
    with pytest.raises((ValueError, TypeError)):
        project(data)


def test_empty_question_is_retained_not_dropped():
    data = make_fixture((2,))
    qas = next(iter(data[1].values()))["qas"]
    qas[0]["question"], qas[1]["question"] = "", " \n\t"
    result = project(data)
    assert [r["question"] for r in result["questions"]] == ["", " \n\t"]
    assert len(result["references"]) == 2
    assert result["public_aggregate"]["invalid_queries"] == 2
    assert result["public_aggregate"]["ready_for_candidate_preparation"] is False


def test_exact_sample_scope_not_full_reference_or_text_audit(monkeypatch):
    data = make_fixture((5,) * 12)
    sample = m.sample_ids([r["doc_id"] for r in data[0]], 8, "documents")
    original_normalize, original_convert = m.normalize_evidence, m.references_from_annotations
    normalized, converted = [], []

    def normalize(value):
        normalized.append(value)
        return original_normalize(value)

    def convert(value):
        converted.append(deepcopy(value))
        return original_convert(value)

    monkeypatch.setattr(m, "normalize_evidence", normalize)
    monkeypatch.setattr(m, "references_from_annotations", convert)
    result = project(data, sample)
    assert len(result["questions"]) == len(result["references"]) == 60
    assert len(result["documents"]) == 12
    assert len(converted) == 16
    sampled_units = sum(len(r["units"]) for r in result["documents"] if r["doc_id"] in sample)
    assert len(normalized) == 2 * sampled_units
    expected = sorted((doc, qid) for doc in sample
                      for qid in m.sample_ids([f"q{i}" for i in range(5)], 2, f"question:{doc}"))
    actual = sorted((r["doc_id"], r["question_id"]) for r in result["spot_check"]["sample_questions"])
    assert actual == expected
    assert result["public_aggregate"]["full_reference_schema_audit_performed"] is False


def test_sampled_zero_question_document_not_replaced_or_topped_up():
    result = project(make_fixture((0, 1, 5)))
    public = result["public_aggregate"]
    assert public["documents"] == public["sample_documents"] == 3
    assert public["documents_without_questions"] == 1
    assert public["sample_questions"] == 3
    assert public["questions"] == 6


def test_unsampled_reference_is_preserved_without_calling_converter():
    data = make_fixture((1, 1))
    second = data[1][data[0][1]["source_id"]]["qas"][0]
    second["answers"] = []
    result = project(data, [data[0][0]["doc_id"]])
    assert len(result["questions"]) == 2
    assert result["references"][1]["answer_annotations"] == []
    assert result["public_aggregate"]["questions_without_annotations"] == 1
    assert result["public_aggregate"]["ready_for_candidate_preparation"] is False


def test_sampled_empty_annotation_list_stops_instead_of_dropping_question():
    data = make_fixture((1,))
    next(iter(data[1].values()))["qas"][0]["answers"] = []
    with pytest.raises(ValueError):
        project(data)


def test_omitted_fields_reported_privately_without_mutating_source():
    data = make_fixture((1,))
    original_answer = next(iter(data[1].values()))["qas"][0]["answers"][0]["answer"]
    original_answer["PRIVATE_CUSTOM_FIELD"] = "WORKER_SECRET"
    before = deepcopy(data)
    result = project(data)
    assert data == before
    assert result["spot_check"]["omitted_non_evaluator_answer_fields"] == {"PRIVATE_CUSTOM_FIELD": 1}
    assert result["public_aggregate"]["omitted_non_evaluator_answer_field_occurrences"] == 1
    assert "PRIVATE_CUSTOM_FIELD" not in json.dumps(result["public_aggregate"])
    assert "PRIVATE_CUSTOM_FIELD" not in json.dumps(result["references"])


def test_exposure_and_admission_flags_do_not_claim_unseen_or_independent():
    public = project()["public_aggregate"]
    assert public["new_qa_machine_read"] is True
    assert public["machine_decoded_validation_member_documents"] == 3
    assert public["api_calls"] == 0
    assert public["all_questions_retained"] is True
    for key in ("raw_text_or_qa_shown_to_agent", "new_model_inputs_submitted", "quality_scores_computed",
                "training_started", "official_test_payload_read", "paid_execution_admitted",
                "independence_established", "model_contamination_free", "human_reviewed"):
        assert public[key] is False


@pytest.mark.parametrize("sample", [["outside"], ["PRIVATE_DOC_0", "PRIVATE_DOC_0"]])
def test_invalid_sample_identity_is_rejected(sample):
    with pytest.raises(ValueError):
        project(make_fixture((1,)), sample)


@pytest.mark.parametrize("values,limit", [(["same", "same"], 1), ([""], 1), ([1], 1),
                                         (["ok"], -1), (["ok"], True), (["ok"], 1.0)])
def test_invalid_sampling_contract_is_rejected(values, limit):
    with pytest.raises(ValueError):
        m.sample_ids(values, limit, "documents")


@pytest.mark.parametrize("field,value", [("source_id", "other"), ("family_id", "other"),
                                        ("original_split", "train")])
def test_canonical_identity_or_split_drift_is_rejected(field, value):
    data = make_fixture((1,))
    next(iter(data[2].values()))[field] = value
    with pytest.raises((ValueError, TypeError)):
        project(data)


def test_archive_lineage_drift_is_rejected():
    data = make_fixture((1,))
    next(iter(data[2].values()))["extra"]["raw_archive_sha256"] = "b" * 64
    with pytest.raises((ValueError, TypeError)):
        project(data)


def test_sampled_text_mismatch_is_rejected():
    data = make_fixture((1,))
    row = next(iter(data[2].values()))
    row["canonical_text"] = row["canonical_text"].replace("Native title", "Wrong title!")
    with pytest.raises((ValueError, TypeError)):
        project(data)


def write_archive(tmp_path, entries):
    path = tmp_path / "qasper-train-dev-v0.3.tgz"
    with tarfile.open(path, "w:gz") as stream:
        for name, content, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(content) if kind == tarfile.REGTYPE else 0
            stream.addfile(member, io.BytesIO(content) if kind == tarfile.REGTYPE else None)
    return path


def test_read_native_only_opens_exact_validation_member(tmp_path, monkeypatch):
    payload = {"synthetic": {"qas": []}}
    path = write_archive(tmp_path, [("qasper-dev-v0.3.json", json.dumps(payload).encode(), tarfile.REGTYPE),
                                    ("qasper-test-v0.3.json", b"TEST MUST NOT PARSE", tarfile.REGTYPE),
                                    ("qasper-train-v0.3.json", b"TRAIN MUST NOT PARSE", tarfile.REGTYPE)])
    original = tarfile.TarFile.extractfile
    seen = []

    def restricted(self, member):
        name = member.name if isinstance(member, tarfile.TarInfo) else member
        assert name == "qasper-dev-v0.3.json"
        seen.append(name)
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", restricted)
    result = m.read_native(path)
    assert seen == ["qasper-dev-v0.3.json"]
    assert payload == (result[0] if isinstance(result, tuple) else result)


@pytest.mark.parametrize("entries", [[],
    [("qasper-dev-v0.3.json", b"{}", tarfile.REGTYPE)] * 2,
    [("qasper-dev-v0.3.json", b"", tarfile.SYMTYPE)],
    [("subdir/qasper-dev-v0.3.json", b"{}", tarfile.REGTYPE)]])
def test_read_native_requires_unique_regular_exact_member(tmp_path, entries):
    with pytest.raises((ValueError, KeyError)):
        m.read_native(write_archive(tmp_path, entries))


def test_duplicate_native_json_keys_are_rejected(tmp_path):
    path = write_archive(tmp_path, [("qasper-dev-v0.3.json", b'{"same":{},"same":{}}', tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="duplicate"):
        m.read_native(path)


def test_oversized_member_is_rejected_before_content_read(tmp_path, monkeypatch):
    path = write_archive(tmp_path, [("qasper-dev-v0.3.json", b"{}", tarfile.REGTYPE)])
    original = tarfile.TarFile.getmembers

    def oversized(self):
        members = original(self)
        members[0].size = 32 * 1024 * 1024 + 1
        return members

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized member content must not be opened")

    monkeypatch.setattr(tarfile.TarFile, "getmembers", oversized)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", forbidden)
    with pytest.raises(ValueError, match="member"):
        m.read_native(path)


class SyntheticExecution:
    """Real I/O against temporary synthetic archives; parent loading is stubbed."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path
        self.art = tmp_path / "artifacts"
        self.art.mkdir()
        self.run = self.art / "export-run"
        self.audit = self.art / "sample-audit"
        self.plan_dir = self.art / "plan"
        self.plan_dir.mkdir()
        self.anchor = self.plan_dir / "synthetic-plan-anchor.json"
        self.anchor.write_text('{"synthetic":true}\n', encoding="utf-8")
        self.parent = tmp_path / "synthetic-parent.json"
        self.parent.write_text('{"synthetic_parent":true}\n', encoding="utf-8")
        self.data = make_fixture((1,) * 248)
        self.cohort, native, canonical = self.data
        # Extra native entries model whole-member decoding, not cohort export.
        native.update({f"outside-cohort-{i}": {"qas": []} for i in range(33)})
        self.archive = write_archive(tmp_path, [(m.MEMBER, json.dumps(native).encode(), tarfile.REGTYPE)])
        archive_sha = m.digest(self.archive)
        for row in canonical.values():
            row["extra"]["raw_archive_sha256"] = archive_sha
        self.shard = tmp_path / "documents-00000.jsonl.gz"
        self.shard.write_bytes(gzip.compress(b"".join(
            json.dumps(row).encode() + b"\n" for row in canonical.values()), mtime=0))
        self.plan = {"schema": m.SCHEMA, "status": "prepared_export_not_executed",
                     "run_output": str(self.run), "input_sha256": {str(self.parent): m.digest(self.parent)},
                     "raw_content_expected_sha256": {
                         "archive": {"path": str(self.archive), "sha256": archive_sha},
                         "canonical": {"path": str(self.shard), "sha256": m.digest(self.shard)}},
                     "sample_doc_ids": m.sample_ids([r["doc_id"] for r in self.cohort], 8, "documents")}
        self.plan_bindings = {str(self.anchor): m.digest(self.anchor)}
        monkeypatch.setattr(m, "ART", self.art)
        monkeypatch.setattr(m, "load_plan", lambda directory: (
            deepcopy(self.plan), deepcopy(self.cohort), dict(self.plan_bindings)))

    def execute(self):
        return m.run(self.plan_dir)

    def reseal_file(self, name, value):
        path = self.run / name
        if name.endswith(".jsonl"):
            path.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in value))
        else:
            path.write_text(json.dumps(value), encoding="utf-8")
        summary = json.loads((self.run / "summary.json").read_bytes())
        summary["output_sha256"][name] = m.digest(path)
        (self.run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture
def execution(tmp_path, monkeypatch):
    return SyntheticExecution(tmp_path, monkeypatch)


def test_normal_run_and_sample_audit_full_248_without_reopening_raw(execution, monkeypatch):
    public = execution.execute()
    assert public["documents"] == public["questions"] == 248
    assert public["machine_decoded_validation_member_documents"] == 281
    assert public["sample_documents"] == public["sample_questions"] == 8
    original_open = Path.open
    forbidden_paths = {execution.archive.resolve(), execution.shard.resolve()}

    def guarded_open(path, *args, **kwargs):
        assert path.resolve() not in forbidden_paths, "sample audit must not reopen raw sources"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    receipt = m.audit_sample(execution.plan_dir, execution.audit)
    assert receipt["status"] == "verified_complete_coverage_and_fixed_sample"
    assert receipt["documents"] == receipt["questions"] == 248
    assert receipt["full_content_audit_performed"] is False
    assert receipt["raw_archive_reopened_for_audit"] is False


def test_duplicate_inventory_rejected_even_with_updated_output_hash(execution):
    execution.execute()
    inventory = m.json_rows(execution.run / "document_inventory.jsonl")
    inventory[1] = deepcopy(inventory[0])
    execution.reseal_file("document_inventory.jsonl", inventory)
    with pytest.raises(ValueError, match="inventory identities"):
        m.audit_sample(execution.plan_dir, execution.audit)
    assert not execution.audit.exists()


def test_wrong_per_document_count_rejected_even_with_updated_hash(execution):
    execution.execute()
    inventory = m.json_rows(execution.run / "document_inventory.jsonl")
    inventory[0]["question_count"] = 987654
    execution.reseal_file("document_inventory.jsonl", inventory)
    with pytest.raises(ValueError, match="mechanical counts"):
        m.audit_sample(execution.plan_dir, execution.audit)


@pytest.mark.parametrize("field,value", [("invalid_queries", 987654),
                                        ("ready_for_candidate_preparation", False),
                                        ("reference_annotations", 987654),
                                        ("independence_established", True),
                                        ("paid_execution_admitted", True)])
def test_public_counts_and_flags_recomputed_not_merely_resealed(execution, field, value):
    execution.execute()
    public = json.loads((execution.run / "public_aggregate.json").read_bytes())
    public[field] = value
    execution.reseal_file("public_aggregate.json", public)
    with pytest.raises(ValueError, match="public mechanical counts or flags"):
        m.audit_sample(execution.plan_dir, execution.audit)


def test_output_hash_inventory_must_be_complete(execution):
    execution.execute()
    path = execution.run / "summary.json"
    summary = json.loads(path.read_bytes())
    del summary["output_sha256"]["document_inventory.jsonl"]
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="seal inventory"):
        m.audit_sample(execution.plan_dir, execution.audit)


@pytest.mark.parametrize("target", ["output", "parent", "plan"])
def test_change_during_audit_is_rejected_before_receipt(execution, monkeypatch, target):
    execution.execute()
    original_load = m.load
    victim = {"output": execution.run / "documents.jsonl", "parent": execution.parent,
              "plan": execution.anchor}[target]
    changed = False

    def mutate_after_parse(path):
        nonlocal changed
        value = original_load(path)
        if Path(path).name == "public_aggregate.json":
            victim.write_bytes(victim.read_bytes() + b"\n")
            changed = True
        return value

    monkeypatch.setattr(m, "load", mutate_after_parse)
    with pytest.raises(ValueError, match="bound input changed"):
        m.audit_sample(execution.plan_dir, execution.audit)
    assert changed and not execution.audit.exists()


def test_existing_run_directory_cannot_be_reused(execution):
    execution.execute()
    before = {p.name: m.digest(p) for p in execution.run.iterdir()}
    with pytest.raises(ValueError, match="new and private"):
        execution.execute()
    assert before == {p.name: m.digest(p) for p in execution.run.iterdir()}


def test_failed_source_read_keeps_registration_and_cannot_restart(execution):
    execution.plan["raw_content_expected_sha256"]["archive"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw content commitment"):
        execution.execute()
    assert {p.name for p in execution.run.iterdir()} == {"registration.json"}
    with pytest.raises(ValueError, match="new and private"):
        execution.execute()


def test_existing_audit_output_cannot_be_reused(execution):
    execution.execute()
    m.audit_sample(execution.plan_dir, execution.audit)
    before = m.digest(execution.audit / "verification.json")
    with pytest.raises(ValueError, match="new private audit"):
        m.audit_sample(execution.plan_dir, execution.audit)
    assert m.digest(execution.audit / "verification.json") == before


def test_prepare_selects_metadata_sample_without_opening_deferred_sources(tmp_path, monkeypatch):
    root, art = tmp_path / "repo", tmp_path / "repo/artifacts/research-foundation"
    art.mkdir(parents=True)
    cohort, _, _ = make_fixture((1,) * 248)
    proposed = art / "qasper-confirmation-metadata-run-01/proposed_cohort.jsonl"
    proposed.parent.mkdir()
    proposed.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in cohort))
    summary_path = proposed.parent / "summary.json"
    summary_path.write_text(json.dumps({"status": "completed_metadata_proposal",
                                       "output_sha256": {proposed.name: m.digest(proposed)}}), encoding="utf-8")
    archive, shard = tmp_path / "qasper-train-dev-v0.3.tgz", tmp_path / "documents-00000.jsonl.gz"
    archive.write_bytes(b"DO NOT READ RAW QA")
    shard.write_bytes(b"DO NOT READ RAW PROSE")
    alignment = art / "qasper-pool/native_qa_alignment.json"
    alignment.parent.mkdir()
    alignment.write_text(json.dumps({"input_sha256": {str(archive): "a" * 64, str(shard): "b" * 64}}), encoding="utf-8")
    protocol = root / "docs/research/results/qasper_confirmation_protocol_v2_20260927.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text('{"synthetic_protocol":true}', encoding="utf-8")
    for relative in ("docs/research/CONFIRMATION_EXPORT_PROTOCOL_20260927.md",
                     "tests/research/test_qasper_confirmation_export.py",
                     "docs/research/qasper_alignment_v2.py", "docs/research/qasper_metrics.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture", encoding="utf-8")
    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(m, "ART", art)
    monkeypatch.setattr(m, "PINS", {str(path.relative_to(art)): m.digest(path)
                                   for path in (summary_path, alignment)})
    monkeypatch.setattr(m, "PROTOCOL_SHA", m.digest(protocol))
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.resolve() not in {archive.resolve(), shard.resolve()}, "prepare read deferred raw content"
        return original_open(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepare must not decode native QA")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(m, "read_native", forbidden)
    report = m.prepare(art / "prepared", art / "future-run")
    assert report["documents"] == 248 and report["sample_documents"] == 8
    assert report["new_qa_read"] is False
    assert not (art / "future-run").exists()
