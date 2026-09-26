"""Synthetic CPU-only correctness and complete-output tamper tests."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/research"))
import run_qasper_lexical_baseline as lexical
from run_qasper_evidence_baselines import Unit, PackCounter


class Tokenizer:
    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens and not truncation
        return [1, *text.encode(), 2]


def annotation(evidence):
    return {"native_answer": {"unanswerable": False, "extractive_spans": [],
        "free_form_answer": "answer", "yes_no": None, "evidence": evidence}}


def fixture_source():
    documents = {doc: [Unit(f"u{i}", i, "paragraph", i*10, i*10+5, f"{doc} text{i}", f"{doc} text{i}")
                      for i in range(5)] for doc in ("a", "b")}
    units, keys, positions = lexical.bridge.globalize_documents(documents)
    ids = [f"u{i}" for i in range(5)]
    q = {"family_id":"f", "doc_id":"a", "question_id":"q", "query":"a text1", "seed_ids":ids,
         "candidate_ids":ids, "ranked_ids":ids}
    return {"prepared":{"queries":[q]}, "documents":documents, "units":units, "keys":keys,
        "positions":positions, "tokenizer":Tokenizer(), "input_sha256":{},
        "qa_by_key":{("a","q"):{"answer_annotations":[annotation(["a text1"])]}},
        "query_positions":{("a","q"):0}, "rankings":{("a","q"):ids},
        "candidate_vectors":torch.ones((10,2)), "query_vectors":torch.ones((1,2))}


def test_unicode_casefold_word_tokenization_without_stemming_or_stopword_removal():
    assert lexical.lexical_tokens("Straße ALPHA_2 βeta, don't studies-study") == [
        "strasse", "alpha_2", "βeta", "don", "t", "studies", "study"]
    assert lexical.lexical_tokens("?! —") == []


def test_formula_matches_manual_global_idf_tf_and_average_length():
    index = lexical.build_index(["a a b", "a c", "b"])
    assert index["N"] == 3 and index["avgdl"] == 2
    assert index["df"] == {"a":2,"b":2,"c":1}
    score = lexical.bm25_scores("a", index)
    idf = math.log(1+(3-2+.5)/(2+.5))
    assert score == pytest.approx([idf*2*2.5/(2+1.5*(.25+.75*3/2)), idf, 0.])
    assert lexical.bm25_scores("a a a", index) == score
    assert lexical.bm25_scores("c a", index) == lexical.bm25_scores("a c", index)


def test_empty_and_all_zero_corpus_keep_every_unit_and_replay_ties():
    index = lexical.build_index(["", "!?", " "])
    assert index["zero_lexical_units"] == 3 and index["avgdl"] == 0
    scores = lexical.bm25_scores("anything", index)
    assert scores == [0.,0.,0.] and lexical.bridge.rank_scores(scores) == [0,1,2]
    with pytest.raises(ValueError):
        lexical.build_index([])


def test_global_scores_are_only_filtered_given_document_not_reestimated(monkeypatch):
    source = fixture_source(); q=source["prepared"]["queries"][0]
    index = lexical.build_index([u.text for u in source["units"]])
    scores = lexical.bm25_scores(q["query"],index)
    local = lexical.bm25_scores(q["query"],lexical.build_index([u.text for u in source["documents"]["a"]]))
    assert scores[:5] != local
    monkeypatch.setattr(lexical,"bm25_scores",lambda *args:pytest.fail("must use already frozen global score array"))
    rows=lexical.evaluate_query(q,source,scores,[1.]*10,index)
    row=next(r for r in rows if r["scope"]=="given_document" and r["method"]=="bm25")
    assert row["candidate_scores"] == [scores[i] for i in row["candidate_global_indices"]]
    assert all(i<5 for i in row["candidate_global_indices"])


def test_empty_all_oov_and_in_vocabulary_but_wrong_scope_are_distinguished():
    source=fixture_source(); original=source["prepared"]["queries"][0]
    index=lexical.build_index([u.text for u in source["units"]])
    for query,empty,oov,given_zero in [("?!",1,0,1),("unseenword",0,1,1),("b",0,0,1)]:
        q={**original,"query":query}
        scores=lexical.bm25_scores(query,index)
        rows=lexical.evaluate_query(q,source,scores,[1.]*10,index)
        given=next(r for r in rows if r["scope"]=="given_document" and r["method"]=="bm25")
        corpus=next(r for r in rows if r["scope"]=="corpus_32" and r["method"]=="bm25")
        assert (given["query_empty_lexical"],given["query_nonempty_all_oov"])==(empty,oov)
        assert given["scope_no_positive_lexical_score"]==given_zero
        assert given["seed_zero_lexical_score_units"]==5
        if query=="b":
            assert corpus["scope_positive_lexical_units"]==5 and corpus["query_oov_unique_fraction"]==0
        else:
            assert given["selected_global_indices"]==[0,1,2]


def test_same_neighbor_and_packing_contract_and_gold_cannot_change_selection():
    source=fixture_source(); q=source["prepared"]["queries"][0]
    source["units"][0]=replace(source["units"][0],text="x"*2000,native_text="x"*2000)
    index=lexical.build_index([u.text for u in source["units"]])
    scores=[10-i for i in range(10)]
    rows=lexical.evaluate_query(q,source,scores,scores,index)
    assert len(rows)==4
    for scope in lexical.SCOPES:
        dense,bm25=[r for r in rows if r["scope"]==scope]
        assert dense["seed_global_indices"]==bm25["seed_global_indices"]
        assert dense["candidate_global_indices"]==bm25["candidate_global_indices"]
        assert dense["selected_global_indices"]==bm25["selected_global_indices"]
    assert all(r["actual_evidence_tokens"]<=1024 and r["selected_units"]<=3 for r in rows)
    assert all(0 not in r["selected_global_indices"] for r in rows)
    source["qa_by_key"][("a","q")]["answer_annotations"]=[annotation(["different gold"])]
    changed=lexical.evaluate_query(q,source,scores,scores,index)
    assert [r["selected_global_indices"] for r in changed]==[r["selected_global_indices"] for r in rows]


def test_neighbors_do_not_cross_document_and_zero_ties_are_global():
    source=fixture_source();q=source["prepared"]["queries"][0]
    index=lexical.build_index([u.text for u in source["units"]])
    rows=lexical.evaluate_query(q,source,[0.]*10,[0.]*10,index)
    for r in rows:
        if r["scope"]=="given_document":
            assert r["candidate_global_indices"]==list(range(5))
        else:
            assert r["seed_global_indices"]==list(range(8))
            assert r["selected_global_indices"]==[0,1,2]
    seeds,candidates=lexical.bridge.expand_corpus_candidates([4],source["keys"],source["positions"],seed_count=1,cap=3)
    assert seeds==[4] and candidates==[3,4]


def test_same_text_different_sources_preserved_and_wrong_source_is_not_a_hit():
    source=fixture_source()
    source["units"][0]=replace(source["units"][0],text="same",native_text="same")
    source["units"][5]=replace(source["units"][5],text="same",native_text="same")
    count=PackCounter(source["tokenizer"],source["units"])
    assert lexical.bridge.pack_source_qualified(source["units"],source["keys"],[5,0],1024,count)==[0,5]
    assert lexical.bridge.source_qualified_metrics([("b","same")],"a",[annotation(["same"])])["evidence_f1"]==0


def test_full_denominator_all_negative_comparisons_and_cluster_weights():
    source=fixture_source(); q=source["prepared"]["queries"][0]
    index=lexical.build_index([u.text for u in source["units"]])
    base=lexical.evaluate_query(q,source,[1.]*10,[1.]*10,index)
    rows,queries=[],[]
    for i,family in enumerate(("f1","f1","f2")):
        queries.append({**q,"family_id":family,"question_id":f"q{i}"})
        for row in base:
            new={**row,"family_id":family,"question_id":f"q{i}"}
            new["source_qualified_evidence_f1"]=(.1 if i<2 else .5) if row["method"]=="bm25" else .8
            rows.append(new)
    first=lexical.summarize(rows,queries)
    assert first==lexical.summarize(rows,queries)
    for scope in first["scopes"]:
        delta=scope["comparison"]["metrics"]["source_qualified_evidence_f1"]
        assert delta["negative_questions"]==3 and delta["question_weighted"]<delta["family_balanced"]<0
        assert delta["question_weighted_percentile95"][1]<0
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError,match="denominator"):
            lexical.summarize(bad,queries)


def prepared(tmp_path,monkeypatch):
    source=fixture_source();bound=tmp_path/"bound";bound.write_text("source")
    source["input_sha256"]={str(bound.resolve()):lexical.digest(bound)}
    monkeypatch.setattr(lexical,"load_inputs",lambda *a,**k:source)
    monkeypatch.setattr(lexical,"environment",lambda:{"synthetic":True})
    output=tmp_path/"plan"
    result=lexical.prepare(SimpleNamespace(prepared=tmp_path/"prepared",dense=tmp_path/"dense",output=output))
    return source,output,result


def test_prepare_only_source_index_no_score_gold_or_gpu(tmp_path,monkeypatch):
    monkeypatch.setattr(lexical,"bm25_scores",lambda *a:pytest.fail("cannot score during prepare"))
    source,path,result=prepared(tmp_path,monkeypatch)
    assert result["status"]=="prepared_before_lexical_scoring" and result["gpu_used"] is False
    plan,index,_=lexical.load_plan(path)
    assert lexical.reloaded(plan,index) is source
    assert "answer_annotations" not in json.dumps(plan)
    with pytest.raises(FileExistsError):
        lexical.prepare(SimpleNamespace(output=path))


def test_explicit_dependencies_are_independent_of_caller_imports(monkeypatch):
    before=lexical.code_hashes()
    monkeypatch.setitem(sys.modules,"new_audit_caller",SimpleNamespace(__file__=__file__))
    assert lexical.code_hashes()==before


def reseal_plan(path,plan=None,index=None):
    for name,value in (("plan.json",plan),("lexical_index.json",index)):
        if value is not None:
            (path/name).write_text(json.dumps(value),encoding="utf-8")
    (path/"plan_seal.json").write_text(json.dumps({name:lexical.digest(path/name) for name in lexical.PLAN_FILES[:-1]}))


def test_index_changed_and_resealed_cannot_replace_source_statistics(tmp_path,monkeypatch):
    _,path,_=prepared(tmp_path,monkeypatch)
    plan,index,_=lexical.load_plan(path)
    index["df"]["a"]-=1;plan["index_sha256"]=lexical.pilot.stable_hash(index)
    reseal_plan(path,plan,index)
    plan,index,_=lexical.load_plan(path)
    with pytest.raises(ValueError,match="global lexical index"):
        lexical.reloaded(plan,index)


@pytest.mark.parametrize("field,value",[("config",{}),("gpu_used",True),("status","completed"),("limits",[])])
def test_fixed_contract_resealed_tamper_refused(tmp_path,monkeypatch,field,value):
    _,path,_=prepared(tmp_path,monkeypatch)
    plan=json.loads((path/"plan.json").read_text());plan[field]=value
    reseal_plan(path,plan)
    with pytest.raises(ValueError,match="contract"):
        lexical.load_plan(path)


def test_source_change_after_freeze_refused_before_output(tmp_path,monkeypatch):
    _,path,_=prepared(tmp_path,monkeypatch)
    (tmp_path/"bound").write_text("changed")
    with pytest.raises(ValueError):
        lexical.run(SimpleNamespace(plan=path,output=tmp_path/"run"))
    assert not (tmp_path/"run").exists()


def completed(tmp_path,monkeypatch):
    source,path,_=prepared(tmp_path,monkeypatch)
    out=tmp_path/"run"
    assert lexical.run(SimpleNamespace(plan=path,output=out))["records"]==4
    assert lexical.audit(SimpleNamespace(plan=path,run=out))["status"]=="verified"
    return path,out


def test_full_cpu_synthetic_roundtrip_and_no_overwrite(tmp_path,monkeypatch):
    path,out=completed(tmp_path,monkeypatch)
    with pytest.raises(FileExistsError):
        lexical.run(SimpleNamespace(plan=path,output=out))
    (out/"extra").write_text("unexpected")
    with pytest.raises(ValueError,match="inventory"):
        lexical.audit(SimpleNamespace(plan=path,run=out))


@pytest.mark.parametrize("target",["records","aggregate","metadata","timing"])
def test_output_tamper_and_reseal_cannot_hide_incomplete_or_false_results(tmp_path,monkeypatch,target):
    path,out=completed(tmp_path,monkeypatch)
    report=json.loads((out/"summary.json").read_text())
    public=report["public"]
    if target=="records":
        rows=(out/"per_question.jsonl").read_text().splitlines();rows.pop()
        (out/"per_question.jsonl").write_text("\n".join(rows)+"\n")
    elif target=="aggregate":
        public["scopes"][0]["comparison"]["metrics"]["source_qualified_evidence_f1"]["question_weighted"]=.123
    elif target=="metadata":
        public["corpus_units"]=999
    else:
        public["execution"]["wall_seconds_before_aggregate"]=-1
    (out/"public_aggregate.json").write_text(json.dumps(public),encoding="utf-8")
    report["output_sha256"]={name:lexical.digest(out/name) for name in lexical.RUN_FILES[:-1]}
    (out/"summary.json").write_text(json.dumps(report),encoding="utf-8")
    with pytest.raises(ValueError):
        lexical.audit(SimpleNamespace(plan=path,run=out))
