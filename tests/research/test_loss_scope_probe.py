"""Four-request scope probe tests; authored inputs and fake transport only."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import socket
import sys

import pytest

from docs.research import run_loss_scope_probe as runner


@pytest.fixture
def sources(monkeypatch):
    def unit(uid, order, text):
        digest = runner.sha(text.encode())
        return {"unit_id": uid, "source_order": order, "text": text,
                "native_text_sha256": digest, "retrieval_text_sha256": digest}
    cases, contrasts = [], []
    for n in (1,2):
        a,r,c = unit("a",0,"Lamp has a red switch."),unit("r",1,"Lamp uses cell X."),unit("c",2,"Lamp has a blue case.")
        inv = {"doc_id":f"invented{n}","original_pack":[a,r],"proposed_pack":[a,c],"candidate":c,"removed":r}
        cases.append(inv | {"ordinal":n,"query":"Base question?"})
        for branch in ("L","R"):
            contrasts.append({"contrast_id":f"Q{len(contrasts)+1}","base_ordinal":n,
                "evidence_inventory":inv,"evidence_inventory_sha256":runner.sha(runner.canonical(inv)),
                "query":f"What {'cell powers' if branch=='L' else 'switch color distinguishes'} invented lamp {n}?",
                "target_claim_id":f"C{n}-{branch}","authored_expected_loss":"yes" if branch=='L' else "no"})
    contrasts += [{"contrast_id":"Q5","query":"HELD_NEVER_SEND"},{"contrast_id":"Q6","query":"HELD_NEVER_SEND"}]
    packet=runner.canonical({"cases":cases})
    review=runner.canonical({"packet_sha256":runner.sha(packet),"authored_contrasts":contrasts})
    blobs={p:b"invented source" for p in runner.SOURCE_NAMES}
    blobs.update({runner.PACKET:packet,runner.REVIEW:review,runner.INDEPENDENT_REVIEW:runner.canonical({"input_sha256":runner.sha(review)}),runner.SAMPLE_PLAN:b"{}"})
    monkeypatch.setattr(runner,"PINNED",{p:runner.sha(blobs[p]) for p in runner.PINNED})
    monkeypatch.setattr(runner,"_snapshots",lambda:blobs.copy())
    monkeypatch.setattr(runner.common,"_load_counter",lambda _:(len,{"version":"TEST_ONLY_FAKE_COUNTER"}))
    return blobs


def test_four_twelve_and_only_query_differs_within_same_evidence_pairs(sources):
    plan,wires=runner._assemble(sources)
    assert len(wires)==len({w.cache_key for w in wires})==4
    assert sum(len(w.expected_ids) for w in wires)==12
    assert sum(Decimal(w.reserved_usd) for w in wires)==Decimal("0.020")
    for i in (0,2):
        a,b=deepcopy(wires[i].payload()),deepcopy(wires[i+1].payload())
        assert a['state'].pop('query')!=b['state'].pop('query')
        assert a==b
    assert plan['readout_only']['loss_patterns']==[['yes','no'],['yes','no']]
    for wire in wires:
        assert not any(s in wire.payload_bytes for s in (b'HELD_NEVER_SEND',b'authored_expected',b'target_claim',b'contrast_id',b'base_ordinal'))
        assert wire.plan.max_tokens==1024 and wire.plan.max_judge_tokens==2048
    assert runner.client.REQUEST_CAP==6 and runner.client.QUESTION_CAP==18
    assert runner.client.base.QUESTION_CAP==10


def test_supervision_and_held_cases_do_not_change_wire(sources):
    review,packet=json.loads(sources[runner.REVIEW]),json.loads(sources[runner.PACKET])
    first=runner.freeze_requests(review,packet,len)[1]
    for c in review['authored_contrasts'][:4]:
        c['authored_expected_loss']='UNKNOWN_READOUT_ONLY'
        c['target_claim_id']='PRIVATE_LABEL'
        c['reason']='PRIVATE_REASON'
    review['authored_contrasts'][4:] = [{'arbitrary':'HELD_CONTENT_CHANGED'}]
    second=runner.freeze_requests(review,packet,len)[1]
    assert [w.payload_bytes for w in first]==[w.payload_bytes for w in second]


@pytest.mark.parametrize('mutation',['order','base','inventory','source_text','proposed'])
def test_input_identity_or_order_changes_fail(sources,mutation):
    review,packet=json.loads(sources[runner.REVIEW]),json.loads(sources[runner.PACKET])
    if mutation=='order': review['authored_contrasts'][:4]=list(reversed(review['authored_contrasts'][:4]))
    elif mutation=='base': review['authored_contrasts'][0]['base_ordinal']=2
    elif mutation=='inventory': review['authored_contrasts'][0]['evidence_inventory']['removed']['source_order']=7
    elif mutation=='source_text': packet['cases'][0]['candidate']['text']='altered'
    else: packet['cases'][0]['proposed_pack'].reverse()
    with pytest.raises(ValueError):runner.freeze_requests(review,packet,len)


@pytest.mark.parametrize('mutation',['source','plan','private_review','tokenizer'])
def test_drift_rejected_before_client_or_consumption(tmp_path,monkeypatch,sources,mutation):
    folder=tmp_path/'plan';runner.prepare_plan(folder);path=folder/'plan.json'
    if mutation=='source':sources[runner.SOURCE_NAMES[0]]+=b'changed'
    elif mutation=='private_review':sources[runner.REVIEW]+=b' '
    elif mutation=='tokenizer':monkeypatch.setattr(runner.common,'_load_counter',lambda _:(_ for _ in ()).throw(ValueError('drift')))
    else:
        value=json.loads(path.read_bytes());value['requests'][0]['wire_body_sha256']='0'*64
        path.write_bytes(runner.canonical(value)+b'\n')
    monkeypatch.setattr(runner,'ScopeProbeClient',lambda *a,**k:pytest.fail('client forbidden'))
    with pytest.raises(ValueError):runner.execute_plan(path,tmp_path/'run',key_file='never-read',live=True)
    assert not (folder/'execution_claim.json').exists()


@pytest.mark.parametrize('failure',[None,'unknown_fee','timeout','known_overrun'])
def test_inherited_four_request_transport_claim_and_failure_accounting(tmp_path,monkeypatch,sources,failure):
    monkeypatch.setattr(socket,'socket',lambda *a,**k:pytest.fail('network forbidden'))
    monkeypatch.setattr(runner.client.base,'read_key',lambda *a,**k:pytest.fail('key forbidden'))
    folder=tmp_path/'plan';runner.prepare_plan(folder);calls=[]
    def fake(w,timeout):
        calls.append(w.cache_key)
        saved=json.loads((tmp_path/'run'/'ledger.json').read_bytes())
        assert Decimal(saved['reservation_total_usd'])==Decimal('.005')*len(calls)
        assert saved['attempts'][-1]['cost_status']=='cost_unknown' and 0<timeout<=30
        if failure=='timeout':raise TimeoutError('must not save private exception')
        return {'model':runner.client.RESPONSE_MODEL,'provider':'typesafe',
                'usage':{'cost':None if failure=='unknown_fee' else '0.006' if failure=='known_overrun' else '0.0001',
                         'input_tokens':25,'output_tokens':3},
                'answers':{d:{'type':'choice','choice':'unknown'} for d in w.expected_ids}}
    result=runner.execute_plan(folder/'plan.json',tmp_path/'run',transport=fake,key_file='never-read')
    assert len(calls)==(4 if failure is None else 1)
    assert result['status']==('completed' if failure is None else 'halted')
    assert (result['request_cap'],result['question_cap'],result['deadline_seconds'])==(4,12,120)
    assert result['budget_usd']=='0.020' and result['automatic_retries']==0
    if failure=='known_overrun':assert result['attempts'][0]['actual_cost_usd']=='0.006'
    elif failure:assert result['attempts'][0]['cost_status']=='cost_unknown'
    else:assert result['actual_reported_cost_usd']=='0.0004' and result['reservation_total_usd']=='0.020'
    with pytest.raises(FileExistsError):runner.execute_plan(folder/'plan.json',tmp_path/'retry',transport=fake)
    assert not (tmp_path/'retry').exists()


def test_client_rejects_wrong_count_and_reuses_no_mode_and_deadline_guards(tmp_path,sources):
    _,wires=runner._assemble(sources)
    with pytest.raises(ValueError):runner.ScopeProbeClient(tmp_path/'short',wires[:3])
    with pytest.raises(ValueError):runner.ScopeProbeClient(tmp_path/'dupe',wires[:3]+wires[:1])
    no_mode=runner.ScopeProbeClient(tmp_path/'no-mode',wires,key_file='never-read')
    assert no_mode.run()['halt_reason']=='execution_mode_not_explicit'
    ticks=iter((0.,121.,121.))
    probe=runner.ScopeProbeClient(tmp_path/'expired',wires,clock=lambda:next(ticks),transport=lambda *a:pytest.fail('expired'))
    result=probe.run()
    assert result['halt_reason']=='deadline_exceeded' and not result['attempts']


def test_bounded_cli_fixes_worker_and_120_second_supervision(tmp_path,monkeypatch,capsys):
    path=tmp_path/'plan.json';path.write_bytes(b'fake plan')
    monkeypatch.setattr(runner,'_contained',lambda p:Path(p));captured={}
    def supervise(command,output,receipt,**kwargs):
        captured.update(command=command,**kwargs)
        return {'status':'worker_exited','returncode':0,'private':'not printed'}
    monkeypatch.setattr(runner,'supervise',supervise)
    monkeypatch.setattr(sys,'argv',['runner','bounded','--plan',str(path),'--output-dir',str(tmp_path/'out'),
                                    '--receipt',str(tmp_path/'receipt'),'--key-file','never-read'])
    assert runner.main()==0 and captured['timeout_seconds']==120
    assert captured['command'][:3]==[sys.executable,str(Path(runner.__file__).resolve()),'run']
    assert json.loads(capsys.readouterr().out)=={'command':'bounded','status':'worker_exited'}
