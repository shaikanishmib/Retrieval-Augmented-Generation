"""Reconstruct published metrics from retained traces and produce matched comparisons."""
from collections import Counter
import csv
import io
from agentic_rag.artifacts import ROOT,read,write,sha,fingerprint
from agentic_rag.evaluation import score,aggregate,paired_interval,normalize,answer_score

def analyze():
    identity=fingerprint()
    metrics={}
    per_question={}
    manifest_hashes={}
    reliability={}
    for path in sorted((ROOT / "data/manifests").glob("*.json")):
        manifest=read(path)
        label=path.stem
        name,split=label.rsplit("-",2)[-2:]
        source_rows={r["id"]:r for r in read(ROOT / f"data/partitions/{name}.json")[split]}
        results=[]
        traces=[]
        for entry in manifest["rows"]:
            raw_path=ROOT / entry["path"]
            raw=read(raw_path)
            assert raw["identity"]==manifest["identity"]
            method_label=label.rsplit('-',2)[0]
            from agentic_rag.artifacts import digest
            expected=digest([identity,method_label,name,split,manifest['strategy'],manifest['configuration'],manifest['kwargs'],raw['planner'],raw['generator']])
            assert raw['identity']==expected,('Stale configuration/source identity',label)
            assert raw['state']['status']=='finished'
            assert raw['state']['calls']<=raw['configuration']['max_calls']
            assert raw['state']['steps']<=raw['configuration']['max_steps']
            assert raw['state']['tokens']<=raw['configuration']['max_tokens']
            assert raw['state']['invalid_actions']<=raw['state']['action_attempts']
            row=source_rows[entry["id"]]
            from agentic_rag.artifacts import digest
            assert raw["row_hash"]==digest(row)
            results.append(score(raw,row))
            traces.append(raw)
        recomputed=aggregate(results)
        assert recomputed==read(ROOT / f"reports/aggregates/{label}.json"),label
        metrics[label]=recomputed
        per_question[label]=results
        manifest_hashes[label]=dict(manifest_sha256=sha(path),trace_hash=digest({r["id"]:sha(ROOT / r["path"]) for r in manifest["rows"]}))
        reliability[label]=dict(termination_reasons=dict(Counter(r["reason"] for r in traces)),
                               controller_stop_reasons=dict(Counter(r["control_reason"] for r in traces)),
                               termination_failures=sum(r['state']['status']!='finished' or r['state']['steps']>r['configuration']['max_steps'] or r['state']['calls']>r['configuration']['max_calls'] or r['state']['tokens']>r['configuration']['max_tokens'] for r in traces),
                               repeated_action_rejections=sum('Repeated action rejected' in t.get('error','') for r in traces for t in r['state']['trace']),
                               generative_input_tokens=sum(g['input_tokens'] for r in traces for g in r['state']['generations']),
                               generative_output_tokens=sum(g['output_tokens'] for r in traces for g in r['state']['generations']),
                               empty_successful_calls=sum(t.get('success',False) and not t.get('returned') for r in traces for t in r['state']['trace'] if 'action' in t),
                               failed_dispatches=sum(not t['success'] for r in traces for t in r['state']['trace'] if 'action' in t),
                               claim_proposals=sum(bool(r['claim']) for r in traces),
                               tool_distribution=dict(Counter(t["action"]["tool"] for r in traces for t in r["state"]["trace"] if "action" in t)),
                               support_truncations=sum(r["support"]["truncated"] for r in traces),
                               answer_without_valid_citation=sum(bool(r["answer"]) and not r["citation_valid"] for r in traces),
                               invalid_trace_events=sum(not t["success"] for r in traces for t in r["state"]["trace"]),
                               source_identity=identity)
        annotated=[r for r in traces if source_rows[r['id']]['gold']]
        if annotated:
            import numpy as np
            reliability[label]['memory_evidence_recall']=float(np.mean([len({e['document'] for e in r['evidence']}&set(source_rows[r['id']]['gold']))/len(source_rows[r['id']]['gold']) for r in annotated]))
            reliability[label]['returned_evidence_recall']=float(np.mean([len({e['document'] for t in r['state']['trace'] for e in t.get('returned',[])}&set(source_rows[r['id']]['gold']))/len(source_rows[r['id']]['gold']) for r in annotated]))
        else:
            reliability[label]['memory_evidence_recall']=None
            reliability[label]['returned_evidence_recall']=None
        claims=[r for r in traces if r['claim']]
        accepted_claims=[r for r in claims if r['answer']]
        reliability[label]['mean_nli_entailment_on_claims']=sum(r['support']['entailment'] for r in claims)/len(claims) if claims else None
        reliability[label]['mean_nli_entailment_on_accepted_claims']=sum(r['support']['entailment'] for r in accepted_claims)/len(accepted_claims) if accepted_claims else None
        scored={r['id']:r for r in results}
        reliability[label]['high_support_answer_em_mismatch']=sum(r['support']['entailment']>=.5 and scored[r['id']]['em']==0 for r in accepted_claims)
        rejected=[r for r in traces if r.get('rejected_answer')]
        reliability[label]['rejected_proposals']=len(rejected)
        reliability[label]['rejected_exact_match_proposals']=sum(answer_score(r['rejected_answer'],source_rows[r['id']]['answers'])['em']==1 for r in rejected)
    intervals={}
    for name in ["hotpot","squad"]:
        baseline=per_question[f"fixed-{name}-evaluation"]
        for strategy in ["adaptive","plan","agent","verify","none","oracle"]:
            label=f"{strategy}-{name}-evaluation"
            intervals[label]=paired_interval(per_question[label],baseline)
        for label,rows in per_question.items():
            if label.endswith(f"-{name}-evaluation") and len(rows)<len(baseline):
                reference="verify" if label.startswith("verify_") else "agent"
                intervals[label]=paired_interval(rows,per_question[f"{reference}-{name}-evaluation"][:len(rows)])
        abstain=[dict(id=r["id"],group=r["group"],f1=float(not r["answers"])) for r in read(ROOT / f"data/partitions/{name}.json")["evaluation"]]
        intervals[f"always_abstain-{name}"]=paired_interval(abstain,baseline)
        write(ROOT / f"reports/always-abstain-{name}.json",dict(n=len(abstain),em=sum(r["f1"] for r in abstain)/len(abstain),f1=sum(r["f1"] for r in abstain)/len(abstain)))
    write(ROOT / "reports/metrics.json",metrics)
    by_type={label:{kind:aggregate([r for r in rows if r['kind']==kind]) for kind in sorted({r['kind'] for r in rows})} for label,rows in per_question.items()}
    write(ROOT/'reports/question-type-metrics.json',by_type)
    write(ROOT / "reports/paired-intervals.json",intervals)
    primary={f'{strategy}-{name}':intervals[f'{strategy}-{name}-evaluation'] for name in ['hotpot','squad'] for strategy in ['adaptive','plan','agent','verify','none','oracle']}
    preview=ROOT/'reports/primary-paired-intervals.json'
    if preview.exists() and read(preview)['source_identity']==identity:
        assert read(preview)['results']==primary,'Primary interim/final uncertainty mismatch'
    write(preview,dict(source_identity=identity,results=primary,scope='Primary conditions; verified against the final trace reconstruction.'))
    write(ROOT / "reports/reliability.json",reliability)
    def raw_records(label):
        return [read(ROOT/e['path']) for e in read(ROOT/f'data/manifests/{label}.json')['rows']]
    utility={}
    recovery={}
    for name in ['hotpot','squad']:
        one=raw_records(f'steps1-{name}-evaluation')
        agent=raw_records(f'agent-{name}-evaluation')[:len(one)]
        a_scores=per_question[f'agent-{name}-evaluation'][:len(one)]
        b_scores=per_question[f'steps1-{name}-evaluation']
        extra=[i for i,(a,b) in enumerate(zip(agent,one)) if a['state']['calls']>b['state']['calls']]
        utility[name]=dict(n=len(one),extra_call_questions=len(extra),
                           same_answer_with_extra_calls=sum(normalize(agent[i]['answer'])==normalize(one[i]['answer']) for i in extra),
                           no_f1_gain_with_extra_calls=sum(a_scores[i]['f1']<=b_scores[i]['f1'] for i in extra),
                           improved_f1_with_extra_calls=sum(a_scores[i]['f1']>b_scores[i]['f1'] for i in extra),
                           interpretation='Matched one-step versus multi-step budget contrast. Same-answer/no-gain counts are waste proxies, not annotated causal necessity.')
        annotations={r['id']:r for r in read(ROOT/f'data/partitions/{name}.json')['evaluation']}
        for corruption in ['empty_first','irrelevant_first','injection_first']:
            cases=raw_records(f'{corruption}-{name}-evaluation')
            applied=[r for r in cases if [t for t in r['state']['trace'] if 'action' in t] and [t for t in r['state']['trace'] if 'action' in t][0]['success']]
            answerable_applied=sum(bool(annotations[r['id']]['gold']) for r in applied)
            restored=0
            for r in applied:
                calls=[t for t in r['state']['trace'] if 'action' in t and t['success']]
                gold=set(annotations[r['id']]['gold'])
                restored+=bool(gold) and any(e['document'] in gold for t in calls[1:] for e in t['returned'])
            recovery[f'{corruption}-{name}']=dict(n=len(cases),intervention_applied=len(applied),answerable_interventions=answerable_applied,later_gold_recovered=restored,
                                                 recovery_rate=restored/answerable_applied if answerable_applied else None,
                                                 overall_rate=restored/len(applied) if applied else None,
                                                 limitation='Intervention requires the first dispatch to succeed. Primary recovery denominator uses answerable cases with gold documents. Recovery means later annotated document retrieval, not necessarily a correct answer.')
    write(ROOT/'reports/call-utility.json',utility)
    write(ROOT/'reports/retrieval-recovery.json',recovery)
    write(ROOT / "reports/experiment-provenance.json",dict(source_identity=identity,artifacts=manifest_hashes))
    columns=["configuration","n","groups","em","f1","evidence_recall","supporting_fact_recall","citation_precision","citation_recall","answered","selective_em","calls","invalid_actions","tool_success_rate","p50_ms","p95_ms","tokens"]
    buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer,fieldnames=columns,extrasaction="ignore",lineterminator="\n")
    writer.writeheader()
    for label,values in metrics.items():
        writer.writerow(dict(configuration=label,**values))
    (ROOT / "reports/metrics.csv").write_text(buffer.getvalue(),encoding="utf-8")
    text=["# Measured comparison","", "All values below were recomputed from retained local traces. These are local held-out partitions of public releases, not official test results.","", "| Dataset | Strategy | n | EM | F1 | Evidence recall | Coverage | Calls | p50 s | p95 s |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in ["hotpot","squad"]:
        for strategy in ["fixed","adaptive","plan","agent","verify","none","oracle"]:
            r=metrics[f"{strategy}-{name}-evaluation"]
            recall=f'{r["evidence_recall"]:.3f}' if r["evidence_recall"] is not None else "n/a"
            text.append(f'| {name} | {strategy} | {r["n"]} | {r["em"]:.3f} | {r["f1"]:.3f} | {recall} | {r["answered"]:.3f} | {r["calls"]:.2f} | {r["p50_ms"]/1000:.2f} | {r["p95_ms"]/1000:.2f} |')
    text.extend(["", "## Matched differences from fixed RAG", "", "Intervals resample questions for HotpotQA and complete articles for SQuAD (2,000 replicates). They are pointwise, unadjusted intervals.", "", "| Dataset | Strategy | F1 difference | 95% interval |", "| --- | --- | ---: | --- |"])
    for name in ["hotpot","squad"]:
        for strategy in ["adaptive","plan","agent","verify"]:
            r=intervals[f"{strategy}-{name}-evaluation"]
            text.append(f'| {name} | {strategy} | {r["delta"]:+.3f} | [{r["low"]:+.3f}, {r["high"]:+.3f}] |')
    text.extend(["", "## Interpretation limits", "", "Source-label citation agreement is distinct from NLI support. Citations are assigned after generation by lexical overlap; generated claims can alter answer meaning. Verification may trade answer coverage for fewer unsupported proposals. The no-retrieval control may answer without citations by design; deployable paths reject answers with no valid source.", "", "Oracle supplies gold source documents but still uses bounded context and the same generator. It is a diagnostic gold-document reference, not a guaranteed upper bound or an answer oracle. Additional calls, context changes and generation retries are confounded in verification-guided control; gains cannot be attributed solely to new evidence.", "", "Full development, validation, model checks, ablations, reliability and resource measurements are in the adjacent JSON/CSV reports and the public notebook. Timing is a single sequential pass with model loading excluded from per-question latency. Fixed strategy includes the same claim-generation and NLI diagnostics as the other methods, so it costs more than a deployment without diagnostics."])
    (ROOT / "reports/benchmark.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    required=[f"{s}-{n}-evaluation" for s in ["fixed","adaptive","plan","agent","verify","none","oracle"] for n in ["hotpot","squad"]]
    write(ROOT / "reports/execution-ledger.json",dict(completed_configurations=len(metrics),completed_question_runs=sum(r["n"] for r in metrics.values()),
                                                     primary_expected=required,primary_complete=all(k in metrics for k in required),
                                                     failed_records=len(list((ROOT / "data/failures").glob("*.json"))),
                                                     development_revision="Prior pilots and interrupted development retained locally. Before held-out evaluation: clarified schemas, required sources, revised planner utility, fixed token/action accounting, retained stopping reasons, and corrected the contradictory single-object example in the plan-array prompt. An eight-question development probe checked the corrected contract.",
                                                     unfinished=["MuSiQue/2Wiki external replication","quantized larger planner models","multiple timing seeds","independent human support annotation","causal call-necessity labels","graph extraction beyond title mentions","per-attempt NLI probability histories; current traces retain final proposal scores and cumulative verifier time"]))
    print("Reconstructed",len(metrics),"configurations from",sum(r["n"] for r in metrics.values()),"question runs.")
    return metrics

if __name__ == "__main__":
    import sys
    if '--primary-only' in sys.argv:
        results={}
        for name in ['hotpot','squad']:
            baseline=read(ROOT/f'data/scores/fixed-{name}-evaluation.json')
            for strategy in ['adaptive','plan','agent','verify','none','oracle']:
                rows=read(ROOT/f'data/scores/{strategy}-{name}-evaluation.json')
                results[f'{strategy}-{name}']=paired_interval(rows,baseline)
        write(ROOT/'reports/primary-paired-intervals.json',dict(source_identity=fingerprint(),results=results,scope='Primary conditions only; full analysis follows completed ablations.'))
        for label,value in results.items():
            print(label,value)
    else:
        analyze()
