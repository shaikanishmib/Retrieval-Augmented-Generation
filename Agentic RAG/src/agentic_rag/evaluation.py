"""Answer scoring and annotation-based evidence diagnostics kept outside the agent."""
from collections import Counter
import re
import string
import numpy as np

def normalize(text):
    text=text.lower().translate(str.maketrans("","",string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b"," ",text).split())

def answer_score(prediction,answers):
    p=normalize(prediction).split()
    best_em,best_f1=0.,0.
    for answer in answers or [""]:
        g=normalize(answer).split()
        em=float(p==g)
        if not p or not g:
            f1=float(p==g)
        elif (prediction.lower() in {"yes","no","noanswer"} or answer.lower() in {"yes","no","noanswer"}) and p!=g:
            f1=0.
        else:
            common=sum((Counter(p)&Counter(g)).values())
            f1=2*common/(len(p)+len(g))
        best_em=max(best_em,em)
        best_f1=max(best_f1,f1)
    return dict(em=best_em,f1=best_f1)

def score(result,row):
    metrics=answer_score(result["answer"],row["answers"])
    sources={p["document"] for p in result["context"]}
    gold=set(row["gold"])
    cited={p["document"] for p in result["context"] if p["id"] in result["citations"]}
    facts=[(t,i) for t,i in row["facts"] if t in row["sentences"] and 0<=i<len(row["sentences"][t])]
    contained=sum(any(p["document"]==t and " ".join(row["sentences"][t][i].split()) in " ".join(p["text"].split()) for p in result["context"]) for t,i in facts)
    calls=[v for v in result["state"]["trace"] if "action" in v]
    all_seen=set()
    redundant=0
    useful_gold=[]
    for call in calls:
        ids={e["id"] for e in call["returned"]}
        redundant+=bool(ids) and not ids-all_seen
        all_seen|=ids
        useful_gold.append({e["document"] for e in call["returned"]}&gold)
    later_gain=bool(useful_gold) and bool(set().union(*useful_gold[1:])-useful_gold[0]) if len(useful_gold)>1 else False
    metrics.update(evidence_recall=len(sources&gold)/len(gold) if gold else None,
                   supporting_fact_recall=contained/len(facts) if facts else None,
                   citation_precision=len(cited&gold)/len(cited) if cited else None,
                   citation_recall=len(cited&gold)/len(gold) if gold else None,
                   answered=bool(result["answer"]),answerable=bool(row["answers"]),
                   redundant_calls=int(redundant),calls=result["state"]["calls"],successful_calls=sum(c["success"] for c in calls),
                   empty_calls=sum(c["empty"] for c in calls),invalid_actions=result["state"]["invalid_actions"],action_attempts=result["state"]["action_attempts"],
                   additional_evidence_gain=later_gain,model_support=result["support"]["entailment"],
                   model_contradiction=result["support"]["contradiction"],claim_contains_answer=result["claim_contains_answer"],
                   end_to_end_ms=result["end_to_end_ms"],tokens=result["state"]["tokens"],steps=result["state"]["steps"],
                   citation_valid=result["citation_valid"],budget_terminated=result["reason"] in {"max_steps","tool_budget","token_budget"},
                   generation_ms=result["state"]["latencies"]["generation"],planning_ms=result["state"]["latencies"]["planning"],
                   retrieval_ms=result["state"]["latencies"]["retrieval"],verification_ms=result["state"]["latencies"]["verification"])
    return dict(id=row["id"],group=row["group"],kind=row["kind"],**metrics)

def aggregate(rows):
    out={"n":len(rows),"groups":len({r["group"] for r in rows})}
    for k in rows[0]:
        if k in {"id","group","kind"}:
            continue
        values=[float(r[k]) for r in rows if r[k] is not None]
        out[k]=float(np.mean(values)) if values else None
    times=[r["end_to_end_ms"] for r in rows]
    out.update(p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),
               tool_success_rate=sum(r["successful_calls"] for r in rows)/max(1,sum(r["calls"] for r in rows)),
               invalid_action_rate=sum(r["invalid_actions"] for r in rows)/max(1,sum(r["action_attempts"] for r in rows)),
               selective_em=float(np.mean([r["em"] for r in rows if r["answered"]])) if any(r["answered"] for r in rows) else None,
               false_answer_rate=float(np.mean([r["answered"] for r in rows if not r["answerable"]])) if any(not r["answerable"] for r in rows) else None,
               unnecessary_retrieval_proxy="duplicate-evidence calls only; no causal necessity claim")
    return out

def paired_interval(a,b,seed=314159):
    assert [r["id"] for r in a]==[r["id"] for r in b]
    groups=sorted({r["group"] for r in a})
    diffs={g:[x["f1"]-y["f1"] for x,y in zip(a,b) if x["group"]==g] for g in groups}
    rng=np.random.default_rng(seed)
    samples=[np.mean([d for g in rng.choice(groups,len(groups),replace=True) for d in diffs[g]]) for _ in range(2000)]
    return dict(delta=float(np.mean([x["f1"]-y["f1"] for x,y in zip(a,b)])),low=float(np.percentile(samples,2.5)),high=float(np.percentile(samples,97.5)),resampling="question groups for HotpotQA; articles for SQuAD",replicates=2000,groups=len(groups))
