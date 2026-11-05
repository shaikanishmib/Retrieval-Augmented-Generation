"""Reconcile question-conditioned validation and its selected operating point."""
from collections import Counter
from statistics import mean
from advanced_rag.data import load
from advanced_rag.pipeline import answer_scores
from advanced_rag.answer_support import accept
from advanced_rag.upgrade.grounding import gate
from advanced_rag.upgrade.artifacts import ROOT, LOCAL, REPORTS, read, save
from advanced_rag.upgrade.evaluate import cluster_interval

selection = read(REPORTS / "selection.json")
question = read(REPORTS / "answer-support-selection.json")["selection"]
adoption = read(REPORTS / "answer-support-adoption.json")["decisions"]
report = {}
for name in ("hotpot", "squad"):
    _, test, _ = load(name,64,320)
    report[name] = {}
    for split, rows in (("fresh",test[64:]),("original",test[:64])):
        raw_manifest = read(LOCAL / "manifests" / f"selected_raw-{name}-{split}.json")
        checked_manifest = read(LOCAL / "answer-support" / f"{name}-{split}-manifest.json")
        nli_f1, qa_f1, default_scores, precisions, recalls, flags, qa_times, nli_times = [], [], [], [], [], [], [], []
        for row, raw_item, checked_item in zip(rows,raw_manifest["rows"],checked_manifest["rows"]):
            assert row["id"] == raw_item["id"] == checked_item["id"]
            raw = read(ROOT / raw_item["path"])["result"]
            checked = read(ROOT / checked_item["path"])
            qa = accept(checked,question[name]["agreement"],question[name]["support"])
            nli = gate(raw,selection["thresholds"][name]["selected"])
            nli_f1.append(answer_scores(nli["final_answer"],row["answers"])[1])
            qa_f1.append(answer_scores(qa["final_answer"],row["answers"])[1])
            chosen = qa if adoption[name]["enabled"] else nli
            em,f1 = answer_scores(chosen["final_answer"],row["answers"])
            default_scores.append({"em":em,"f1":f1,"answered":bool(chosen["final_answer"]),"answerable":bool(row["answers"])})
            cited = {raw["evidence"][s]["document_id"] for s in qa["final_citations"]}
            gold = set(row["gold"])
            if cited:
                precisions.append(len(cited & gold)/len(cited))
            if gold:
                recalls.append(len(cited & gold)/len(gold))
            flags.append({"kind":row["kind"],"flag":qa["flag"]})
            qa_times.append(checked["qa_ms"])
            nli_times.append(checked["nli_ms"])
        negatives = [r for r in default_scores if not r["answerable"]]
        positives = [r for r in default_scores if r["answerable"]]
        accepted = [r for r in default_scores if r["answered"]]
        report[name][split] = {"n":len(rows), "question_check_vs_nli_f1":cluster_interval(nli_f1,qa_f1,[r['group'] for r in rows]),
            "qa_citation_source_precision":mean(precisions) if precisions else None,"qa_citation_source_recall":mean(recalls) if recalls else None,
            "qa_ms":mean(qa_times),"nli_ms":mean(nli_times),
            "default":{"question_check_enabled":adoption[name]["enabled"],"em":mean(r['em'] for r in default_scores),"f1":mean(r['f1'] for r in default_scores),
                       "coverage":mean(r['answered'] for r in default_scores),"answerable_f1":mean(r['f1'] for r in positives) if positives else None,
                       "unanswerable_recall":mean(not r['answered'] for r in negatives) if negatives else None,
                       "selective_risk":mean(1-r['em'] for r in accepted) if accepted else None},
            "flags_by_kind":{kind:dict(Counter(r['flag'] for r in flags if r['kind']==kind)) for kind in sorted({r['kind'] for r in flags})}}
save(REPORTS / "answer-support-analysis.json",report)
print(report)
