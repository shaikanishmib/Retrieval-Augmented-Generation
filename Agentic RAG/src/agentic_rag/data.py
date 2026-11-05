"""Retain authentic annotations while exposing only questions and candidates to agents."""
from dataclasses import dataclass
from .artifacts import ROOT, REFERENCE, read, write, digest

@dataclass(frozen=True)
class Task:
    id: str
    question: str
    documents: dict
    sentences: dict

def prior_ids():
    ids = set()
    for p in (REFERENCE / "data/upgrade/manifests").glob("*.json"):
        ids.update(r["id"] for r in read(p).get("rows", []))
    for p in (REFERENCE / "data/experiments").glob("*-traces.json"):
        obj = read(p)
        if isinstance(obj, list):
            ids.update(r.get("id", r.get("question_id", "")) for r in obj)
    return ids

def load_rows(name):
    raw = read(ROOT / ("data/raw/squad-train.json" if name=="squad" else "data/raw/hotpot.json"))
    rows = []
    if name == "hotpot":
        for r in raw:
            sentences = dict(r["context"])
            rows.append(dict(id=r["_id"], group=r["_id"], question=r["question"],
                             documents={k: "".join(v) for k,v in sentences.items()}, sentences=sentences,
                             answers=[r["answer"]], gold=sorted({t for t,i in r["supporting_facts"]}), facts=r["supporting_facts"], kind=r["type"]))
    else:
        for a in raw["data"]:
            for i,p in enumerate(a["paragraphs"]):
                did = f'{a["title"]}:{i}'
                for q in p["qas"]:
                    for ans in q["answers"]:
                        assert p["context"][ans["answer_start"]:ans["answer_start"]+len(ans["text"])] == ans["text"]
                    rows.append(dict(id=q["id"], group=a["title"], question=q["question"], documents={did:p["context"]},
                                     sentences={did:[p["context"]]}, answers=[v["text"] for v in q["answers"]],
                                     gold=[] if q["is_impossible"] else [did], facts=[], kind="unanswerable" if q["is_impossible"] else "answerable"))
    assert len({r["id"] for r in rows}) == len(rows)
    return rows

def prepare():
    cfg = read(ROOT / "configs/study.json")
    excluded = prior_ids()
    reports = {}
    for name in ["hotpot", "squad"]:
        rows = load_rows(name)
        # Exclude previously used articles as well as exact questions on SQuAD.
        used_groups = {r["group"] for r in rows if r["id"] in excluded}
        available = [r for r in rows if r["id"] not in excluded and (name != "squad" or r["group"] not in used_groups)]
        groups = sorted({r["group"] for r in available}, key=lambda x:digest(["agentic-v1",x]))
        # Allocation occurs at group level before selecting question counts.
        cuts = [len(groups)//3, 2*len(groups)//3]
        pools = [set(groups[:cuts[0]]),set(groups[cuts[0]:cuts[1]]),set(groups[cuts[1]:])]
        partitions = {}
        for split,pool in zip(["development","validation","evaluation"],pools):
            candidates = sorted((r for r in available if r["group"] in pool), key=lambda r:digest(["agentic-v1",r["id"]]))
            partitions[split] = candidates[:cfg[f"{split}_per_dataset"]]
            assert len(partitions[split]) == cfg[f"{split}_per_dataset"], (name,split,len(candidates))
        question_sets = [{r["question"].strip().casefold() for r in rs} for rs in partitions.values()]
        for i in range(3):
            for j in range(i):
                assert not question_sets[i] & question_sets[j], "Duplicate question across splits"
        invalid = [{"id":r["id"],"title":t,"index":i} for r in rows for t,i in r["facts"] if t not in r["sentences"] or not 0<=i<len(r["sentences"][t])]
        write(ROOT / f"data/partitions/{name}.json",partitions)
        reports[name] = dict(original_questions=len(rows), original_groups=len({r["group"] for r in rows}),
                             previous_questions_excluded=sum(r["id"] in excluded for r in rows), previous_groups_excluded=len(used_groups) if name=="squad" else 0,
                             available_questions=len(available), invalid_facts=invalid,
                             duplicate_question_texts=len(rows)-len({r["question"].strip().casefold() for r in rows}),
                             partitions={k:dict(count=len(v),groups=len({r["group"] for r in v}),ids=[r["id"] for r in v],
                                                unanswerable=sum(not r["answers"] for r in v),hash=digest(v)) for k,v in partitions.items()})
    write(ROOT / "reports/data-validation.json",reports)
    return reports

def task(row):
    return Task(row["id"],row["question"],row["documents"],row["sentences"])
