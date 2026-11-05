"""Load original annotations and keep evaluation groups separate."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def key(text):
    return hashlib.sha256(text.encode()).hexdigest()

def load(name, development=24, heldout=64):
    raw = json.loads((ROOT / f"data/raw/{name}.json").read_text(encoding="utf-8"))
    rows = []
    invalid_facts = []
    if name == "hotpot":
        assert len({r["_id"] for r in raw}) == len(raw)
        for r in raw:
            docs = {title: "".join(sentences) for title, sentences in r["context"]}
            for title, index in r["supporting_facts"]:
                sentences = dict(r["context"])[title]
                if not 0 <= index < len(sentences):
                    invalid_facts.append({"question_id": r["_id"], "sentence_index": index, "sentence_count": len(sentences)})
            rows.append(dict(id=r["_id"], group=r["_id"], question=r["question"], answers=[r["answer"]], docs=docs,
                             gold=sorted({t for t, _ in r["supporting_facts"]}), facts=r["supporting_facts"], kind=r["type"]))
    else:
        for article in raw["data"]:
            for i, paragraph in enumerate(article["paragraphs"]):
                did = article["title"] + ":" + str(i)
                for q in paragraph["qas"]:
                    for a in q["answers"]:
                        assert paragraph["context"][a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]
                    rows.append(dict(id=q["id"], group=article["title"], question=q["question"], answers=[a["text"] for a in q["answers"]],
                                     docs={did: paragraph["context"]}, gold=[] if q["is_impossible"] else [did], facts=[], kind="unanswerable" if q["is_impossible"] else "answerable"))
    assert len({r["id"] for r in rows}) == len(rows)
    groups = sorted({r["group"] for r in rows}, key=key)
    cut = max(1, len(groups) // 3)
    dev_groups = set(groups[:cut])
    dev = sorted((r for r in rows if r["group"] in dev_groups), key=lambda r: key(r["id"]))[:development]
    test = sorted((r for r in rows if r["group"] not in dev_groups), key=lambda r: key(r["id"]))[:heldout]
    test_text = {r["question"].strip().casefold() for r in test}
    dev = [r for r in dev if r["question"].strip().casefold() not in test_text]
    assert not {r["group"] for r in dev} & {r["group"] for r in test}
    return dev, test, {"original_questions": len(rows), "original_groups": len(groups), "development": len(dev), "heldout": len(test),
                       "development_ids": [r["id"] for r in dev], "heldout_ids": [r["id"] for r in test],
                       "heldout_unanswerable": sum(not r["answers"] for r in test), "invalid_sentence_facts": invalid_facts}
