from agentic_rag.data import prepare
from agentic_rag.data import load_rows
from agentic_rag.artifacts import ROOT,read,write,digest
if __name__ == "__main__":
    lock=ROOT / "reports/data-validation.json"
    if lock.exists():
        reports=read(lock)
        for name,report in reports.items():
            rows={r["id"]:r for r in load_rows(name)}
            partitions={split:[rows[i] for i in values["ids"]] for split,values in report["partitions"].items()}
            for split,values in report["partitions"].items():
                assert digest(partitions[split])==values["hash"], (name,split)
            write(ROOT / f"data/partitions/{name}.json",partitions)
    else:
        reports=prepare()
    for name,report in reports.items():
        print(name, {k:v["count"] for k,v in report["partitions"].items()}, "available:",report["available_questions"])
