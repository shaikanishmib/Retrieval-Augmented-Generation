"""Answer from a local document mapping with the frozen bounded controller."""
import argparse
import json
from agentic_rag.artifacts import ROOT,read
from agentic_rag.models import Generator,Services
from agentic_rag.controller import Controller
from agentic_rag.data import Task

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents",required=True)
    parser.add_argument("--question",required=True)
    parser.add_argument("--strategy",choices=["fixed","adaptive","plan","agent","verify"],default="verify")
    args=parser.parse_args()
    documents=read(args.documents)
    if not isinstance(documents,dict) or not documents or not all(isinstance(k,str) and isinstance(v,str) for k,v in documents.items()):
        parser.error("documents must be a nonempty JSON mapping from source IDs to text")
    freeze=read(ROOT / "reports/selection-freeze.json")
    generator=Generator("flan-base")
    planner=generator if freeze["planner"]=="flan-base" else Generator(freeze["planner"])
    task=Task("local",args.question,documents,{k:[v] for k,v in documents.items()})
    result=Controller(generator,planner,Services(),freeze["config"]).run(task,args.strategy,retrieval=freeze["retrieval"])
    print(json.dumps(result,ensure_ascii=False,indent=2))
