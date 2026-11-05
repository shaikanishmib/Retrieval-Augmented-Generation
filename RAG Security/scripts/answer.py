"""Answer a question over explicitly supplied local documents."""
import argparse
from pathlib import Path
from rag_security.artifacts import ROOT, read
from rag_security.policy import Document, Registry, Principal, output_valid
from rag_security.retrieval import retrieve
from rag_security.models import LocalModel
from rag_security.detectors import Screening
from dataclasses import replace
import json

parser=argparse.ArgumentParser(description='Local evidence-based QA with bounded inputs.')
parser.add_argument('--question',required=True)
parser.add_argument('--document',action='append',required=True)
parser.add_argument('--model',choices=['flan-base','qwen06'],default='flan-base')
parser.add_argument('--defense',choices=['D0','D2','D3','D4','D5','D6','D7'],required=True)
args=parser.parse_args()
if len(args.document)>64:
    parser.error('At most 64 local documents are accepted.')
docs=[Document(f'source-{i}',Path(p).read_text(encoding='utf-8'),title=Path(p).name) for i,p in enumerate(args.document)]
reg=Registry(docs)
principal=Principal('local','reader',frozenset(reg.documents))
selected=retrieve(args.question,docs,reg,principal,k=min(3,len(docs)),integrity=args.defense=='D2')
if args.defense in ('D3','D6','D7'):
    screen=Screening()
    if args.defense=='D3' and any(screen.flagged(d.text) for d in selected):
        selected=[]
    elif args.defense in ('D6','D7'):
        selected=[replace(d,text='\n'.join(line for line in d.text.splitlines() if not screen.flagged(line))) for d in selected]
        selected=[d for d in selected if d.text.strip()]
model=LocalModel(args.model)
result=model.generate(args.question,selected,structured=args.defense in ('D4','D7')) if selected else {'answer':'unknown'}
if args.defense in ('D5','D7') and not output_valid(result['answer'],[d.id for d in selected],reg,principal)[0]:
    result['answer']='unknown'
print(json.dumps({'answer':result['answer'],'sources':[{'id':d.id,'start':d.start,'end':d.start+len(d.text)} for d in selected]},indent=2))
