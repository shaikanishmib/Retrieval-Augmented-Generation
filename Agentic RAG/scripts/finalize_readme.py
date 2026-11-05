"""Insert measured held-out findings into the self-contained primary README."""
import re
from agentic_rag.artifacts import ROOT,read

if __name__=='__main__':
    metrics=read(ROOT/'reports/metrics.json')
    intervals=read(ROOT/'reports/paired-intervals.json')
    freeze=read(ROOT/'reports/selection-freeze.json')
    lines=['<!-- measured-results -->','## Recorded results','',f'The development stage selected `{freeze["planner"]}` for planning and `{freeze["retrieval"]}` for the fixed reference. Each row below uses 128 held-out questions per dataset.','',
           '| Strategy | HotpotQA F1 | SQuAD F1 | HotpotQA evidence recall | HotpotQA p50 s |',
           '| --- | ---: | ---: | ---: | ---: |']
    for strategy in ['fixed','adaptive','plan','agent','verify']:
        hotpot=metrics[f'{strategy}-hotpot-evaluation']
        squad=metrics[f'{strategy}-squad-evaluation']
        lines.append(f'| {strategy} | {hotpot["f1"]:.3f} | {squad["f1"]:.3f} | {hotpot["evidence_recall"]:.3f} | {hotpot["p50_ms"]/1000:.2f} |')
    lines+=['','![Measured answer quality on the held-out samples](reports/figures/answer-quality.png)','']
    for name in ['hotpot','squad']:
        difference=intervals[f'agent-{name}-evaluation']
        ratio=metrics[f'agent-{name}-evaluation']['p50_ms']/metrics[f'fixed-{name}-evaluation']['p50_ms']
        lines.append(f'For {name}, the tool-selecting agent\'s F1 difference from fixed RAG was {difference["delta"]:+.3f} (paired 95% interval [{difference["low"]:+.3f}, {difference["high"]:+.3f}]); its median latency was {ratio:.1f} times the fixed reference.\n')
    lines.append('The verification gate answered '+', '.join(f'{metrics[f"verify-{n}-evaluation"]["answered"]:.1%} of {n} questions' for n in ['hotpot','squad'])+'. Read coverage alongside full-set F1 and selective correctness. Source recall and NLI support do not establish faithful attribution.')
    lines+=['','No-retrieval and oracle-document controls, uncertainty intervals, all 11 ablations, failure traces, citation diagnostics and resource observations are available in the benchmark and notebook.','<!-- /measured-results -->','']
    path=ROOT/'README.md'
    text=path.read_text(encoding='utf-8')
    text=re.sub(r'<!-- measured-results -->.*?<!-- /measured-results -->\s*','',text,flags=re.S)
    text=text.replace('## Study design','\n'.join(lines)+'\n## Study design',1)
    path.write_text(text,encoding='utf-8',newline='\n')
    print('README now includes the measured main comparison and uncertainty.')
