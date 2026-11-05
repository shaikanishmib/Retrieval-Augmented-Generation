"""Publication figures derived from aggregate measurements."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((ROOT/'reports'/name).read_text(encoding='utf-8'))

def figure(kind):
    metrics=read('metrics.json')['evaluation']
    detectors=read('detectors.json')['evaluation']
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    if kind=='families':
        p=read('partitions.json')
        labels=[f'{s}: {a["family"]}' for s in p['attacks'] for a in p['attacks'][s]]
        ax.barh(labels,list(range(1,len(labels)+1)),color='steelblue')
        # One fixed payload per family entry, rather than cumulative row number.
        ax.clear()
        ax.barh(labels,[1]*len(labels),color='steelblue')
        ax.set(xlabel='Distinct fixed payload definitions',title='Attack diversity (repeated tasks do not add payloads)')
        fig.set_size_inches(10,6)
    elif kind=='asr':
        names=list(metrics)
        vals=[metrics[n]['strict_asr'] for n in names]
        intervals=np.array([metrics[n]['asr_wilson_95'] for n in names])
        ax.bar(range(len(names)),vals,color=['#4676a9' if n.startswith('flan') else '#b87035' for n in names])
        ax.errorbar(range(len(names)),vals,yerr=[np.maximum(0,np.array(vals)-intervals[:,0]),np.maximum(0,intervals[:,1]-np.array(vals))],fmt='none',color='black',capsize=2)
        ax.set_xticks(range(len(names)),names,rotation=70,ha='right')
        ax.set(ylabel='Strict attack success rate',ylim=(0,1),title='Held-out objectives; Wilson intervals assume independent attempts')
    elif kind=='utility':
        names=list(metrics)
        x=np.arange(len(names))
        ax.bar(x-.18,[metrics[n]['clean_completion'] for n in names],.36,label='Clean completion')
        ax.bar(x+.18,[metrics[n]['attacked_completion'] for n in names],.36,label='Completion under attack')
        ax.set_xticks(x,names,rotation=70,ha='right')
        ax.set(ylabel='Task completion',ylim=(0,1),title='Utility includes refusals and policy failures')
        ax.legend()
    elif kind=='tradeoff':
        for n,m in metrics.items():
            ax.scatter(m['strict_asr'],m['attacked_completion'],marker='o' if n.startswith('flan') else 's')
            ax.annotate(n,(m['strict_asr'],m['attacked_completion']),xytext=(4,4),textcoords='offset points',fontsize=7)
        ax.set(xlabel='Strict ASR (lower is better)',ylabel='Completion under attack',xlim=(-.03,1),ylim=(-.03,1),title='Security and legitimate task utility')
    elif kind=='detection':
        x=np.arange(len(detectors))
        ax.bar(x-.25,[d['precision'] for d in detectors.values()],.25,label='Precision')
        ax.bar(x,[d['recall'] for d in detectors.values()],.25,label='Recall')
        ax.bar(x+.25,[d['pr_auc_average_precision'] for d in detectors.values()],.25,label='Average precision')
        ax.set_xticks(x,list(detectors))
        ax.set(ylim=(0,1),title='Held-out injection detection at frozen thresholds')
        ax.legend()
    elif kind=='false-positives':
        names=list(detectors)
        vals=np.array([detectors[n]['false_positive_rate'] for n in names])
        ci=np.array([detectors[n]['false_positive_interval'] for n in names])
        ax.bar(names,vals,color='#aa5b50')
        ax.errorbar(names,vals,yerr=[np.maximum(0,vals-ci[:,0]),np.maximum(0,ci[:,1]-vals)],fmt='none',color='black',capsize=3)
        ax.set(ylabel='Benign passage false-positive rate',ylim=(0,1),title='Original passages and quoted attack discussions')
    elif kind=='latency':
        names=list(metrics)
        x=np.arange(len(names))
        ax.bar(x-.18,[metrics[n]['p50_seconds'] for n in names],.36,label='p50')
        ax.bar(x+.18,[metrics[n]['p95_seconds'] for n in names],.36,label='p95')
        ax.set_xticks(x,names,rotation=70,ha='right')
        ax.set(ylabel='Seconds',title='Measured inference plus per-policy preparation; loading separate')
        ax.legend()
    elif kind=='uncertainty':
        pairs=read('paired.json')
        names=[p['model']+'/'+p['defense'] for p in pairs]
        vals=np.array([p['asr_difference'] for p in pairs])
        ci=np.array([p['asr_difference_group_bootstrap_95'] for p in pairs])
        ax.errorbar(vals,np.arange(len(names)),xerr=[np.maximum(0,vals-ci[:,0]),np.maximum(0,ci[:,1]-vals)],fmt='o',capsize=3)
        ax.set_yticks(range(len(names)),names)
        ax.axvline(0,color='gray',linestyle=':')
        ax.set(xlabel='Defended ASR minus D0',title='Paired task-group bootstrap, 95% pointwise intervals')
        fig.set_size_inches(9,6)
    elif kind=='positions':
        rows=read('positions-details.json')
        for model in sorted({r['model'] for r in rows}):
            for defense in sorted({r['defense'] for r in rows}):
                rs=[r for r in rows if r['model']==model and r['defense']==defense]
                values=[np.mean([r['strict_asr'] for r in rs if r['position']==p]) for p in ('start','middle','end')]
                ax.plot(['start','middle','end'],values,marker='o',label=model+'/'+defense)
        ax.set(ylabel='ASR across two forced evidence ranks',ylim=(0,1),title='Small prespecified placement diagnostic')
        ax.legend()
    else:
        raise ValueError(kind)
    return fig

KINDS=['families','asr','utility','tradeoff','detection','false-positives','latency','uncertainty','positions']
if __name__=='__main__':
    directory=ROOT/'reports/figures'
    directory.mkdir(exist_ok=True)
    for kind in KINDS:
        fig=figure(kind)
        fig.savefig(directory/(kind+'.png'),dpi=150)
        plt.close(fig)
