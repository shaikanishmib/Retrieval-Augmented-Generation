"""Check the corrected plan schema on development questions before the freeze."""
from agentic_rag.study import Study
from agentic_rag.artifacts import ROOT,write

if __name__=='__main__':
    study=Study()
    study.set_planner('qwen17')
    results={name:study.run_config('plan_probe',name,'development','plan',4) for name in ['hotpot','squad']}
    write(ROOT/'reports/plan-schema-probe.json',results)
    print(results)
