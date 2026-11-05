"""Staged development selection, frozen evaluation and resumable per-question traces."""
import os
import time
import traceback
import psutil
import torch
from .artifacts import ROOT,read,write,digest,fingerprint
from .models import Generator,Services
from .controller import Controller,SEARCH,LOOKUP
from .data import task
from .evaluation import score,aggregate

class Study:
    def __init__(self):
        self.config=read(ROOT / "configs/study.json")
        torch.manual_seed(self.config["seed"])
        self.identity=fingerprint()
        print("Loading retrieval and verification services",flush=True)
        self.services=Services()
        print("Loading answer generator",flush=True)
        self.generator=Generator("flan-base")
        self.planner=None
        self.resources=dict(encoder_load_ms=self.services.encoder_ms,reranker_load_ms=self.services.reranker_ms,nli_load_ms=self.services.nli_ms,
                            generator_load_ms=self.generator.load_ms,planner_loads={},device=self.services.device,torch=torch.__version__,
                            cuda=torch.version.cuda,seed=self.config["seed"],quantization="none",dtype="float16 generators on CUDA; float32 retrieval and NLI",
                            pid=os.getpid(),source_identity=self.identity)

    def set_planner(self,alias):
        if self.planner is not None and self.planner.alias==alias:
            return
        if self.planner is not None and self.planner is not self.generator:
            self.planner.close()
        self.planner=self.generator if alias=="flan-base" else Generator(alias)
        self.resources["planner_loads"][alias]=self.planner.load_ms

    def run_config(self,label,name,split,strategy,limit=None,overrides=None,**kwargs):
        config={**self.config,**(overrides or {})}
        rows=read(ROOT / f"data/partitions/{name}.json")[split]
        if limit:
            rows=rows[:limit]
        identity=digest([self.identity,label,name,split,strategy,config,kwargs,self.planner.alias,self.generator.alias])
        scores=[]
        manifest=[]
        controller=Controller(self.generator,self.planner,self.services,config)
        began=time.perf_counter()
        for i,row in enumerate(rows):
            out=ROOT / f"data/traces/{identity}/{row['id']}.json"
            if out.exists():
                result=read(out)
                assert result["identity"]==identity
                assert result["row_hash"]==digest(row),"Cached inputs changed"
            else:
                try:
                    result=controller.run(task(row),strategy,oracle=row["gold"] if strategy=="oracle" else None,**kwargs)
                    result.update(identity=identity,question_hash=digest(task(row).__dict__),row_hash=digest(row),configuration=config,
                                  planner=self.planner.alias,generator=self.generator.alias)
                    write(out,result)
                except Exception as exc:
                    write(ROOT / f"data/failures/{identity}-{row['id']}.json",dict(error=repr(exc),traceback=traceback.format_exc(),id=row["id"]))
                    raise
            metrics=score(result,row)
            scores.append(metrics)
            manifest.append(dict(id=row["id"],path=str(out.relative_to(ROOT))))
            if (i+1)%16==0:
                print(label,name,split,f"{i+1}/{len(rows)}",round(time.perf_counter()-began,1),"seconds",flush=True)
        write(ROOT / f"data/scores/{label}-{name}-{split}.json",scores)
        write(ROOT / f"data/manifests/{label}-{name}-{split}.json",dict(identity=identity,configuration=config,strategy=strategy,kwargs=kwargs,rows=manifest))
        write(ROOT / f"reports/aggregates/{label}-{name}-{split}.json",aggregate(scores))
        process=psutil.Process()
        self.resources.update(rss_bytes=process.memory_info().rss,peak_working_set_bytes=getattr(process.memory_info(),"peak_wset",None),
                              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                              available_ram_bytes=psutil.virtual_memory().available)
        write(ROOT / "reports/resources.json",self.resources)
        write(ROOT / f"reports/resource-stages/{label}-{name}-{split}.json",self.resources)
        return aggregate(scores)

    def pilot(self):
        results={}
        n=self.config["pilot_per_dataset"]
        for alias in ["flan-base","qwen06","qwen17"]:
            self.set_planner(alias)
            results[alias]={}
            for name in ["hotpot","squad"]:
                results[alias][name]=self.run_config("planner_"+alias,name,"development","agent",n)
        # This criterion was revised on development traces before any held-out run.
        utility=lambda alias:sum(results[alias][n]["f1"]+.10*(results[alias][n]["evidence_recall"] or 0)-.02*results[alias][n]["invalid_actions"]-.01*results[alias][n]["calls"] for n in results[alias])
        winner=max(results,key=utility)
        write(ROOT / "reports/planner-selection.json",dict(selected=winner,criterion="macro F1 + 0.10 evidence recall - 0.02 invalid actions - 0.01 calls; development only",results=results,source_identity=self.identity))
        return winner

    def development(self):
        selection=read(ROOT / "reports/planner-selection.json")
        assert selection["source_identity"]==self.identity
        self.set_planner(selection["selected"])
        results={}
        for retrieval in ["bm25","dense","hybrid","rerank"]:
            results[retrieval]={n:self.run_config("retrieval_"+retrieval,n,"development","fixed",retrieval=retrieval) for n in ["hotpot","squad"]}
        winner=max(results,key=lambda k:sum(results[k][n]["f1"] for n in results[k]))
        for name in ["hotpot","squad"]:
            for strategy in ["adaptive","plan","agent","verify"]:
                self.run_config(strategy,name,"development",strategy)
        # Validation is a separate diagnostic stage, with no parameter fitting.
        for name in ["hotpot","squad"]:
            for strategy in ["fixed","adaptive","plan","agent","verify"]:
                self.run_config(strategy,name,"validation",strategy,retrieval=winner)
        freeze=dict(planner=selection["selected"],retrieval=winner,config=self.config,source_identity=self.identity,
                    timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),retrieval_development=results,
                    validation_policy="diagnostic only; thresholds and other settings prespecified, no tuning on validation or evaluation")
        freeze["freeze_hash"]=digest(freeze)
        path=ROOT / "reports/selection-freeze.json"
        if path.exists():
            old=read(path)
            assert old["source_identity"]==self.identity
        else:
            write(path,freeze)

    def evaluation(self):
        freeze=read(ROOT / "reports/selection-freeze.json")
        assert freeze["source_identity"]==self.identity
        self.set_planner(freeze["planner"])
        for name in ["hotpot","squad"]:
            for strategy in ["fixed","adaptive","plan","agent","verify","none","oracle"]:
                self.run_config(strategy,name,"evaluation",strategy,retrieval=freeze["retrieval"])

    def ablations(self):
        freeze=read(ROOT / "reports/selection-freeze.json")
        assert freeze["source_identity"]==self.identity
        self.set_planner(freeze["planner"])
        n=self.config["ablation_per_dataset"]
        variants=[("steps1","agent",dict(max_steps=1),{}),("calls2","agent",dict(max_calls=2),{}),
                  ("context280","agent",dict(context_tokens=280),{}),("no_dedup","agent",dict(deduplicate=False),{}),
                  ("search_only","agent",{},dict(allowed=list(SEARCH))),
                  ("bm25_only","agent",{},dict(allowed=["bm25"])),
                  ("verify_zero","verify",dict(verification_threshold=0.),{}),
                  ("verify_high","verify",dict(verification_threshold=.8),{}),
                  ("empty_first","agent",{},dict(corruption="empty")),
                  ("irrelevant_first","agent",{},dict(corruption="irrelevant")),
                  ("injection_first","agent",{},dict(corruption="injection"))]
        for label,strategy,overrides,kwargs in variants:
            for name in ["hotpot","squad"]:
                self.run_config(label,name,"evaluation",strategy,n,overrides=overrides,**kwargs)
