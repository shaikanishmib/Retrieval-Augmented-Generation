"""Small controlled answer-model comparison; never selects from evaluation data."""
from agentic_rag.study import Study
from agentic_rag.models import Generator
from agentic_rag.artifacts import ROOT,read,write

if __name__ == "__main__":
    study=Study()
    study.set_planner(read(ROOT / "reports/planner-selection.json")["selected"])
    results={}
    for alias in ["flan-base","qwen06","qwen17"]:
        if alias!="flan-base":
            # Planner is unused by fixed retrieval; release it before loading another generator.
            if study.planner is not study.generator:
                study.planner.close()
            study.generator.close()
            study.generator=Generator(alias)
            study.planner=study.generator
            study.resources["generator_load_ms"]=study.generator.load_ms
        results[alias]={name:study.run_config("answer_"+alias,name,"development","fixed",study.config["pilot_per_dataset"]) for name in ["hotpot","squad"]}
    write(ROOT / "reports/generator-comparison.json",dict(results=results,policy="FLAN-T5-base prespecified for primary controlled comparisons; this eight-question-per-dataset check is exploratory, with model-specific tokenizer/context packing."))
