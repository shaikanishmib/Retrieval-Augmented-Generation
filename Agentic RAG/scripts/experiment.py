"""Execute one research stage, reusing unchanged per-question checkpoints."""
import argparse
from agentic_rag.study import Study

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["pilot","development","evaluation","ablations"])
    args=parser.parse_args()
    study=Study()
    getattr(study,args.stage)()
