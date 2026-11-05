import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_data.research import main

if __name__ == "__main__":
    print(f"Research process PID: {os.getpid()}", flush=True)
    main()
