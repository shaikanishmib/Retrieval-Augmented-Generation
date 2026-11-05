"""Record final checks and publication provenance from completed artifacts."""
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter
from advanced_rag.upgrade.artifacts import ROOT, REPORTS, read, save, preserved

preserved()
suites = list(ET.parse(REPORTS / "tests.xml").getroot().iter("testsuite"))
tests = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
assert tests["errors"] == tests["failures"] == tests["skipped"] == 0
review = read(REPORTS / "support-review.json")
assert len(review["cases"]) == 20
assert dict(Counter(case["final_label"] for case in review["cases"])) == review["final_counts"]
notebook = read(REPORTS / "notebook-verification.json")
assert notebook["sources_identical"] and notebook["public_outputs_empty"] and notebook["executed_local_unchanged"]
freeze = read(REPORTS / "selection-freeze.json")
assert hashlib.sha256((REPORTS / "selection.json").read_bytes()).hexdigest() == freeze["selection_sha256"]
files = [ROOT / "models.upgrade.lock.json", ROOT / "configs/upgrade-study.json", REPORTS / "environment.json",
         REPORTS / "requirements-lock.txt", REPORTS / "experiment-identity.json", REPORTS / "metrics.json",
         REPORTS / "before-after.json", REPORTS / "support-review.json", REPORTS / "selection.json"]
files += [ROOT / "src/advanced_rag/answer_support.py", REPORTS / "answer-support-selection.json", REPORTS / "answer-support-results.json",
          REPORTS / "answer-support-adoption.json", REPORTS / "answer-support-analysis.json"]
hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
save(REPORTS / "publication-review.json", {"baseline_commit": "92a85e1485be22bc762422b52bb554c09b8590f8", "baseline_preserved": True,
     "tests": tests, "syntax_checks": "Source, scripts, tests, and notebook code compiled",
     "notebook": notebook, "provenance_sha256": hashes,
     "prose_review": "README, design, notebook, reports, CLI messages, comments, and commit description reviewed using the installed Humanizer 3.1.0 instructions",
     "support_review": "Twenty matched cases reviewed by the coding assistant with references visible; no independent human labels",
     "paid_api_charges_usd": 0,
     "limitations": ["Bounded candidate pools and public development releases", "Small development and controlled-comparison subsets",
                     "Model-based NLI scores can miss relevance and entailment errors", "Citation-capable model selection is distinct from plain-answer model selection",
                     "No independent human support annotations", "Per-question seeds do not guarantee cross-device bitwise identity",
                     "Question-conditioned follow-up reuses evaluation outputs after the frozen study; untouched confirmation remains pending",
                     "Cache reuse assumes the recorded hardware and runtime", "Unquantized 4B and larger retrieval candidates were not evaluated",
                     "Only two of five synthetic probes produce the expected answer; three fail JSON parsing",
                     "No production injection-resistance claim", "Whole-process resource peaks and limited repeated timing"]})
print(tests)
