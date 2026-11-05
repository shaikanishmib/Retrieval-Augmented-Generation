"""Check question-conditioned answer agreement alongside claim entailment."""
import time
from .pipeline import answer_scores, normalize
from .upgrade.grounding import parse, verify
from .upgrade.engine import plain_prompt

def candidate(raw):
    text = next((call["text"] for call in reversed(raw["calls"]) if call["kind"] == "answer"), "")
    parsed = parse(text, raw["evidence"])
    # A literal anchor failure is a candidate for semantic checking, not automatic acceptance.
    if parsed["reason"] == "answer_not_in_claims":
        parsed["valid"] = True
        parsed["reason"] = "requires_question_check"
    return parsed

def check(question, raw, qa_generator, verifier):
    started = time.perf_counter()
    parsed = candidate(raw)
    result = {**parsed, "support": 0., "claim_scores": [], "citation_scores": [], "qa_answer": "", "qa_agreement": 0.,
              "qa_checked": False, "qa_ms": 0., "nli_ms": 0., "qa_input_tokens": 0, "qa_output_tokens": 0}
    if not parsed["valid"] or not parsed["answer"]:
        result["validation_ms"] = (time.perf_counter() - started) * 1000
        return result
    evidence = {s: raw["evidence"][s] for s in parsed["citations"]}
    prompt = plain_prompt(question, evidence)
    if qa_generator.count(prompt) > qa_generator.limit:
        result["reason"] = "qa_context_limit"
        result["validation_ms"] = (time.perf_counter() - started) * 1000
        return result
    response = qa_generator.generate(prompt, limit=32)
    answer = response["text"]
    if normalize(answer) in {"unknown", "unanswerable", "no answer", "insufficient evidence"}:
        answer = ""
    result.update(qa_answer=answer, qa_checked=True, qa_ms=response["generation_ms"],
                  qa_input_tokens=response["input_tokens"], qa_output_tokens=response["output_tokens"],
                  qa_agreement=answer_scores(parsed["answer"], [answer])[1] if answer else 0.)
    checked = verify(parsed, evidence, verifier)
    result["nli_ms"] = checked.pop("verification_ms")
    result.update(checked)
    result["validation_ms"] = (time.perf_counter() - started) * 1000
    return result

def accept(checked, agreement=.75, support=0.):
    result = dict(checked)
    accepted = (result["valid"] and bool(result["answer"]) and result["qa_checked"] and bool(result["qa_answer"])
                and result["qa_agreement"] >= agreement and result["support"] >= support)
    result["accepted"] = bool(accepted)
    result["final_answer"] = result["answer"] if accepted else ""
    result["final_citations"] = result["citations"] if accepted else []
    if accepted:
        result["flag"] = "accepted" if result["support"] >= .5 else "low_model_support"
    elif not result["valid"] or not result["answer"] or result["reason"] == "qa_context_limit":
        result["flag"] = result["reason"]
    elif not result["qa_answer"]:
        result["flag"] = "qa_evidence_insufficient"
    elif result["qa_agreement"] < agreement:
        result["flag"] = "qa_disagreement"
    else:
        result["flag"] = "insufficient_support"
    return result
