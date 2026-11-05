"""Parse explicit citations and score generated claims against cited evidence."""
import json
import re
from ..pipeline import normalize

def parse(text, evidence):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        result = json.loads(text)
    except (ValueError, TypeError):
        return {"answer": "", "claims": [], "citations": [], "valid": False, "reason": "invalid_json"}
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str) or not isinstance(result.get("claims"), list):
        return {"answer": "", "claims": [], "citations": [], "valid": False, "reason": "invalid_schema"}
    answer = result["answer"].strip()
    claims = result["claims"]
    if not answer and not claims:
        return {"answer": "", "claims": [], "citations": [], "valid": True, "reason": "model_abstention"}
    citations = []
    for claim in claims:
        if (not isinstance(claim, dict) or not isinstance(claim.get("text"), str) or not claim["text"].strip()
                or not isinstance(claim.get("sources"), list) or not claim["sources"]
                or not all(isinstance(s, str) for s in claim["sources"])):
            return {"answer": "", "claims": [], "citations": [], "valid": False, "reason": "missing_claim_citation"}
        for source in claim["sources"]:
            if source not in evidence:
                return {"answer": "", "claims": [], "citations": [], "valid": False, "reason": "fabricated_citation"}
            if source not in citations:
                citations.append(source)
    if not answer or not claims:
        return {"answer": "", "claims": [], "citations": [], "valid": False, "reason": "missing_answer_or_claim"}
    # This is an answer-claim consistency guard, not an entailment test.
    anchor = normalize(answer) in normalize(" ".join(c["text"] for c in claims))
    return {"answer": answer, "claims": claims, "citations": citations, "valid": anchor, "reason": "parsed" if anchor else "answer_not_in_claims"}

def verify(parsed, evidence, verifier):
    if not parsed["answer"] or not parsed["valid"]:
        return {"support": 0., "claim_scores": [], "citation_scores": [], "verification_ms": 0.}
    pairs, individual_counts = [], []
    for claim in parsed["claims"]:
        sources = list(dict.fromkeys(claim["sources"]))
        individual_counts.append(len(sources))
        pairs.append(("\n".join(evidence[s]["text"] for s in sources), claim["text"]))
        pairs.extend((evidence[s]["text"], claim["text"]) for s in sources)
    values, elapsed = verifier.predict(pairs)
    joint, individual, cursor = [], [], 0
    for count in individual_counts:
        joint.append(values[cursor])
        individual.extend(values[cursor + 1:cursor + 1 + count])
        cursor += count + 1
    return {"support": min(joint), "claim_scores": joint, "citation_scores": individual, "verification_ms": elapsed}

def gate(result, threshold):
    result = dict(result)
    reason = result["reason"]
    accepted = bool(result["answer"]) and result["valid"] and result["support"] >= threshold
    result["accepted"] = accepted
    result["final_answer"] = result["answer"] if accepted else ""
    result["final_citations"] = result["citations"] if accepted else []
    result["flag"] = ("accepted" if result["support"] >= .5 else "low_model_support") if accepted else (reason if not result["valid"] or not result["answer"] else "insufficient_support")
    return result
