"""Split text while retaining original document identifiers."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    tokens: int


def chunks(documents, method="whole", size=128, overlap=24):
    if method not in {"whole", "fixed", "recursive", "sentence"}:
        raise ValueError(f"Unknown chunking method: {method}")
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    result = []
    for did, text in sorted(documents.items()):
        words = text.split()
        if method == "whole":
            segments = [text]
        elif method == "fixed":
            segments = [" ".join(words[i:i + size]) for i in range(0, len(words), size - overlap)]
        else:
            pattern = r"\n\s*\n|(?<=[.!?])\s+" if method == "recursive" else r"(?<=[.!?])\s+"
            units = re.split(pattern, text)
            segments, pending = [], []
            for unit in units:
                tokens = unit.split()
                while tokens:
                    capacity = size - len(pending)
                    pending.extend(tokens[:capacity])
                    tokens = tokens[capacity:]
                    if len(pending) == size or (tokens == [] and len(pending) >= size * .75):
                        segments.append(" ".join(pending))
                        pending = pending[-overlap:] if overlap else []
            if pending and (not segments or " ".join(pending) != segments[-1]):
                segments.append(" ".join(pending))
        for i, segment in enumerate(segments or [""]):
            result.append(Chunk(f"{did}::{i}", did, segment, len(segment.split())))
    return result


def aggregate(ranking, mapping, k=20):
    if k < 1:
        raise ValueError("k must be positive")
    seen, result = set(), []
    for identifier, score in ranking:
        did = mapping[identifier]
        if did not in seen:
            seen.add(did)
            result.append((did, float(score)))
            if len(result) == k:
                break
    return result
