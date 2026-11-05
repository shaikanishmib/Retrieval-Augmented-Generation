import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_records(path: Path, *, corpus: bool, allow_empty: bool = False) -> tuple[dict, dict]:
    records = {}
    empty = Counter()
    texts = Counter()
    content = Counter()
    nulls = Counter()
    empty_metadata = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Record is not an object at {path}:{line_number}")
            nulls.update(key for key, value in row.items() if value is None)
            empty_metadata += row.get("metadata") == {}
            identifier = row.get("_id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"Invalid ID at {path}:{line_number}")
            if identifier in records:
                raise ValueError(f"Duplicate ID at {path}:{line_number}")
            for field in (["text", "title"] if corpus else ["text"]):
                if not isinstance(row.get(field), str):
                    raise ValueError(f"Missing or non-string {field} at {path}:{line_number}")
                if not row[field].strip():
                    empty[field] += 1
            if not row["text"].strip() and not allow_empty:
                raise ValueError(f"Empty text at {path}:{line_number}")
            texts[row["text"]] += 1
            content[(row.get("title", ""), row["text"])] += 1
            records[identifier] = row
    if not records:
        raise ValueError(f"No records in {path}")
    return records, {
        "count": len(records), "empty_fields": dict(empty),
        "duplicate_text_excess": sum(n - 1 for n in texts.values()),
        "duplicate_content_excess": sum(n - 1 for n in content.values()),
        "duplicate_ids": 0, "missing_required_fields": 0,
        "null_fields": dict(nulls), "empty_metadata_objects": empty_metadata,
    }


def validate(root: Path, *, allow_empty_corpus: bool = False) -> dict:
    corpus, corpus_stats = load_records(root / "corpus.jsonl", corpus=True, allow_empty=allow_empty_corpus)
    queries, query_stats = load_records(root / "queries.jsonl", corpus=False)
    splits = {}
    errors = []
    memberships = {}
    paths = sorted((root / "qrels").glob("*.tsv"))
    if not paths:
        raise ValueError("No qrels splits")
    for path in paths:
        pairs = set()
        labels = Counter()
        judged = set()
        positives = set()
        missing_queries = 0
        missing_documents = 0
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames != ["query-id", "corpus-id", "score"]:
                raise ValueError(f"Invalid qrels header: {path}")
            for row in reader:
                qid, did = row["query-id"], row["corpus-id"]
                score = int(row["score"])
                if score < 0:
                    raise ValueError(f"Negative qrels score: {path}")
                missing_queries += qid not in queries
                missing_documents += did not in corpus
                if (qid, did) in pairs:
                    raise ValueError(f"Duplicate qrels pair: {path}")
                pairs.add((qid, did))
                labels[str(score)] += 1
                judged.add(qid)
                if score > 0:
                    positives.add(qid)
        if not pairs or judged != positives:
            raise ValueError(f"Empty qrels or query without positives: {path}")
        memberships[path.stem] = judged
        splits[path.stem] = {"rows": len(pairs), "queries": len(judged), "labels": dict(labels),
                             "missing_query_references": missing_queries,
                             "missing_document_references": missing_documents}
        if missing_queries or missing_documents:
            errors.append(f"{path.stem}: {missing_queries} missing query and {missing_documents} missing document references")
    union = set().union(*memberships.values())
    overlap = {f"{a}/{b}": len(memberships[a] & memberships[b])
               for a in memberships for b in memberships if a < b}
    errors.extend(f"Query IDs overlap between splits {pair}: {count}" for pair,count in overlap.items() if count)
    return {
        "valid": not errors, "errors": errors,
        "corpus": corpus_stats, "queries": query_stats, "splits": splits,
        "unjudged_queries": len(queries.keys() - union),
        "split_query_overlap": overlap,
        "query_corpus_id_overlap": sum(q in corpus for q in queries),
        "files": {str(p.relative_to(root)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": digest(p)}
                  for p in [root / "corpus.jsonl", root / "queries.jsonl", *paths]},
    }
