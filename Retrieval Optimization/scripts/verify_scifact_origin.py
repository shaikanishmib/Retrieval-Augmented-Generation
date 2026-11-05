"""Verify BEIR SciFact against the locally pinned original release."""

import csv
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / "data/raw"
    corpus = {row["_id"]: row for row in map(json.loads, (raw / "scifact/corpus.jsonl").read_text(encoding="utf-8").splitlines())}
    queries = {row["_id"]: row["text"] for row in map(json.loads, (raw / "scifact/queries.jsonl").read_text(encoding="utf-8").splitlines())}
    with tarfile.open(raw / "scifact-original.tar.gz") as archive:
        original = [json.loads(line) for line in archive.extractfile("data/corpus.jsonl")]
        result = {
            "same_corpus_ids": set(corpus) == {str(row["doc_id"]) for row in original},
            "titles_match": all(corpus[str(row["doc_id"])]["title"] == row["title"] for row in original),
            "abstracts_match_after_whitespace_normalization": all(
                " ".join(corpus[str(row["doc_id"])]["text"].split()) == " ".join(" ".join(row["abstract"]).split()) for row in original),
        }
        for split, original_split in [("train", "train"), ("test", "dev")]:
            rows = [json.loads(line) for line in archive.extractfile(f"data/claims_{original_split}.jsonl")]
            with (raw / "scifact/qrels" / f"{split}.tsv").open() as stream:
                actual = {(row["query-id"], row["corpus-id"]) for row in csv.DictReader(stream, delimiter="\t")}
            cited = {(str(row["id"]), str(did)) for row in rows for did in row["cited_doc_ids"]}
            result[split] = {"original_split": original_split, "original_claims": len(rows),
                             "query_texts_match": all(queries[str(row["id"])] == row["claim"] for row in rows),
                             "qrels_equal_cited_document_pairs": cited == actual, "pairs": len(cited)}
        assert all(result[k] for k in ("same_corpus_ids", "titles_match", "abstracts_match_after_whitespace_normalization"))
        assert all(result[s]["query_texts_match"] and result[s]["qrels_equal_cited_document_pairs"] for s in ("train", "test"))
    (ROOT / "reports/scifact-original-mapping.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
