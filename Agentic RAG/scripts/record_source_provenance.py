"""Bridge executed workspace bytes and Git's LF-normalized publication bytes."""
import hashlib
from agentic_rag.artifacts import ROOT,sha,digest,fingerprint,write

if __name__=='__main__':
    paths=sorted((ROOT/'src').rglob('*.py'))+[ROOT/'configs/study.json',ROOT/'models.lock.json',ROOT/'sources.lock.json']
    records={}
    raw={}
    normalized={}
    for path in paths:
        name=str(path.relative_to(ROOT))
        payload=path.read_bytes()
        raw[name]=sha(path)
        normalized[name]=hashlib.sha256(payload.replace(b'\r\n',b'\n')).hexdigest()
        records[name]=dict(workspace_sha256=raw[name],git_lf_sha256=normalized[name],workspace_line_endings='CRLF' if b'\r\n' in payload else 'LF')
    assert digest(raw)==fingerprint()
    write(ROOT/'reports/source-provenance.json',dict(executed_source_identity=fingerprint(),lf_normalized_source_identity=digest(normalized),files=records,
          scope='Frozen inference/evaluation source, study configuration and source/model locks. Paths use the recorded Windows representation. Git normalizes text line endings only; executable contents and JSON values are preserved. Reporting scripts are versioned by the publication commit.'))
    print('Recorded execution and Git-normalized hashes for',len(records),'source inputs')
