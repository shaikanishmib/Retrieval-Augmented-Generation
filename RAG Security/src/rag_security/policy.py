"""Application policy uses registered state, never passage-supplied permissions."""
from dataclasses import dataclass, asdict, replace
import json
from .artifacts import digest

@dataclass(frozen=True)
class Document:
    id: str
    text: str
    tenant: str = 'public'
    version: int = 1
    title: str = 'Evidence'
    visible: bool = True
    start: int = 0

    @property
    def stamp(self):
        return digest(asdict(self))

@dataclass(frozen=True)
class Principal:
    tenant: str
    user: str
    allowed: frozenset
    epoch: int = 0
    verified: bool = True

class Registry:
    def __init__(self, documents):
        if len({d.id for d in documents}) != len(documents):
            raise ValueError('Duplicate document IDs.')
        self.documents = {d.id: d for d in documents}
        self.revoked = set()

    @property
    def snapshot(self):
        return digest([sorted((k, v.stamp) for k, v in self.documents.items()), sorted(self.revoked)])

    def authorized(self, principal, sid):
        d = self.documents.get(sid)
        return bool(principal.verified and principal.tenant and principal.user and d and d.visible and sid not in self.revoked and sid in principal.allowed and d.tenant in ('public', principal.tenant))

    def intact(self, doc):
        return doc.id in self.documents and self.documents[doc.id].stamp == doc.stamp

    def ingest(self, doc, approved=False):
        # The queue has no privilege to register or replace authoritative documents.
        if not approved or doc.id in self.documents:
            return False
        self.documents[doc.id] = doc
        return True

class AnswerCache:
    def __init__(self):
        self.entries = {}

    def scope(self, p, registry, question, model):
        return digest([p.tenant, p.user, sorted(p.allowed), p.epoch, registry.snapshot, question, model])

    def put(self, p, registry, question, model, answer, sources, approved=False):
        if not approved or not sources or not all(registry.authorized(p, s) for s in sources):
            return False
        self.entries[self.scope(p, registry, question, model)] = {'answer': answer, 'sources': {s: registry.documents[s].stamp for s in sources}}
        return True

    def get(self, p, registry, question, model):
        entry = self.entries.get(self.scope(p, registry, question, model))
        if entry and all(registry.authorized(p, s) and registry.documents[s].stamp == stamp for s, stamp in entry['sources'].items()):
            return entry['answer']
        return None

def parse_action(raw):
    try:
        action = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(action, dict) or set(action) != {'tool', 'document_id'}:
        return None
    if not all(isinstance(v, str) and len(v) <= 100 for v in action.values()):
        return None
    return action

class ReadOnlyTools:
    def __init__(self, registry, principal, budget=2):
        if not 0 <= budget <= 2:
            raise ValueError('Tool budget must be between zero and two.')
        self.registry, self.principal, self.budget = registry, principal, budget
        self.calls = 0
        self.log = []

    def dispatch(self, raw):
        action = parse_action(raw)
        self.calls += 1
        allowed = bool(self.calls <= self.budget and action and action['tool'] == 'lookup' and self.registry.authorized(self.principal, action['document_id']))
        self.log.append({'raw': raw, 'allowed': allowed})
        return self.registry.documents[action['document_id']].text if allowed else None

def output_valid(answer, sources, registry, principal, canaries=()):
    import re
    citations = re.findall(r'\[([^\[\]]+)\]', answer)
    valid = all(s in sources and registry.authorized(principal, s) for s in citations)
    leaked = any(c in answer for c in canaries)
    return valid and not leaked, {'citation_count': len(citations), 'invalid_citations': sum(s not in sources for s in citations), 'canary_exposed': leaked}
