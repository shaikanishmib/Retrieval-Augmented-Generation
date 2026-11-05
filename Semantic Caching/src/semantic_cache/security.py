"""Current authorization and corpus state define cache eligibility."""
from dataclasses import dataclass
import re
import unicodedata
from .artifacts import digest

def normalize(question):
    # Preserve case, punctuation, numbers and negation; collapse whitespace only.
    return ' '.join(unicodedata.normalize('NFC', question).split())

def lexical_guard(left, right):
    def features(text):
        tokens = re.findall(r"\b[\w'-]+\b", text.casefold())
        return (sorted(t for t in tokens if any(c.isdigit() for c in t)), sorted(t for t in tokens if t in {'no','not','never','without',"isn't","wasn't","didn't","don't"}))
    return features(left)==features(right)

@dataclass(frozen=True)
class Principal:
    tenant: str
    user: str
    allowed: frozenset
    epoch: int = 0
    verified: bool = True

@dataclass(frozen=True)
class Document:
    id: str
    text: str
    version: str = '1'
    tenant: str = 'public'
    visible: bool = True
    updated_at: str = 'frozen-release'

    def stamp(self):
        return digest(self.__dict__)

class Corpus:
    def __init__(self, documents):
        self.documents = {d.id:d for d in documents}

    def signature(self):
        return digest({k:v.stamp() for k,v in sorted(self.documents.items())})

    def authorized(self, principal, source_ids):
        return bool(principal.verified and principal.tenant and principal.user) and all(
            sid in principal.allowed and sid in self.documents and self.documents[sid].visible and
            self.documents[sid].tenant in {'public',principal.tenant} for sid in source_ids)

def scope(principal, corpus, model, prompt):
    if not principal.verified or not principal.tenant or not principal.user:
        raise PermissionError('Verified tenant and user permissions are required.')
    return digest([principal.tenant,principal.user,sorted(principal.allowed),principal.epoch,corpus.signature(),model,prompt])
