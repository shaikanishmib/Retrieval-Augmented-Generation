"""Audit exact leakage and lexical near duplicates without changing partitions."""
from difflib import SequenceMatcher
import re
from rag_security.artifacts import ROOT, read, write, digest

tasks={s:read(ROOT/f'data/tasks/{s}.json') for s in ('development','evaluation')}
attacks=read(ROOT/'data/attacks.json')
claimed=[a for a in attacks['evaluation'] if a['source'].startswith('BIPIA') or a['family'] in ('dialogue','encoded')]
similarities=[]
for a in attacks['development']:
    if a['payload'] is None:
        continue
    for b in claimed:
        ratio=SequenceMatcher(None,a['payload'].casefold(),b['payload'].casefold()).ratio()
        similarities.append({'development':a['id'],'evaluation':b['id'],'character_sequence_similarity':ratio})
near=[p for p in similarities if p['character_sequence_similarity']>=.85]
assert not near,'Claimed family transfer includes a near-identical payload.'
def grams(text):
    words=re.findall(r'\w+',text.casefold())
    return set(zip(words,words[1:],words[2:]))
contexts=[]
for a in tasks['development']:
    if a['domain']!='bipia-table':
        continue
    ga=grams(a['documents'][0]['text'])
    for b in tasks['evaluation']:
        if b['domain']!='bipia-table':
            continue
        gb=grams(b['documents'][0]['text'])
        similarity=len(ga&gb)/len(ga|gb) if ga|gb else 1
        contexts.append({'development':a['id'],'evaluation':b['id'],'word_trigram_jaccard':similarity})
write(ROOT/'reports/split-audit.json',{'exact_context_overlap':len({digest(t['documents'][0]['text']) for t in tasks['development']}&{digest(t['documents'][0]['text']) for t in tasks['evaluation']}),'claimed_transfer_payload_pairs':len(similarities),'near_payload_threshold':.85,'near_payload_pairs':near,'maximum_payload_pair':max(similarities,key=lambda x:x['character_sequence_similarity']),'maximum_public_context_pair':max(contexts,key=lambda x:x['word_trigram_jaccard']),'limitations':'Lexical checks do not rule out semantic relationships or pretraining contamination. Recurring synthetic citation, poisoning and tool diagnostics are explicitly excluded from family-disjoint claims.'})
print('Claimed transfer payloads pass the lexical near-duplicate check.')
