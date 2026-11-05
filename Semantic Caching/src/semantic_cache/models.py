"""Offline model adapters with synchronized inference timings."""
import gc
import time
import numpy as np
import torch
from .artifacts import ROOT,read

torch.set_num_threads(4)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def elapsed(start):
    sync()
    return (time.perf_counter()-start)*1000

def location(alias):
    record = read(ROOT/'models.lock.json')[alias]
    return ROOT.parent/record['local_path']

def pack_prompt(tokenizer, question, passages, limit):
    prefix = 'Answer the question using only the evidence. If the evidence does not answer it, say unanswerable. Give only the short answer.\nQuestion: '+question+'\nEvidence:\n'
    suffix = '\nAnswer:'
    reserved = len(tokenizer.encode(prefix+suffix))
    if reserved>=limit:
        raise ValueError('The question and instructions exceed the input budget.')
    evidence = tokenizer.encode('\n'.join(passages),add_special_tokens=False)[:max(0,limit-reserved-4)]
    while True:
        prompt = prefix+tokenizer.decode(evidence,skip_special_tokens=True)+suffix
        if len(tokenizer.encode(prompt))<=limit:
            return prompt
        if not evidence:
            raise ValueError('The question cannot fit within the input budget.')
        evidence.pop()

class Encoder:
    def __init__(self, alias):
        from sentence_transformers import SentenceTransformer
        self.alias = alias
        self.identity = alias+':'+read(ROOT/'models.lock.json')[alias]['revision']
        start = time.perf_counter()
        self.model = SentenceTransformer(str(location(alias)),local_files_only=True,device='cuda' if torch.cuda.is_available() else 'cpu')
        self.model.max_seq_length = 256
        self.load_ms = elapsed(start)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        sync()
        start = time.perf_counter()
        vectors = self.model.encode(texts,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False,batch_size=32)
        return vectors,elapsed(start)

    def close(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class Generator:
    def __init__(self, alias):
        from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
        self.alias = alias
        self.identity = alias+':'+read(ROOT/'models.lock.json')[alias]['revision']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(location(alias),local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(location(alias),local_files_only=True,torch_dtype=torch.float16 if self.device=='cuda' else torch.float32).to(self.device).eval()
        self.load_ms = elapsed(start)

    def generate(self, question, passages, max_input=480, max_new=48):
        prompt = pack_prompt(self.tokenizer,question,passages,max_input)
        tokens = self.tokenizer(prompt,return_tensors='pt',truncation=False).to(self.device)
        assert tokens.input_ids.shape[1]<=max_input
        sync()
        start = time.perf_counter()
        first = last = None
        if self.device=='cuda':
            first,last = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            first.record()
        with torch.inference_mode():
            out = self.model.generate(**tokens,max_new_tokens=max_new,do_sample=False)
        if last:
            last.record()
        ms = elapsed(start)
        answer = self.tokenizer.decode(out[0],skip_special_tokens=True).strip()
        if answer.casefold() in {'unanswerable','unanswerable.','not enough information','unknown'}:
            answer = ''
        return dict(answer=answer,generation_ms=ms,gpu_generation_ms=first.elapsed_time(last) if first else 0.,input_tokens=int(tokens.input_ids.shape[1]),output_tokens=int(out.shape[1]),prompt=prompt)

    def close(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
