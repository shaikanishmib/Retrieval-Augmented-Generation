"""Local inference adapters; never load remote code or execute retrieved instructions."""
import gc
import time
import numpy as np
import torch
from .artifacts import ROOT, read

torch.set_num_threads(4)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def elapsed(start):
    sync()
    return (time.perf_counter()-start)*1000

def path(alias):
    return ROOT.parent / read(ROOT / "models.lock.json")[alias]["local_path"]

class Generator:
    def __init__(self, alias):
        from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSeq2SeqLM
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chat = alias.startswith("qwen")
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(path(alias),local_files_only=True)
        cls = AutoModelForCausalLM if self.chat else AutoModelForSeq2SeqLM
        self.model = cls.from_pretrained(path(alias),local_files_only=True,torch_dtype=torch.float16 if self.device=="cuda" else torch.float32).to(self.device).eval()
        self.load_ms = elapsed(start)
        self.limit = 2048 if self.chat else 512

    def render(self,text):
        return self.tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True,enable_thinking=False) if self.chat else text

    def count(self,text):
        return len(self.tokenizer.encode(self.render(text),add_special_tokens=not self.chat))

    def generate(self,text,limit=48):
        inputs = self.tokenizer(self.render(text),return_tensors="pt",add_special_tokens=not self.chat).to(self.device)
        n = inputs.input_ids.shape[1]
        if n>self.limit:
            raise ValueError(f"Prompt too long: {n}>{self.limit}")
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs,max_new_tokens=limit,do_sample=False)
        ms = elapsed(start)
        out = out[:,n:] if self.chat else out
        return dict(text=self.tokenizer.decode(out[0],skip_special_tokens=True).strip(),input_tokens=n,output_tokens=int(out.shape[1]),ms=ms,hit_limit=out.shape[1]>=limit)

    def close(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class Services:
    def __init__(self, encoder="bge-small", reranker="minilm-rerank"):
        from sentence_transformers import SentenceTransformer,CrossEncoder
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder_alias = encoder
        start = time.perf_counter()
        self.encoder = SentenceTransformer(str(path(encoder)),local_files_only=True,device=self.device)
        self.encoder.max_seq_length = 256
        self.encoder_ms = elapsed(start)
        start = time.perf_counter()
        self.reranker = CrossEncoder(str(path(reranker)),local_files_only=True,max_length=384,device=self.device)
        self.reranker_ms = elapsed(start)
        start = time.perf_counter()
        self.nli_tokenizer = AutoTokenizer.from_pretrained(path("nli"),local_files_only=True)
        self.nli = AutoModelForSequenceClassification.from_pretrained(path("nli"),local_files_only=True).to(self.device).eval()
        self.labels = self.nli.config.id2label
        self.entailment = next(int(k) for k,v in self.labels.items() if v.lower()=="entailment")
        self.contradiction = next(int(k) for k,v in self.labels.items() if v.lower()=="contradiction")
        self.nli_ms = elapsed(start)

    def encode(self,texts,query=False):
        if query:
            texts = ["Represent this sentence for searching relevant passages: "+s for s in texts]
        return self.encoder.encode(texts,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False,batch_size=32)

    def rerank(self,question,texts):
        return np.asarray(self.reranker.predict([(question,t) for t in texts],show_progress_bar=False)).reshape(-1)

    def verify(self,premise,claim):
        count = len(self.nli_tokenizer.encode(premise,claim))
        tokens = self.nli_tokenizer(premise,claim,return_tensors="pt",truncation="only_first",max_length=512).to(self.device)
        with torch.inference_mode():
            probs = self.nli(**tokens).logits.softmax(-1)[0].cpu().tolist()
        return dict(entailment=probs[self.entailment],contradiction=probs[self.contradiction],truncated=count>512)
