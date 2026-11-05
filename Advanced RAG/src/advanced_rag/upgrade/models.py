"""Adapt sequence-to-sequence, Qwen chat, embedding, and NLI models."""
import gc
import time
import numpy as np
import torch
from .artifacts import ROOT, read

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def milliseconds(start):
    sync()
    return (time.perf_counter() - start) * 1000

def model_path(alias):
    original = {"flan-small": "generator", "minilm": "embedding", "tinybert": "reranker"}
    return ROOT / "data/models" / original[alias] if alias in original else ROOT / "data/upgrade/models" / alias

class Generator:
    def __init__(self, alias, device=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
        self.alias = alias
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chat = alias.startswith("qwen")
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path(alias), local_files_only=True)
        loader = AutoModelForCausalLM if self.chat else AutoModelForSeq2SeqLM
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = loader.from_pretrained(model_path(alias), local_files_only=True, torch_dtype=dtype).to(self.device).eval()
        self.load_ms = milliseconds(start)
        self.limit = 2048 if self.chat else 512

    def prompt(self, text):
        if self.chat:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False)
        return text

    def count(self, text):
        return len(self.tokenizer.encode(self.prompt(text), add_special_tokens=not self.chat))

    def generate(self, text, seed=42, limit=128):
        rendered = self.prompt(text)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=not self.chat).to(self.device)
        length = int(inputs.input_ids.shape[1])
        if length > self.limit:
            raise ValueError(f"Prompt exceeds {self.alias} input limit: {length}")
        torch.manual_seed(seed)
        settings = {"max_new_tokens": limit, "do_sample": self.chat}
        if self.chat:
            settings.update(temperature=.7, top_p=.8, top_k=20)
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **settings)
        elapsed = milliseconds(start)
        if self.chat:
            output = output[:, length:]
        text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return {"text": text, "input_tokens": length, "output_tokens": int(output.shape[1]), "generation_ms": elapsed,
                "truncated": bool(output.shape[1] >= limit)}

    def close(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class Encoder:
    def __init__(self, alias):
        from sentence_transformers import SentenceTransformer
        self.alias = alias
        started = time.perf_counter()
        self.model = SentenceTransformer(str(model_path(alias)), local_files_only=True, device="cuda" if torch.cuda.is_available() else "cpu")
        self.model.max_seq_length = 256
        self.load_ms = milliseconds(started)

    def encode(self, texts, query=False):
        if query and self.alias.startswith("bge"):
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        start = time.perf_counter()
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return vectors, milliseconds(start)

class Reranker:
    def __init__(self, alias):
        from sentence_transformers import CrossEncoder
        self.alias = alias
        start = time.perf_counter()
        self.model = CrossEncoder(str(model_path(alias)), local_files_only=True, max_length=384, device="cuda" if torch.cuda.is_available() else "cpu")
        self.load_ms = milliseconds(start)

    def predict(self, pairs):
        start = time.perf_counter()
        values = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        return np.asarray(values).reshape(-1).tolist(), milliseconds(start)

class Verifier:
    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        start = time.perf_counter()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = model_path("nli")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(self.device).eval()
        labels = self.model.config.id2label
        matches = [int(k) for k, v in labels.items() if v.casefold() == "entailment"]
        if len(matches) != 1:
            raise ValueError(f"NLI snapshot has no unambiguous entailment label: {labels}")
        self.entailment = matches[0]
        self.truncated_pairs = 0
        self.total_pairs = 0
        self.load_ms = milliseconds(start)

    def predict(self, pairs):
        if not pairs:
            return [], 0.
        start = time.perf_counter()
        values = []
        for offset in range(0, len(pairs), 16):
            batch = pairs[offset:offset + 16]
            self.total_pairs += len(batch)
            self.truncated_pairs += sum(len(self.tokenizer.encode(p, h)) > 512 for p, h in batch)
            inputs = self.tokenizer([p for p, _ in batch], [h for _, h in batch], padding=True, truncation="only_first", max_length=512, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                probabilities = self.model(**inputs).logits.softmax(dim=-1)[:, self.entailment]
            values.extend(probabilities.cpu().tolist())
        return values, milliseconds(start)
