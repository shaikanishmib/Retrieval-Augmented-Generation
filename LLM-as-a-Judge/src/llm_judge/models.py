"""Offline model adapters with measured inference and retained evidence windows."""
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from llm_judge.artifacts import ROOT, read
from llm_judge.claims import claims, select_windows
from llm_judge.prompts import SYSTEM, prompt, pairwise_prompt

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class Adapter:
    def __init__(self, key):
        self.key = key
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lock = read(ROOT / 'models.lock.json')[key]
        path = ROOT.parent / self.lock['local_path']
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        cls = AutoModelForSequenceClassification if key.startswith('nli') else AutoModel if key == 'bge-small' else AutoModelForCausalLM
        self.model = cls.from_pretrained(path, local_files_only=True, dtype=torch.float32 if key.startswith('nli') or key == 'bge-small' else torch.float16).to(self.device).eval()
        if key.startswith('qwen'):
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
        sync()
        self.load_seconds = time.perf_counter() - start
        self.labels = self.model.config.id2label
        torch.set_num_threads(8)

    def binary(self, text, choices=('Yes', 'No')):
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = self.tokenizer(rendered, return_tensors='pt').to(self.device)
        # Preserve instructions and complete answer; evidence is explicitly bounded upstream.
        ids = [self.tokenizer.encode(c, add_special_tokens=False) for c in choices]
        assert all(len(i) == 1 for i in ids), ids
        with torch.inference_mode():
            logits = self.model(**batch).logits[0, -1, [i[0] for i in ids]].float()
        return float(logits.softmax(-1)[0].cpu()), int(batch.input_ids.shape[1])

    def qwen(self, row, rubric):
        original_tokens = len(self.tokenizer.encode(row['context'], add_special_tokens=False))
        context_ids = self.tokenizer.encode(row['context'], add_special_tokens=False)
        bounded = {**row, 'context': self.tokenizer.decode(context_ids[:2400])}
        score, count = self.binary(prompt(bounded, rubric))
        if rubric == 'affirmative':
            score = 1 - score
        return {'score': score, 'input_tokens': count, 'context_tokens_original': original_tokens, 'context_truncated': original_tokens > 2400, 'evidence': bounded['context'], 'output_protocol': 'Conditional probability of single-token Yes versus No; no generated rationale or JSON.'}

    def nli(self, row, decompose=True, count=3):
        units = claims(row['response']) if decompose else [{'text': row['response'], 'start': 0, 'end': len(row['response'])}]
        pairs, metadata = [], []
        for unit in units:
            selected = select_windows(row['context'], unit['text'], count)
            for window in selected:
                pairs.append((window['text'], unit['text']))
            metadata.append((unit, selected))
        if not pairs:
            raise ValueError('No evaluable claims or evidence')
        probs, token_count = [], 0
        for start in range(0, len(pairs), 16):
            batch = self.tokenizer(pairs[start:start + 16], padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            token_count += int(batch.attention_mask.sum())
            with torch.inference_mode():
                probs.extend(self.model(**batch).logits.float().softmax(-1).cpu().tolist())
        labels = {str(v).lower(): int(k) for k, v in self.labels.items()}
        # The pinned cross-encoder card defines contradiction=0, entailment=1, neutral=2.
        entailment = labels.get('entailment', 1)
        contradiction = labels.get('contradiction', 0)
        details, offset = [], 0
        for unit, selected in metadata:
            ps = probs[offset:offset + len(selected)]
            best = max(range(len(ps)), key=lambda i: ps[i][entailment])
            details.append({**unit, 'unsupported': 1 - ps[best][entailment], 'contradiction': max(p[contradiction] for p in ps), 'selected_evidence': selected, 'probabilities': ps, 'best_window': best})
            offset += len(selected)
        return {'score': max(d['unsupported'] for d in details), 'mean_score': float(np.mean([d['unsupported'] for d in details])), 'input_tokens': token_count, 'claims': details}

    def embedding(self, row):
        texts = [row['response']] + [w['text'] for w in select_windows(row['context'], row['response'], 3)]
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.model(**batch).last_hidden_state[:, 0]
            vectors = torch.nn.functional.normalize(output.float(), dim=-1)
        return {'score': float((1 - (vectors[1:] @ vectors[0]).max()).cpu()), 'input_tokens': int(batch.attention_mask.sum())}

    def pairwise(self, row, a, b):
        context = self.tokenizer.decode(self.tokenizer.encode(row['context'])[:2400])
        return self.binary(pairwise_prompt(row['question'], context, a, b), ('A', 'B'))
