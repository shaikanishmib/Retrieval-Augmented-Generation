import pytest
from semantic_cache.models import pack_prompt

class WordTokenizer:
    def encode(self,text,add_special_tokens=True):
        return text.split()+(['<eos>'] if add_special_tokens else [])
    def decode(self,tokens,skip_special_tokens=True):
        return ' '.join(tokens)

def test_question_and_answer_suffix_survive_long_evidence():
    tokenizer = WordTokenizer()
    question = 'Which city hosted the games in 2020?'
    prompt = pack_prompt(tokenizer,question,['Evidence '*1000],80)
    assert question in prompt
    assert prompt.endswith('\nAnswer:')
    assert len(tokenizer.encode(prompt))<=80
    assert len(prompt.split())>50

def test_oversize_question_fails_explicitly():
    with pytest.raises(ValueError,match='input budget'):
        pack_prompt(WordTokenizer(),'Question '*1000,['evidence'],80)
