from advanced_rag.upgrade.models import Generator

class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.options = kwargs
        return "CHAT"

def test_qwen_uses_chat_template_and_disables_thinking():
    generator = object.__new__(Generator)
    generator.chat = True
    generator.tokenizer = Tokenizer()
    assert generator.prompt("Question") == "CHAT"
    assert generator.tokenizer.options["enable_thinking"] is False
    assert generator.tokenizer.options["add_generation_prompt"] is True
    assert generator.tokenizer.messages == [{"role": "user", "content": "Question"}]

def test_flan_keeps_seq2seq_prompt():
    generator = object.__new__(Generator)
    generator.chat = False
    assert generator.prompt("Question") == "Question"
