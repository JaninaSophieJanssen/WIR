import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from topic_gen.generate import PROMPTS, Generator
from topic_gen.models import TRECTopic, Topics

DATA = Path(__file__).parent / "data"


def make_fake_response(topics: list[dict]) -> str:
    return json.dumps({"topics": topics})


VALID_TOPICS_RESPONSE = make_fake_response(
    [{"title": "Test Title", "description": "Test description.", "narrative": "Test narrative."}]
)

VALID_TOPIC_RESPONSE = json.dumps(
    {"title": "Test Title", "description": "Test description.", "narrative": "Test narrative."}
)

TREC_PROMPT = "trec-base"


def fake_llm(responses: list[str]) -> FakeListChatModel:
    return FakeListChatModel(responses=responses)


@pytest.fixture
def minimal_prompt(tmp_path):
    """A prompt without {format_instructions}, for testing parse=False paths."""
    f = tmp_path / "minimal.yaml"
    f.write_text("input_variables:\n- topic\ntemplate: 'Generate about {topic}'\n")
    return str(f)


class TestLoadPrompt:
    def test_builtin_prompt_loads(self):
        llm = fake_llm([VALID_TOPICS_RESPONSE])
        gen = Generator(llm=llm, prompt=TREC_PROMPT, output_class=Topics[TRECTopic])
        assert gen.prompt_template is not None

    def test_unknown_prompt_raises(self):
        llm = fake_llm([VALID_TOPICS_RESPONSE])
        with pytest.raises(FileNotFoundError):
            Generator(llm=llm, prompt="nonexistent-prompt", output_class=Topics[TRECTopic])

    def test_custom_prompt_path(self, minimal_prompt):
        llm = fake_llm(["raw output"])
        gen = Generator(llm=llm, prompt=minimal_prompt, parse=False)
        assert gen.prompt_template is not None

    def test_all_builtin_prompts_listed(self):
        assert TREC_PROMPT in PROMPTS


class TestGenerateOne:
    def test_returns_parsed_model(self):
        llm = fake_llm([VALID_TOPICS_RESPONSE])
        gen = Generator(llm=llm, prompt=TREC_PROMPT, output_class=Topics[TRECTopic])
        result = gen.generate_one(queries="test query", relevant_documents="doc snippet")
        assert isinstance(result, Topics)
        assert result.topics[0].title == "Test Title"

    def test_returns_none_on_dry_run(self):
        llm = fake_llm([VALID_TOPICS_RESPONSE])
        gen = Generator(llm=llm, prompt=TREC_PROMPT, output_class=Topics[TRECTopic])
        result = gen.generate_one(dry_run=True, queries="q", relevant_documents="d")
        assert result is None

    def test_returns_none_on_invalid_output(self):
        llm = fake_llm(["this is not valid json {{{"])
        gen = Generator(llm=llm, prompt=TREC_PROMPT, output_class=Topics[TRECTopic])
        result = gen.generate_one(queries="q", relevant_documents="d")
        assert result is None

    def test_attaches_item_id(self):
        # item_id attachment works on direct topic classes (not Topics[T] containers)
        llm = fake_llm([VALID_TOPIC_RESPONSE])
        gen = Generator(llm=llm, prompt=TREC_PROMPT, output_class=TRECTopic)
        result = gen.generate_one(item_id="topic-42", queries="q", relevant_documents="d")
        assert result.topic_id == "topic-42"

    def test_no_output_class_returns_string(self, minimal_prompt):
        llm = fake_llm(["some raw text"])
        gen = Generator(llm=llm, prompt=minimal_prompt, parse=False)
        result = gen.generate_one(topic="aurora borealis")
        assert isinstance(result, str)


class TestGenerateBatch:
    def test_rejects_non_list_kwargs(self):
        llm = fake_llm([VALID_TOPICS_RESPONSE])
        gen = Generator(llm=llm, prompt=TREC_PROMPT, output_class=Topics[TRECTopic])
        with pytest.raises(ValueError, match="Expected a list"):
            gen.generate(queries="not a list", relevant_documents=["d1"])

    def test_dry_run_returns_none(self):
        llm = fake_llm([VALID_TOPICS_RESPONSE, VALID_TOPICS_RESPONSE])
        gen = Generator(llm=llm, prompt=TREC_PROMPT, output_class=Topics[TRECTopic])
        result = gen.generate(
            dry_run=True,
            queries=["q1", "q2"],
            relevant_documents=["d1", "d2"],
        )
        assert result is None

    def test_batch_zips_kwargs(self):
        llm = fake_llm([VALID_TOPICS_RESPONSE, VALID_TOPICS_RESPONSE])
        gen = Generator(llm=llm, prompt=TREC_PROMPT, output_class=Topics[TRECTopic])
        results = gen.generate(
            queries=["q1", "q2"],
            relevant_documents=["d1", "d2"],
        )
        assert len(results) == 2

    def test_attaches_item_ids(self):
        # item_id attachment works on direct topic classes (not Topics[T] containers)
        llm = fake_llm([VALID_TOPIC_RESPONSE, VALID_TOPIC_RESPONSE])
        gen = Generator(llm=llm, prompt=TREC_PROMPT, output_class=TRECTopic)
        results = gen.generate(
            item_ids=["id-1", "id-2"],
            queries=["q1", "q2"],
            relevant_documents=["d1", "d2"],
        )
        assert results[0].topic_id == "id-1"
        assert results[1].topic_id == "id-2"


class TestStructuredOutput:
    def test_uses_structured_output_when_supported(self):
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        
        gen = Generator(
            llm=mock_llm, 
            prompt=TREC_PROMPT, 
            output_class=Topics[TRECTopic], 
            use_structured_output=True
        )
        
        mock_llm.with_structured_output.assert_called_once_with(Topics[TRECTopic])
        assert gen.use_structured_output is True
        assert gen.parser is None

    def test_falls_back_when_not_supported(self):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = NotImplementedError("Not implemented")
        
        gen = Generator(
            llm=mock_llm, 
            prompt=TREC_PROMPT, 
            output_class=Topics[TRECTopic], 
            use_structured_output=True
        )
        
        mock_llm.with_structured_output.assert_called_once()
        assert gen.use_structured_output is False
        assert gen.parser is not None
