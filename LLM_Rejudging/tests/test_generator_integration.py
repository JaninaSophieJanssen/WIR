"""
Integration tests — call a real LLM. Skipped unless --run-integration is passed.

Run with:
    uv run pytest -m integration
"""

import pytest

from topic_gen.generate import Generator
from topic_gen.models import TRECTopic, Topics


@pytest.mark.integration
class TestGeneratorIntegration:
    def test_generate_one_trec_topic(self, llm):
        gen = Generator(llm=llm, prompt="trec-base", output_class=Topics[TRECTopic])
        result = gen.generate_one(
            queries="What causes aurora borealis?",
            relevant_documents="Aurora borealis is caused by solar wind interacting with Earth's magnetic field.",
        )
        assert isinstance(result, Topics)
        assert len(result.topics) >= 1
        topic = result.topics[0]
        assert topic.title
        assert topic.description
        assert topic.narrative

    def test_generate_batch(self, llm):
        gen = Generator(llm=llm, prompt="trec-base", output_class=Topics[TRECTopic])
        results = gen.generate(
            queries=["aurora borealis causes", "northern lights location"],
            relevant_documents=["Solar wind doc.", "Magnetic field doc."],
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, Topics)
