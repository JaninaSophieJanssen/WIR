import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests against a real LLM (requires API key)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="pass --integration to run against a real LLM")
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def llm():
    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
    provider = os.environ.get("LLM_PROVIDER", "google_genai")
    base_url = os.environ.get("LLM_BASE_URL")
    kwargs = {"model": model, "model_provider": provider, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return init_chat_model(**kwargs)
