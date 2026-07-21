# Generating Retrieval Topics and Judgments

A small python toolkit to efficiently generate artifacts such as retrieval topics and relevance labels.



## Installation
```
pip install git+https://github.com/irgroup/topic-gen.git
```

## Usage
Also see the examples for generating [multiple topics](https://github.com/irgroup/topic-gen/blob/main/examples/generate-topics.py), [single topics](https://github.com/irgroup/topic-gen/blob/main/examples/generate-topic-one.py), and [qrels](https://github.com/irgroup/topic-gen/blob/main/examples/generate-qrels.py).

### Generate a single TREC topic

```python
from langchain.chat_models import init_chat_model
from topic_gen.generate import Generator
from topic_gen.models import TRECTopic, Topics

llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0)

generator = Generator(llm=llm, prompt="trec-base", output_class=Topics[TRECTopic])

topics = generator.generate_one(
    queries="What causes aurora borealis?",
    relevant_documents="Aurora borealis is caused by solar wind interacting with Earth's magnetic field.",
)

for topic in topics.topics:
    print(topic.title)
    print(topic.description)
    print(topic.narrative)
```

### Generate a batch of TREC topics

```python
topics_list = generator.generate(
    queries=["aurora borealis causes", "climate change effects"],
    relevant_documents=["Solar wind doc.", "IPCC report excerpt."],
)
```

### Export topics

```python
topics.to_xml("topics.xml")
topics.to_jsonl("topics.jsonl")
topics.to_csv("topics.csv")
```

### Generate relevance judgments (qrels)

The `-DNA-zero-shot` prompt scores a document against a query using the MTO schema (intent match, trustworthiness, overall score) from Thomas et al. (2024).

```python
from langchain.chat_models import init_chat_model
from topic_gen.generate import Generator
from topic_gen.models import MTO_responds, Qrel, Qrels

llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0)

generator = Generator(
    llm=llm,
    prompt="-DNA-zero-shot",
    output_class=MTO_responds,
    config={"max_concurrency": 8},
)

results = generator.generate(
    query=queries,           # list of query strings
    description=descriptions,
    narrative=narratives,
    document=documents,      # list of document texts to judge
)

qrels = Qrels[Qrel](qrels=[
    Qrel(query_id=qid, doc_id=did, relevance=res.O)
    for qid, did, res in zip(query_ids, doc_ids, results)
    if isinstance(res, MTO_responds)
])

qrels.to_trec("qrels.txt")
```

## Development

Clone the repo and install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/irgroup/topic-gen.git
cd topic-gen
uv sync --extra evaluate
```

### Running tests

```bash
uv run pytest                  # unit tests only (default)
uv run pytest --integration    # include integration tests against a real LLM
```

Integration tests require a valid API key. By default they use Gemini 2.5 Flash via `GOOGLE_API_KEY`. To use a different model or a local vLLM instance, set the following variables in your `.env` file:

```
# vLLM (OpenAI-compatible)
LLM_PROVIDER=openai
LLM_MODEL=<model-name>
LLM_BASE_URL=http://<host>:<port>/v1
OPENAI_API_KEY=dummy

# Gemini (default)
GOOGLE_API_KEY=<your-key>
```

### Linting and formatting

```bash
uv run ruff check topic_gen/   # lint
uv run ruff format topic_gen/  # format
```

### Type checking

```bash
uv run mypy topic_gen/
```

