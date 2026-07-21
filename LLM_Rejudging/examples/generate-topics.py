from langchain.chat_models import init_chat_model

from topic_gen.generate import Generator
from topic_gen.models import TRECTopic
from dotenv import load_dotenv
from topic_gen import logger

load_dotenv()


logger.setLevel("DEBUG")

def main():
    llm = init_chat_model(
        model="gemini-2.5-flash", model_provider = "google_genai", temperature=0
    )

    components = {
        "queries": ["I need milk", "Where is Paris", "what are the odds?"],
        "relevant_documents": [
            "Get milk free and fast",
            "Paris is a city in France",
            "The odds are high",
        ],
        "query_ids": [1, 2, 3],
    }

    generator = Generator(
        llm=llm,
        output_class=TRECTopic,
        prompt="trec-base",
        config={"max_concurrency": 2},
    )

    generated_topics = generator.generate(
        item_ids=components["query_ids"], **components
    )

    print(generated_topics)


if __name__ == "__main__":
    main()
