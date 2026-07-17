from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from topic_gen import logger
from topic_gen.generate import Generator
from topic_gen.models import Topics, TRECTopic

load_dotenv()
logger.setLevel("INFO")


# Reference topic from TREC Robust 2004 (topic 650)
topic = {
    "title": "International Organized Crime",
    "description": "Identify organizations that participate in international criminal activity, the activity, and, if possible, collaborating organizations and the countries involved.",
    "narrative": "A relevant document must as a minimum identify the organization and the type of illegal activity (e.g., Columbian cartel exporting cocaine). Vague references to international drug trade without identification of the organization(s) involved would not be relevant.",
    "queries": [
        "organizations international criminal activity",
        "international crime groups",
        "organized crime mafia",
    ],
    "rel_docs": [
        "here goes a relevant document body",
    ],
}


llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0,
)
generator = Generator(llm=llm, prompt="trec-base", output_class=Topics[TRECTopic])


generated_topics = generator.generate_one(
    queries="\n".join(topic["queries"]),
    relevant_documents="\n\n".join(topic["rel_docs"]),
)


generated_topic = generated_topics.topics[0]

print("Title:")
print("  Reference:", topic["title"])
print("  Generated:", generated_topic.title)

print("\nDescription:")
print("  Reference:", topic["description"])
print("  Generated:", generated_topic.description)

print("\nNarrative:")
print("  Reference:", topic["narrative"])
print("  Generated:", generated_topic.narrative)


generated_topics.to_xml("topics.xml")
