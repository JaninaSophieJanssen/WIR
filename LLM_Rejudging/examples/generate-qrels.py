import ir_datasets
from dotenv import load_dotenv
from ir_measures import Qrel
from langchain.chat_models import init_chat_model

from topic_gen import logger
from topic_gen.generate import Generator
from topic_gen.models import MTO_responds

load_dotenv()
logger.setLevel("INFO")


dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
store = dataset.docs_store()

queries_map = {q.query_id: q for q in dataset.queries_iter()}
qrels_map = {}
for q in dataset.qrels_iter():
    qrels_map.setdefault(q.query_id, []).append(q)


llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0,
)
generator = Generator(llm=llm, prompt="-DNA-zero-shot", output_class=MTO_responds)


query = queries_map["652"]

my_qrels = []
for qrel in qrels_map[query.query_id][:10]:
    document = store.get(qrel.doc_id).default_text()
    res = generator.generate_one(
        document=document,
        query=query.title,
        description=query.description,
        narrative=query.narrative,
    )
    my_qrels.append(Qrel(qrel.query_id, qrel.doc_id, res.O))


my_qrels
