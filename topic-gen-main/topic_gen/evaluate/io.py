import json
from typing import Dict
import ir_datasets
from topic_gen.evaluate import Experiment


def read_jsonl_topics(path: str) -> dict[str, dict[str, str]]:
    topics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            topic_id = str(item["topic_id"])
            topics[topic_id] = {k: v for k,
                                v in item.items() if k != "topic_id"}
    return topics


def read_irds_topics(irds: str) -> Dict[str, Dict[str, str]]:
    dataset = ir_datasets.load(irds)
    ret = {}

    # Get title field
    q = next(dataset.queries_iter())
    title_field = "title" if "title" in q._fields else "text"

    for topic in dataset.queries_iter():
        ret[topic.query_id] = {
            "title": getattr(topic, title_field),
            "description": getattr(topic, "description", ""),
            "narrative": getattr(topic, "narrative", "")
        }
    return ret


def load_from_irds(irds: str, binarize_qrels: int = 0) -> Experiment:
    from ir_measures.util import QrelsConverter
    topics = read_irds_topics(irds)
    qrels = ir_datasets.load(irds).qrels
    qrels = QrelsConverter(qrels).as_dict_of_dict()

    experiment = Experiment(topics=topics, qrels=qrels,
                            name=irds, binarize_qrels=binarize_qrels)
    return experiment
