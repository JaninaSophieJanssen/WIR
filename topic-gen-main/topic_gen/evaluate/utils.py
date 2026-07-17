import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from ir_measures.util import TYPE_QREL, QrelsConverter
from topic_gen import logger


class QrelsTransformer:
    @staticmethod
    def count_by_relevance(experiments: List[Any]) -> List[dict[int, int]]:
        labels = set()
        for exp in experiments:
            for topic, qrels in exp.qrels.items():
                for doc_id, relevance in qrels.items():
                    labels.add(relevance)

        item_relevance_counts = defaultdict(lambda: {label: 0 for label in labels})

        for exp in experiments:
            for topic, qrels in exp.qrels.items():
                for doc_id, relevance in qrels.items():
                    item_relevance_counts[f"{topic}, {doc_id}"][relevance] += 1

        count_list = []
        for counts in item_relevance_counts.values():
            count_list.append(list(counts.values()))

        return count_list

    @staticmethod
    def flatten_relevance(dict_of_dict: dict[str, dict[str, int]]) -> List[int]:
        ret = []
        for topic, qrels in dict_of_dict.items():
            for doc_id, relevance in qrels.items():
                ret.append(relevance)
        return ret

    @staticmethod
    def binarize_relevance(
        qrels: TYPE_QREL, threshold: int = 1
    ) -> Dict[str, Dict[str, int]]:
        qrels = QrelsConverter(qrels).as_dict_of_dict()
        ret = defaultdict(dict)
        for topic, qrels in qrels.items():
            for doc_id, relevance in qrels.items():
                if relevance >= threshold:
                    ret[topic][doc_id] = 1
                else:
                    ret[topic][doc_id] = 0
        return ret

    @staticmethod
    def replace_relevance(
        qrels: TYPE_QREL, mapping: dict[int, int]
    ) -> Dict[str, Dict[str, int]]:
        qrels = QrelsConverter(qrels).as_dict_of_dict()
        c = 0
        ret = defaultdict(dict)
        for topic, qrels in qrels.items():
            for doc_id, relevance in qrels.items():
                if relevance in mapping:
                    c += 1
                ret[topic][doc_id] = mapping.get(relevance, relevance)
        return ret, c

    @staticmethod
    def drop_relevance(qrels: TYPE_QREL, drop_values: int) -> Dict[str, Dict[str, int]]:
        qrels = QrelsConverter(qrels).as_dict_of_dict()
        ret = defaultdict(dict)
        c = 0
        for topic, qrels in qrels.items():
            for doc_id, relevance in qrels.items():
                if relevance != drop_values:
                    c += 1
                    ret[topic][doc_id] = relevance
        return ret, c

    @staticmethod
    def align_qrels_inner(
        exp: Any, baseline: Any
    ) -> Union[List[str], List[str], int, int]:
        exp_intersecting = {}
        baseline_intersecting = {}

        unique_to_exp_count = 0
        unique_to_baseline_count = 0
        intersecting_topic_ids = set(exp.qrels.keys()) | set(baseline.qrels.keys())

        for topic_id in intersecting_topic_ids:
            keys_1 = set(exp.qrels[topic_id].keys()) if topic_id in exp.qrels else set()
            keys_2 = (
                set(baseline.qrels[topic_id].keys())
                if topic_id in baseline.qrels
                else set()
            )

            intersecting_doc_ids = keys_1 & keys_2
            if intersecting_doc_ids:
                exp_intersecting[topic_id] = {
                    k: exp.qrels[topic_id][k] for k in intersecting_doc_ids
                }
                baseline_intersecting[topic_id] = {
                    k: baseline.qrels[topic_id][k] for k in intersecting_doc_ids
                }

            unique_to_exp_count += len(keys_1 - keys_2)
            unique_to_baseline_count += len(keys_2 - keys_1)

        exp.qrels = exp_intersecting
        baseline.qrels = baseline_intersecting
        return exp, baseline, unique_to_baseline_count, unique_to_exp_count

    @staticmethod
    def sample_qrels(exp: Any, baseline: Any) -> Tuple[Any, Any]:
        exp_flat = []
        for topic, qrels in exp.qrels.items():
            for doc_id in qrels.keys():
                exp_flat.append((topic, doc_id))

        samples = random.choices(exp_flat, k=len(exp_flat))

        exp_qrels = defaultdict(dict)
        baseline_qrels = defaultdict(dict)

        for i, (topic, doc_id) in enumerate(samples):
            # Create a unique key like to allow duplicates in the dictionary
            unique_key = f"{doc_id}_s{i}"

            exp_qrels[topic][unique_key] = exp.qrels[topic][doc_id]
            baseline_qrels[topic][unique_key] = baseline.qrels[topic][doc_id]

        # Update the objects
        exp.qrels = dict(exp_qrels)
        baseline.qrels = dict(baseline_qrels)

        return exp, baseline


class TopicsTransformer:
    @staticmethod
    def align_topics_inner(
        exp: Any, baseline: Any
    ) -> Union[List[str], List[str], int, int]:
        topic_ids = set(exp.topics.keys()).intersection(set(baseline.topics.keys()))
        ret_exp = {}
        ret_baseline = {}
        for topic_id in topic_ids:
            ret_exp[topic_id] = exp.topics[topic_id]
            ret_baseline[topic_id] = baseline.topics[topic_id]

        exp_missing = set(baseline.topics.keys()) - topic_ids
        if exp_missing:
            logger.warning(f"Missing topics in exp: {exp_missing}")
        baseline_missing = set(exp.topics.keys()) - topic_ids
        if baseline_missing:
            logger.warning(f"Missing topics in baseline: {baseline_missing}")

        exp.topics = ret_exp
        baseline.topics = ret_baseline
        return exp, baseline, len(exp_missing), len(baseline_missing)

    @staticmethod
    def sample_topics(exp: Any, baseline: Any) -> Tuple[Any, Any]:
        topic_ids = set(exp.topics.keys())
        samples = random.choices(list(topic_ids), k=len(topic_ids))

        exp_topics = defaultdict(dict)
        baseline_topics = defaultdict(dict)

        for i, topic_id in enumerate(samples):
            # Create a unique key like to allow duplicates in the dictionary
            unique_key = f"{topic_id}_s{i}"

            exp_topics[unique_key] = exp.topics[topic_id]
            baseline_topics[unique_key] = baseline.topics[topic_id]

        # Update the objects
        exp.topics = dict(exp_topics)
        baseline.topics = dict(baseline_topics)

        return exp, baseline

    @staticmethod
    def add_combined_field(exp: Any, field_names: Optional[List[str]] = None) -> Any:
        # infere field names from first topic if not provided
        if not field_names:
            first_topic = next(iter(exp.topics.values()))
            field_names = list(first_topic.keys())

        for topic_id, topic in exp.topics.items():
            combined_value = "\n\n".join(
                [str(topic.get(field, "")) for field in field_names]
            )
            topic["combined"] = combined_value
        return exp
