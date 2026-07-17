from collections import defaultdict
from typing import Dict, List

import evaluate
import numpy as np
import torch
from scipy.stats import kendalltau
from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from topic_gen.evaluate import BaseMeasure, Experiment, MeasureResult, MeasureType
from topic_gen.evaluate.utils import QrelsTransformer


# Topic Measures
class JaccardIndex(BaseMeasure):
    def __init__(self):
        super().__init__(name="JaccardIndex", m_type=MeasureType.AGREEMENT)

    def jaccard_index(self, text1: str, text2: str) -> float:
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    def calc(self, experiment: Experiment, baseline: Experiment) -> List[float]:
        fields = baseline[next(iter(baseline))].keys()
        ret = defaultdict(dict)
        for topic_id in experiment.keys():
            for field in fields:
                text1 = experiment[topic_id].get(field, "")
                text2 = baseline[topic_id][field]
                score = self.jaccard_index(text1, text2)
                ret[topic_id][field] = score
        return ret

    def calc_agg(
        self, experiment: Experiment, baseline: Experiment
    ) -> List[MeasureResult]:
        assert len(experiment.topics.keys()) == len(baseline.topics.keys()), (
            "Number of topics in both topics must be the same."
        )
        scores = self.calc(experiment.topics, baseline.topics)

        # aggregate by topic
        averages = {}
        for topic_id in scores.keys():  # {topic_id: {field: score}}
            for field, score in scores[topic_id].items():
                if field not in averages:
                    averages[field] = []
                averages[field].append(score)

        ret = []
        for field, topic_scores in averages.items():
            jaccard_index = sum(topic_scores) / len(topic_scores)

            ret.append(
                MeasureResult(
                    name=experiment.name,
                    measure=f"{self.name}({field})",
                    value=jaccard_index,
                )
            )
        return ret


class BertScore(BaseMeasure):
    def __init__(
        self,
        model: str = "microsoft/deberta-xlarge-mnli",
        batch_size: int = 32,
        device: str = None,
    ):
        super().__init__(name="BertScore", m_type=MeasureType.AGREEMENT)
        self.model = model
        self.name = "BertScore"
        self.batch_size = batch_size
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.bertscore_metric = evaluate.load("bertscore")

    def calc(
        self, experiment: Experiment, baseline: Experiment
    ) -> Dict[str, Dict[str, float]]:
        ret = defaultdict(dict)
        map_keys = []
        all_preds = []
        all_refs = []

        fields = baseline[next(iter(baseline))].keys()

        for topic_id in experiment.keys():
            for field in fields:
                map_keys.append((topic_id, field))

                all_preds.append(experiment[topic_id].get(field, ""))
                all_refs.append(baseline[topic_id].get(field, ""))

        if not all_preds:
            return ret

        results = self.bertscore_metric.compute(
            predictions=all_preds,
            references=all_refs,
            model_type=self.model,
            batch_size=self.batch_size,
            device=self.device,
            verbose=False,
        )

        f1_scores = results["f1"]
        for i, (topic_id, field) in enumerate(map_keys):
            ret[topic_id][field] = f1_scores[i]

        return ret

    def calc_agg(
        self, experiment: Experiment, baseline: Experiment
    ) -> List[MeasureResult]:
        assert len(experiment.topics.keys()) == len(baseline.topics.keys()), (
            "Number of topics in both experiments must be the same."
        )

        scores = self.calc(experiment.topics, baseline.topics)

        averages = defaultdict(list)
        for topic_id in scores.keys():
            for field, score in scores[topic_id].items():
                averages[field].append(score)

        ret = []
        for field, topic_scores in averages.items():
            if not topic_scores:
                continue

            score = sum(topic_scores) / len(topic_scores)

            ret.append(
                MeasureResult(
                    name=experiment.name,
                    measure=f"{self.name}({field})",
                    value=score,
                )
            )
        return ret


class RougeScore(BaseMeasure):
    def __init__(self, use_stemmer: bool = True):
        super().__init__(name="RougeScore", m_type=MeasureType.AGREEMENT)
        self.rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        self.use_stemmer = use_stemmer
        self.metric = evaluate.load("rouge")

    def calc(
        self, experiment: Experiment, baseline: Experiment
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        ret = defaultdict(lambda: defaultdict(dict))
        map_keys = []
        all_preds = []
        all_refs = []

        fields = baseline[next(iter(baseline))].keys()

        for topic_id in experiment.keys():
            for field in fields:
                pred_text = experiment[topic_id].get(field, "")
                ref_text = baseline[topic_id].get(field, "")
                if not pred_text or not ref_text:
                    continue

                map_keys.append((topic_id, field))
                all_preds.append(pred_text)
                all_refs.append(ref_text)

        if not all_preds:
            return ret

        results = self.metric.compute(
            predictions=all_preds,
            references=all_refs,
            use_stemmer=self.use_stemmer,
            use_aggregator=False,
        )

        for rouge_type in self.rouge_types:
            rouge_scores = results.get(rouge_type, [])
            for i, (topic_id, field) in enumerate(map_keys):
                if i < len(rouge_scores):
                    score = float(rouge_scores[i])
                    ret[topic_id][field][rouge_type] = score

        return ret

    def calc_agg(
        self, experiment: Experiment, baseline: Experiment
    ) -> List[MeasureResult]:
        assert len(experiment.topics.keys()) == len(baseline.topics.keys()), (
            "Number of topics in both experiments must be the same."
        )

        scores = self.calc(experiment.topics, baseline.topics)

        averages = defaultdict(lambda: defaultdict(list))
        for topic_id in scores.keys():
            for field, rouge_scores in scores[topic_id].items():
                for rouge_type, score in rouge_scores.items():
                    averages[field][rouge_type].append(score)

        ret = []
        for field, rouge_types_dict in averages.items():
            for rouge_type, topic_scores in rouge_types_dict.items():
                if not topic_scores:
                    continue

                score = sum(topic_scores) / len(topic_scores)

                ret.append(
                    MeasureResult(
                        name=experiment.name,
                        measure=f"{rouge_type}({field})",
                        value=score,
                    )
                )
        return ret


class RelativeLength(BaseMeasure):
    def __init__(self):
        super().__init__(name="RelativeLength", m_type=MeasureType.AGREEMENT)
        self.name = "RelativeLength"

    def calc(self, experiment: Experiment, baseline: Experiment) -> List[float]:
        fields = baseline[next(iter(baseline))].keys()
        ret = defaultdict(dict)
        for topic_id in experiment.keys():
            for field in fields:
                score = (
                    len(experiment[topic_id].get(field, ""))
                    / len(baseline[topic_id].get(field, ""))
                    if len(baseline[topic_id].get(field, "")) > 0
                    else 0
                )
                ret[topic_id][field] = score
        return ret

    def calc_agg(
        self, experiment: Experiment, baseline: Experiment
    ) -> List[MeasureResult]:
        assert len(experiment.topics.keys()) == len(baseline.topics.keys()), (
            "Number of topics in both topics must be the same."
        )
        scores = self.calc(experiment.topics, baseline.topics)

        # aggregate by topic
        averages = {}
        for topic_id in scores.keys():  # {topic_id: {field: score}}
            for field, score in scores[topic_id].items():
                if field not in averages:
                    averages[field] = []
                averages[field].append(score)

        ret = []
        for field, topic_scores in averages.items():
            score = sum(topic_scores) / len(topic_scores)

            ret.append(
                MeasureResult(
                    name=experiment.name,
                    measure=f"{self.name}({field})",
                    value=score,
                )
            )
        return ret


class CosineSimilarity(BaseMeasure):
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = None,
        max_cache_size: int = 10000,
    ):
        super().__init__(name="CosineSimilarity", m_type=MeasureType.AGREEMENT)
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = SentenceTransformer(model, device=self.device)
        self.name = "CosineSimilarity"
        self.batch_size = batch_size
        self.max_cache_size = max_cache_size
        self.cache = {}

    def encode(self, texts: List[str], cache: bool = False) -> torch.Tensor:
        if cache:
            if len(self.cache) + len(texts) > self.max_cache_size:
                self.cache = {}

            missing_texts = [t for t in texts if t not in self.cache]
            if missing_texts:
                embeddings = self.model.encode(
                    sentences=missing_texts,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device,
                )
                for text, emb in zip(missing_texts, embeddings):
                    self.cache[text] = emb

            embeddings = torch.stack([self.cache[t] for t in texts])
            return embeddings
        else:
            embeddings = self.model.encode(
                sentences=texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            )
            return embeddings

    def calc(
        self, experiment: Experiment, baseline: Experiment
    ) -> Dict[str, Dict[str, float]]:
        ret = defaultdict(dict)
        exp_texts = []
        base_texts = []
        keys_map = []

        first_topic = next(iter(baseline), None)
        if not first_topic:
            return ret

        fields = baseline[first_topic].keys()

        for topic_id in experiment.keys():
            if topic_id not in baseline:
                continue

            for field in fields:
                exp_text = experiment[topic_id].get(field, "")
                base_text = baseline[topic_id].get(field, "")
                if not exp_text or not base_text:
                    continue

                exp_texts.append(exp_text)
                base_texts.append(base_text)
                keys_map.append((topic_id, field))

        if not exp_texts:
            return ret

        t_exp = self.encode(exp_texts)
        t_base = self.encode(base_texts, cache=True)

        scores = torch.nn.functional.cosine_similarity(t_exp, t_base, dim=1)
        scores = scores.cpu().numpy()

        for i, (topic_id, field) in enumerate(keys_map):
            ret[topic_id][field] = float(scores[i])

        return ret

    def calc_agg(
        self, experiment: Experiment, baseline: Experiment
    ) -> List[MeasureResult]:
        assert len(experiment.topics.keys()) == len(baseline.topics.keys()), (
            "Number of topics in both topics must be the same."
        )

        scores = self.calc(experiment.topics, baseline.topics)

        # Aggregate by topic
        averages = defaultdict(list)
        for topic_id in scores.keys():
            for field, score in scores[topic_id].items():
                averages[field].append(score)

        ret = []
        for field, topic_scores in averages.items():
            if not topic_scores:
                continue
            score = sum(topic_scores) / len(topic_scores)

            ret.append(
                MeasureResult(
                    name=experiment.name,
                    measure=f"{self.name}({field})",
                    value=score,
                )
            )
        return ret


# Qrel Measures
class CohenKappa(BaseMeasure):
    def __init__(self, weights="linear"):
        super().__init__(name="Cohens $\\kappa$", m_type=MeasureType.AGREEMENT)
        self.weights = weights

    def calc_agg(
        self, experiment: Experiment, baseline: Experiment
    ) -> List[MeasureResult]:
        assert len(experiment.qrels) == len(baseline.qrels), (
            "Number of topics in both qrels must be the same."
        )

        kappa_score = cohen_kappa_score(
            QrelsTransformer.flatten_relevance(experiment.qrels),
            QrelsTransformer.flatten_relevance(baseline.qrels),
            weights=self.weights,
        )

        return [
            MeasureResult(
                name=experiment.name,
                measure=self.name,
                value=kappa_score,
            )
        ]


class AreaUnderReceiver(BaseMeasure):
    def __init__(self):
        super().__init__(name="AUR", m_type=MeasureType.AGREEMENT)

    def calc_agg(
        self, experiment: Experiment, baseline: Experiment
    ) -> List[MeasureResult]:
        assert len(experiment.qrels) == len(baseline.qrels), (
            "Number of topics in both qrels must be the same."
        )
        assert len(experiment.qrels) + len(baseline.qrels) > 0, "Qrels cannot be empty."

        auc_score = roc_auc_score(
            QrelsTransformer.flatten_relevance(experiment.qrels),
            QrelsTransformer.flatten_relevance(baseline.qrels),
        )

        return [
            MeasureResult(
                name=experiment.name,
                measure=self.name,
                value=auc_score,
            )
        ]


class MeanAverageError(BaseMeasure):
    def __init__(self):
        super().__init__(name="MAE", m_type=MeasureType.AGREEMENT)

    def calc_agg(
        self, experiment: Experiment, baseline: Experiment
    ) -> List[MeasureResult]:
        assert len(experiment.qrels) == len(baseline.qrels), (
            "Number of topics in both qrels must be the same."
        )

        experiment_labels = QrelsTransformer.flatten_relevance(experiment.qrels)
        baseline_labels = QrelsTransformer.flatten_relevance(baseline.qrels)
        mae_score = np.abs(
            np.array(experiment_labels) - np.array(baseline_labels)
        ).mean()

        return [
            MeasureResult(
                name=experiment.name,
                measure=self.name,
                value=mae_score,
            )
        ]


class KendallTau(BaseMeasure):
    def __init__(self):
        super().__init__(name="KendallTau", m_type=MeasureType.AGREEMENT)

    def calc_agg(
        self, experiment: Experiment, baseline: Experiment
    ) -> List[MeasureResult]:
        assert len(experiment.qrels) == len(baseline.qrels), (
            "Number of topics in both qrels must be the same."
        )

        experiment_labels = QrelsTransformer.flatten_relevance(experiment.qrels)
        baseline_labels = QrelsTransformer.flatten_relevance(baseline.qrels)
        score = kendalltau(experiment_labels, baseline_labels).correlation

        return [
            MeasureResult(
                name=experiment.name,
                measure=self.name,
                value=score,
            )
        ]
