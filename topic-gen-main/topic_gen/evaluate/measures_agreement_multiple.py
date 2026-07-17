from collections import defaultdict
import krippendorff
from typing import List, Optional

from scipy.stats import kendalltau
from statsmodels.stats.inter_rater import fleiss_kappa

from topic_gen.evaluate import BaseMeasure, Experiment, MeasureResult, MeasureType
from topic_gen.evaluate.utils import QrelsTransformer


class RosKTau(BaseMeasure):
    def __init__(self, measures: List[str]):
        super().__init__(name="RosKTau", m_type=MeasureType.AGREEMENT_MANY)
        self.measures = measures

    def get_ros_by_scores(self, runs: List[Experiment], measure: str) -> List[float]:
        ros_scores = {}
        for run in runs:
            score = run.get_arp([measure])[measure]
            if run.name in ros_scores:
                raise ValueError(f"Duplicate run name `{run.name}`.")
            ros_scores[run.name] = score

        sorted_names = sorted(ros_scores.keys())
        return [ros_scores[name] for name in sorted_names]

    def calc_agg(
        self, experiments: List[Experiment], baseline: List[Experiment]
    ) -> List[MeasureResult]:
        # align experiments and baseline
        assert len(experiments) == len(
            baseline
        ), "Number of experiments and baseline experiments must be the same."
        assert set([exp.name for exp in experiments]) == set(
            [exp.name for exp in baseline]
        ), "Experiment names and baseline experiment names must match."

        ret = []
        for measure in self.measures:
            experimental_scores = self.get_ros_by_scores(experiments, measure)
            baseline_scores = self.get_ros_by_scores(baseline, measure)
            tau = kendalltau(experimental_scores, baseline_scores)
            ret.append(
                MeasureResult(
                    name="ALL",
                    measure=f"RosKTau({measure})",
                    value=tau.correlation,
                )
            )
        return ret


class RosRBO(BaseMeasure):
    def __init__(self, measures: List[str], p: float, k: int):
        super().__init__(name="RosRBO", m_type=MeasureType.AGREEMENT_MANY)
        self.measures = measures
        self.p = p
        self.k = k

    def get_ros_by_scores(self, runs: List[Experiment], measure: str) -> List[float]:
        ros_scores = {}
        for run in runs:
            score = run.get_arp([measure])[measure]
            if run.name in ros_scores:
                raise ValueError(f"Duplicate run name `{run.name}`.")
            ros_scores[run.name] = score

        sorted_names = sorted(ros_scores.values())
        return list(sorted_names.keys())

    def calc_agg(
        self, experiments: List[Experiment], baseline: List[Experiment]
    ) -> List[MeasureResult]:
        # align experiments and baseline
        assert len(experiments) == len(
            baseline
        ), "Number of experiments and baseline experiments must be the same."
        assert set([exp.name for exp in experiments]) == set(
            [exp.name for exp in baseline]
        ), "Experiment names and baseline experiment names must match."

        ret = []
        for measure in self.measures:
            experimental_ranking = self.get_ros_by_scores(experiments, measure)
            baseline_ranking = self.get_ros_by_scores(baseline, measure)

            # Implementation taken from the TREC Health Misinformation Track with modifications
            # see also: https://github.com/claclark/Compatibility
            run_set = set()
            ideal_set = set()

            score = 0.0
            normalizer = 0.0
            weight = 1.0
            for i in range(self.k):
                if i < len(experimental_ranking):
                    run_set.add(experimental_ranking[i])
                if i < len(baseline_ranking):
                    ideal_set.add(baseline_ranking[i])
                score += weight * \
                    len(ideal_set.intersection(run_set)) / (i + 1)
                normalizer += weight
                weight *= self.p

            score_RBO = score / normalizer
            ret.append(
                MeasureResult(
                    name="ALL",
                    measure=f"{self.name}({measure})",
                    value=score_RBO,
                )
            )
        return ret


class FleissKappa(BaseMeasure):
    def __init__(self):
        super().__init__(name="FleissKappa", m_type=MeasureType.AGREEMENT_MANY)

    def calc_agg(
        self,
        experiments: List[Experiment],
        baseline: Optional[List[Experiment]] = None,
    ) -> List[MeasureResult]:

        score = fleiss_kappa(
            QrelsTransformer.count_by_relevance(experiments),
            method="fleiss",
        )

        return [
            MeasureResult(
                name="ALL",
                measure=self.name,
                value=score,
            )
        ]


class KrippendorffAlpha(BaseMeasure):
    def __init__(self, level_of_measurement="ordinal"):
        super().__init__(name="KrippendorffAlpha", m_type=MeasureType.AGREEMENT_MANY)

        self.level_of_measurement = level_of_measurement

    def calc_agg(
        self,
        experiments: List[Experiment],
        baseline: Optional[List[Experiment]] = None,
    ) -> List[MeasureResult]:

        rater = []
        for exp in experiments:
            rater.append(QrelsTransformer.flatten_relevance(exp.qrels))

        score = krippendorff.alpha(
            rater, level_of_measurement=self.level_of_measurement)

        return [
            MeasureResult(
                name="ALL",
                measure=self.name,
                value=score,
            )
        ]


class LabelDistribution(BaseMeasure):
    def __init__(self):
        super().__init__(name="LabelDistribution", m_type=MeasureType.AGREEMENT_MANY)

    def get_distribution(
        self,
        experiment: Experiment,
    ) -> dict:
        label_counts = defaultdict(int)
        c = 0
        for _, qrel in experiment.qrels.items():
            for label in qrel.values():
                label_counts[label] += 1
                c += 1

        label_distribution = {}
        for label, count in label_counts.items():
            label_distribution[label] = count / c

        return label_distribution

    def calc_agg(
        self,
        experiments: List[Experiment],
        baseline: Experiment,
    ) -> List[MeasureResult]:

        ret = []
        distribution = self.get_distribution(baseline)
        for label in distribution.keys():
            ret.append(
                MeasureResult(
                    name=baseline.name,
                    measure=f"label_dist({label})",
                    value=distribution[label],
                )
            )

        for experiment in experiments:
            distribution = self.get_distribution(experiment)
            for label in distribution.keys():
                ret.append(
                    MeasureResult(
                        name=experiment.name,
                        measure=f"label_dist({label})",
                        value=distribution[label],
                    )
                )
        return ret
