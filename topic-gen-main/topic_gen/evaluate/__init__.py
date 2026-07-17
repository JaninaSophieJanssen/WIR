from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from itertools import combinations
from typing import List, NamedTuple, Optional, Union

import ir_measures
import numpy as np
import pandas as pd
from ir_measures.util import TYPE_QREL, TYPE_RUN, QrelsConverter, RunConverter
from scipy import stats

from topic_gen.evaluate.utils import QrelsTransformer, TopicsTransformer


class MeasureType(Enum):
    EFFECTIVENESS = auto()  # Requires (Run, Qrels)
    AGREEMENT = auto()  # Requires (Qrels, Qrels)
    AGREEMENT_MANY = auto()  # Requires (Qrels,Qrels,Qrels,...)
    SIMILARITY = auto()  # Requires (Run, Run)


class Topic(NamedTuple):
    title: str
    description: str
    narrative: str
    id: str


class MeasureResult(NamedTuple):
    name: str
    measure: str
    value: float


class BaseMeasure:
    def __init__(self, name: str, m_type: MeasureType):
        self.name = name
        self.type = m_type


class Experiment:
    def __init__(
        self,
        name: Optional[str] = "",
        run: Optional[TYPE_RUN] = None,
        qrels: Optional[TYPE_QREL] = None,
        topics: Optional[Topic] = None,
        binarize_qrels: bool = 0,
    ):
        self.run = RunConverter(run).as_dict_of_dict() if run else None
        self.qrels = QrelsConverter(qrels).as_dict_of_dict() if qrels else None
        self.topics = topics
        self.name = name
        if binarize_qrels > 0 and self.qrels:
            self.binarize_qrels(threshold=binarize_qrels)
        self.arp = {}

    def get_arp(self, measures: List[ir_measures.Measure]) -> float:
        not_computed = [m for m in measures if m not in self.arp]
        if not_computed:
            results = ir_measures.calc_aggregate(not_computed, self.qrels, self.run)
            for measure in not_computed:
                self.arp[measure] = results[measure]

        ret = {}
        for measure in measures:
            arp = self.arp.get(measure)
            ret[measure] = arp
        return ret

    def binarize_qrels(self, threshold: int = 1):
        from topic_gen.evaluate.utils import QrelsTransformer

        self.qrels = QrelsTransformer.binarize_relevance(self.qrels, threshold)


class MetaExperiment:
    def __init__(
        self,
        experiments: List[Union[Experiment, List[Experiment]]],
        measures: List[BaseMeasure],
        baseline: Optional[Union[Experiment, List[Experiment]]] = None,
        filter_topics: Optional[bool] = False,
        filter_qrels: Optional[bool] = False,
        bootstrap: Optional[int] = 0,
        test: Optional[str] = None,
        alpha: Optional[float] = 0.05,
        correction: Optional[bool] = True,
    ):
        self.experiments = experiments
        self.measures = measures
        self.baseline = baseline
        self.align_topics = filter_topics
        self.align_qrels = filter_qrels
        self.bootstrap = bootstrap
        self.test = test
        self.alpha = alpha
        self.correction = correction
        
    def run_bootstrap(self, exp, baseline, measure, return_samples=False):
        bootstraps = defaultdict(list)
        for _ in range(self.bootstrap):
            exp_copy = deepcopy(exp)
            baseline_copy = deepcopy(baseline)
            if exp.qrels:
                exp_sampled, baseline_sampled = QrelsTransformer.sample_qrels(
                    exp_copy, baseline_copy
                )
            else:
                exp_sampled, baseline_sampled = TopicsTransformer.sample_topics(
                    exp_copy, baseline_copy
                )

            res_sampled = measure.calc_agg(exp_sampled, baseline_sampled)

            for r in res_sampled:
                bootstraps[r.measure].append(r.value)

        if return_samples:
            return bootstraps

        ret = []
        for measure in bootstraps.keys():
            values = np.array(bootstraps.get(measure))
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            ci = (ci_upper - ci_lower) / 2
            ret.extend(
                [
                    MeasureResult(exp.name, f"{measure}_ci_lower", float(ci_lower)),
                    MeasureResult(exp.name, f"{measure}_ci_upper", float(ci_upper)),
                    MeasureResult(exp.name, f"{measure}_ci", float(ci)),
                ]
            )
        return ret

    def evaluate(self):
        ret = []
        for measure in self.measures:
            if measure.type == MeasureType.EFFECTIVENESS:
                for exp in self.experiments:
                    res = measure.calc_agg(exp)
                    ret.extend(res)

            elif measure.type == MeasureType.AGREEMENT:
                for exp in self.experiments:
                    # copy experiments and baselines before we align them
                    exp_copy = deepcopy(exp)
                    baseline_copy = deepcopy(self.baseline)

                    if self.align_topics:
                        (
                            exp_copy,
                            baseline_copy,
                            n_missing_exp_topics,
                            n_missing_baseline_topics,
                        ) = TopicsTransformer.align_topics_inner(
                            exp_copy, baseline_copy
                        )
                        ret.append(
                            MeasureResult(
                                exp.name,
                                "missing_topics",
                                n_missing_exp_topics,
                            )
                        )

                    if self.align_qrels:
                        (
                            exp_copy,
                            baseline_copy,
                            n_missing_exp_qrels,
                            n_missing_baseline_qrels,
                        ) = QrelsTransformer.align_qrels_inner(exp_copy, baseline_copy)
                        ret.append(
                            MeasureResult(
                                exp.name,
                                "missing_qrels",
                                n_missing_exp_qrels,
                            )
                        )

                    ret.extend(measure.calc_agg(exp_copy, baseline_copy))

                    if self.bootstrap:
                        res_boot = self.run_bootstrap(exp_copy, baseline_copy, measure)
                        ret.extend(res_boot)

                if self.test:
                    bootstraps = defaultdict(list)
                    for exp in self.experiments:
                        res_boot = self.run_bootstrap(
                            deepcopy(exp), baseline_copy, measure, return_samples=True
                        )
                        bootstraps[exp.name].extend(res_boot[measure.name])



                    pairs = list(combinations(bootstraps.keys(), 2))
                    
                    if self.correction:
                        n_comparisons = len(pairs)
                        alpha = self.alpha / n_comparisons if n_comparisons > 0 else self.alpha
                        corrected = "_bonferroni"
                    else:
                        alpha = self.alpha
                        corrected = ""

                for g1, g2 in pairs:
                    t_stat, p_val = stats.ttest_ind(bootstraps[g1], bootstraps[g2])

                    ret.extend(
                        [
                            MeasureResult(
                                g1,
                                f"{self.test}_pvalue_{measure.name}_vs_{g2}",
                                p_val,
                            ),
                            MeasureResult(
                                g2,
                                f"{self.test}_pvalue_{measure.name}_vs_{g1}",
                                p_val,
                            ),
                        ]
                    )
                    if self.alpha:
                        is_sig = p_val < alpha
                        ret.extend(
                            [
                                MeasureResult(
                                    name=g1,
                                    measure=f"{self.test}_significant{corrected}_{measure.name}_vs_{g2}",
                                    value=is_sig,
                                ),
                                MeasureResult(
                                    name=g2,
                                    measure=f"{self.test}_significant{corrected}_{measure.name}_vs_{g1}",
                                    value=is_sig,
                                ),
                            ]
                        )

            elif measure.type == MeasureType.AGREEMENT_MANY:
                res = measure.calc_agg(self.experiments, self.baseline)
                ret.extend(res)
            else:
                raise NotImplementedError(
                    f"Measure type {measure.type} not implemented in MetaExperiment."
                )
        return ret
