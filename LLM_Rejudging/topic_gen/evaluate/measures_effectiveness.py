from typing import List

from topic_gen.evaluate import BaseMeasure, Experiment, MeasureResult, MeasureType


class ARP(BaseMeasure):
    def __init__(self, measures: List[str]):
        super().__init__(name="ARP", m_type=MeasureType.EFFECTIVENESS)
        self.measures = measures

    def calc_agg(self, experiment: Experiment):
        results = experiment.get_arp(self.measures)

        ret = []
        for measure, value in results.items():
            ret.append(
                MeasureResult(
                    name=experiment.name,
                    measure=f"arp({measure})",
                    value=value,
                )
            )
        return ret

