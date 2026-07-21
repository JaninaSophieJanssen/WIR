

from topic_gen.evaluate.measures_agreement import AreaUnderReceiver
from topic_gen.evaluate import Experiment
import pytest
import ir_measures


class TestAreaUnderReceiver:
    def test_calc_agg_empty_qrels(self, test_data_dir):
        measure = AreaUnderReceiver()

        qrels = ir_measures.read_trec_qrels(
            str(test_data_dir / "test-qrels-1.qrels"))
        experiment = Experiment(qrels=qrels, binarize_qrels=1)

        qrels = ir_measures.read_trec_qrels(
            str(test_data_dir / "test-qrels-1.qrels"))
        baseline = Experiment(qrels=qrels, binarize_qrels=1)

        res = measure.calc_agg(experiment, baseline)
        print(res)
        assert True == False
