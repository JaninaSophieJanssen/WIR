import ir_measures
from ir_measures.util import QrelsConverter
from topic_gen.evaluate import Experiment
from topic_gen.evaluate.utils import QrelsTransformer


class TestQrelsTransformer:
    def test_binarize_qrels(self, test_data_dir):
        qrels = ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels"))
        qrels = QrelsConverter(qrels).as_dict_of_dict()

        assert qrels["0"]["D3"] == 2
        assert qrels["1"]["D3"] == 2
        assert qrels["1"]["D5"] == 2

        assert qrels["0"]["D0"] == 0
        assert qrels["0"]["D1"] == 1

        binarized = QrelsTransformer.binarize_relevance(qrels)

        assert binarized["0"]["D3"] == 1
        assert binarized["1"]["D3"] == 1
        assert binarized["1"]["D5"] == 1

        assert binarized["0"]["D0"] == 0
        assert binarized["0"]["D1"] == 1

    def test_align_qrels_inner(self, test_data_dir):
        qrels = ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels"))
        exp = Experiment(name="exp", qrels=qrels)

        qrels = ir_measures.read_trec_qrels(
            str(test_data_dir / "test-qrels-other.qrels")
        )
        baseline = Experiment(name="baseline", qrels=qrels)

        assert "2" in baseline.qrels.keys()
        assert "D6" in baseline.qrels["0"].keys()

        # apply alignment
        aligned_exp, aligned_baseline, n_missing_exp, n_missing_baseline = (
            QrelsTransformer.align_qrels_inner(exp, baseline)
        )

        assert "2" not in aligned_baseline.qrels.keys()

        assert "D6" not in aligned_baseline.qrels["0"].keys()

        assert aligned_exp.qrels.keys() == aligned_baseline.qrels.keys()
        for topic_id in aligned_baseline.qrels.keys():
            assert (
                aligned_exp.qrels[topic_id].keys()
                == aligned_baseline.qrels[topic_id].keys()
            )

    def test_sample_qrels(self, test_data_dir):
        qrels = ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels"))
        exp = Experiment(name="exp", qrels=qrels)

        qrels = ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-2.qrels"))
        baseline = Experiment(name="baseline", qrels=qrels)

        sampled_exp, sampled_baseline = QrelsTransformer.sample_qrels(exp, baseline)

        assert sampled_exp.qrels.keys() == sampled_baseline.qrels.keys()
        for topic_id in sampled_baseline.qrels.keys():
            assert (
                sampled_exp.qrels[topic_id].keys()
                == sampled_baseline.qrels[topic_id].keys()
            )
