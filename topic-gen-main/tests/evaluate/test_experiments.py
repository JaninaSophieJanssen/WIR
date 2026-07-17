import ir_measures
from ir_measures import MAP, nDCG
from topic_gen.evaluate import Experiment, MetaExperiment
from topic_gen.evaluate.io import read_jsonl_topics
from topic_gen.evaluate.measures_agreement import CohenKappa, JaccardIndex
from topic_gen.evaluate.measures_agreement_multiple import FleissKappa, RosKTau
from topic_gen.evaluate.measures_effectiveness import ARP

# from topic_gen.evaluate.utils import binarize_qrels

# def test_binarize_qrels():
#     qrels = ir_measures.read_trec_qrels("tests/data/test-qrels-1.qrels")

#     binarized = binarize_qrels(qrels)
#     qrels = ir_measures.read_trec_qrels("tests/data/test-qrels-1.qrels")
#     for original, b in zip(qrels, binarized):
#         if original.relevance == 999:
#             assert b.relevance == 999
#         elif original.relevance >= 1:
#             assert b.relevance == 1
#         else:
#             assert b.relevance == 0


# def test_qrel_comparison_two():
#     """Compare two qrels directly"""
#     from topic_gen.evaluate.experiment import QrelsEvaluator2
#     from topic_gen.evaluate.measures_qrel import CohenKappa

#     qrels1 = ir_measures.read_trec_qrels("tests/data/test-qrels-1.qrels")
#     qrels2 = ir_measures.read_trec_qrels("tests/data/test-qrels-2.qrels")

#     res = QrelsEvaluator2.experiment(
#         predictions=[qrels1], reference=qrels2, measures=[CohenKappa()]
#     )

#     assert res[0].value == 0.5


# def test_qrel_comparison_many():
#     """Compare multiple qrels directly"""
#     from topic_gen.evaluate.experiment import QrelsEvaluator
#     from topic_gen.evaluate.measures_qrel import FleissKappa

#     qrels1 = ir_measures.read_trec_qrels("tests/data/test-qrels-1.qrels")
#     qrels2 = ir_measures.read_trec_qrels("tests/data/test-qrels-2.qrels")
#     qrels3 = ir_measures.read_trec_qrels("tests/data/test-qrels-3.qrels")

#     res = QrelsEvaluator.experiment(
#         predictions=[qrels1, qrels2, qrels3],
#         reference=qrels1,
#         measures=[FleissKappa()],
#         names=["rater1", "rater2", "rater3"],
#     )
#     assert len(res) == 1


# EFFECTIVENESS = auto() # Requires (Run, Qrels)
# SIMILARITY = auto()    # Requires (Run, Run)
# AGREEMENT = auto()     # Requires (Qrels, Qrels)


def test_effectiveness_task(test_data_dir):
    """Similar to PyTerrier's evaluation"""
    exp1 = Experiment(
        run=ir_measures.read_trec_run(str(test_data_dir / "test-run1.run")),
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels")),
        name="run1",
    )
    exp2 = Experiment(
        run=ir_measures.read_trec_run(str(test_data_dir / "test-run2.run")),
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels")),
        name="run2",
    )
    exp3 = Experiment(
        run=ir_measures.read_trec_run(str(test_data_dir / "test-run3.run")),
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels")),
        name="run3",
    )

    meta_exp = MetaExperiment(
        experiments=[exp1, exp2, exp3], measures=[ARP(measures=[nDCG, MAP])]
    )

    res = meta_exp.evaluate()
    print(res)
    assert len(res) == 6
    assert res[0].name == "run1"
    assert res[0].measure == "arp(nDCG)"
    assert res[0].value == 0.5275760647440868

    assert res[1].measure == "arp(AP)"
    assert res[1].value == 0.48611111111111105

    assert res[2].name == "run2"
    assert res[2].measure == "arp(nDCG)"
    assert res[2].value == 0.5101737995448081

    assert res[3].name == "run2"
    assert res[3].measure == "arp(AP)"
    assert res[3].value == 0.45833333333333326

    assert res[4].name == "run3"
    assert res[4].measure == "arp(nDCG)"
    assert res[4].value == 0.5541923159780615

    assert res[5].name == "run3"
    assert res[5].measure == "arp(AP)"
    assert res[5].value == 0.48611111111111105


def test_system_stability_task(test_data_dir):
    exp1 = Experiment(
        run=ir_measures.read_trec_run(str(test_data_dir / "test-run1.run")),
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels")),
        name="run1",
    )
    exp2 = Experiment(
        run=ir_measures.read_trec_run(str(test_data_dir / "test-run2.run")),
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels")),
        name="run2",
    )
    exp3 = Experiment(
        run=ir_measures.read_trec_run(str(test_data_dir / "test-run3.run")),
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels")),
        name="run3",
    )

    meta_exp = MetaExperiment(
        experiments=[exp1, exp2, exp3],
        baseline=[exp1, exp2, exp3],
        measures=[RosKTau(measures=[nDCG, MAP])],
    )
    res = meta_exp.evaluate()
    print(res)
    assert len(res) == 2
    assert res[0].name == "ALL"
    assert res[0].measure == "RosKTau(nDCG)"
    assert res[0].value == 1.0

    assert res[1].name == "ALL"
    assert res[1].measure == "RosKTau(AP)"
    assert res[1].value == 0.9999999999999999


def test_annotator_agreement_task(test_data_dir):
    # Annotator agreement
    # Two annotators
    exp1 = Experiment(
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels")),
        name="annotator1",
    )
    exp2 = Experiment(
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-2.qrels")),
        name="annotator2",
    )
    exp3 = Experiment(
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-2.qrels")),
        name="annotator3",
    )
    meta_exp = MetaExperiment(
        experiments=[exp1, exp3],
        baseline=exp2,
        measures=[CohenKappa()],
    )
    res = meta_exp.evaluate()
    assert len(res) == 2

    assert res[0].name == "annotator1"
    assert res[0].value == 1.0

    assert res[1].name == "annotator3"
    assert res[1].value == 1.0

    assert res[0].measure == "Cohens $\\kappa$"
    assert res[1].measure == "Cohens $\\kappa$"


def test_inter_annotator_agreement_task(test_data_dir):
    # Multiple annotators
    exp1 = Experiment(
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-1.qrels")),
    )
    exp2 = Experiment(
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-2.qrels")),
    )
    exp3 = Experiment(
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-3.qrels")),
    )
    meta_exp = MetaExperiment(
        experiments=[exp1, exp2, exp3],
        measures=[FleissKappa()],
    )

    res = meta_exp.evaluate()
    assert len(res) == 1
    assert res[0].name == "ALL"
    assert res[0].measure == "FleissKappa"
    assert res[0].value == 1


# def test_repro_task():
#     qrels = ir_measures.read_trec_qrels("tests/data/test-qrels-1.qrels")
#     exp1 = Experiment(
#         run=ir_measures.read_trec_run("tests/data/test-run1.run"),
#         qrel=qrels,
#     )
#     exp2 = Experiment(
#         run=ir_measures.read_trec_run("tests/data/test-run2.run"),
#         qrel=qrels,
#     )
#     exp3 = Experiment(
#         run=ir_measures.read_trec_run("tests/data/test-run3.run"),
#         qrel=qrels,
#     )
#     meta_exp = MetaExperiment(
#         experiments=[exp2, exp3],
#         baseline=exp1,
#         measures=[RD],
#     )


# def test_temp_task():
#     qrels = ir_measures.read_trec_qrels("tests/data/test-qrels-1.qrels")
#     exp1 = Experiment(
#         run=ir_measures.read_trec_run("tests/data/test-run1.run"),
#         qrel=qrels,
#     )
#     exp2 = Experiment(
#         run=ir_measures.read_trec_run("tests/data/test-run2.run"),
#         qrel=qrels,
#     )
#     exp3 = Experiment(
#         run=ir_measures.read_trec_run("tests/data/test-run3.run"),
#         qrel=qrels,
#     )
#     exp4 = Experiment(
#         run=ir_measures.read_trec_run("tests/data/test-run4.run"),
#         qrel=qrels,
#     )
#     meta_exp = MetaExperiment(
#         experiments=[exp1, exp3],
#         baseline=[exp2, exp4],
#         measures=[RD],
#     )


def test_topic_sim_task(test_data_dir):
    exp1 = Experiment(
        topics=read_jsonl_topics(str(test_data_dir / "test-topic-1.topics")),
        name="gen",
    )
    exp2 = Experiment(
        topics=read_jsonl_topics(str(test_data_dir / "test-topic-2.topics")),
        name="ref",
    )
    meta_exp = MetaExperiment(
        experiments=[exp1],
        baseline=exp2,
        measures=[JaccardIndex()],
    )

    res = meta_exp.evaluate()

    assert len(res) == 3
    assert res[0].name == "gen"
    assert res[0].measure == "JaccardIndex(title)"
    assert res[0].value == 0.6488095238095238

    assert res[1].measure == "JaccardIndex(description)"
    assert res[1].value == 0.3666666666666667

    assert res[2].measure == "JaccardIndex(narrative)"
    assert res[2].value == 0.3988095238095238
