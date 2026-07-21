import ir_measures
from topic_gen.evaluate import Experiment, MetaExperiment
from topic_gen.evaluate.measures_agreement import CohenKappa


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
        qrels=ir_measures.read_trec_qrels(str(test_data_dir / "test-qrels-3.qrels")),
        name="annotator3",
    )
    meta_exp = MetaExperiment(
        experiments=[exp1, exp3],
        baseline=exp2,
        measures=[CohenKappa()],
        bootstrap=20,
        filter_qrels=True,
    )

    res = meta_exp.evaluate()
    assert len(res) == 8
    print(res)
    assert res[1].name == "annotator1"
    assert res[1].measure == "Cohens $\\kappa$ ci lower"
    assert res[1].value == 1.0
    assert res[1].measure == "Cohens $\\kappa$ ci upper"
    assert res[1].value == 1.0
    assert res[1].measure == "Cohens $\\kappa$ ci"
    assert res[1].value == 1.0


    assert res[5].name == "annotator3"
    assert res[5].measure == "Cohens $\\kappa$ ci lower"
    assert res[5].value == 1.0
    assert res[5].measure == "Cohens $\\kappa$ ci upper"
    assert res[5].value == 1.0
    assert res[5].measure == "Cohens $\\kappa$ ci"
    assert res[5].value == 1.0
