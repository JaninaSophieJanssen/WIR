import importlib.util

import pytest

from topic_gen.models import Qrel, Qrels, Topics, TRECTopic


@pytest.fixture
def sample_topics():
    t1 = TRECTopic(
        title="Aurora Borealis",
        description="Causes of aurora borealis.",
        narrative="A relevant document must explain the solar wind interaction.",
    )
    t1._id = "topic-1"
    t2 = TRECTopic(
        title="Climate Change",
        description="Effects of climate change.",
        narrative="A relevant document must discuss global warming impacts.",
    )
    t2._id = "topic-2"
    return Topics[TRECTopic](topics=[t1, t2])


class TestToXml:
    def test_returns_xml_string(self, sample_topics):
        xml = sample_topics.to_xml()
        assert "<topics>" in xml
        assert "<topic>" in xml
        assert "<title>Aurora Borealis</title>" in xml

    def test_writes_to_file(self, sample_topics, tmp_path):
        path = tmp_path / "topics.xml"
        sample_topics.to_xml(str(path))
        assert path.exists()
        assert "<title>Aurora Borealis</title>" in path.read_text()

    def test_xml_roundtrip(self, sample_topics, tmp_path):
        path = tmp_path / "topics.xml"
        sample_topics.to_xml(str(path))
        loaded = Topics[TRECTopic].read_xml(str(path))
        assert len(loaded.topics) == 2
        assert loaded.topics[0].title == "Aurora Borealis"
        assert loaded.topics[1].description == "Effects of climate change."

    def test_xml_does_not_preserve_topic_id(self, sample_topics, tmp_path):
        path = tmp_path / "topics.xml"
        sample_topics.to_xml(str(path))
        loaded = Topics[TRECTopic].read_xml(str(path))
        assert loaded.topics[0].topic_id != "topic-1"


class TestJsonl:
    def test_writes_one_line_per_topic(self, sample_topics, tmp_path):
        path = tmp_path / "topics.jsonl"
        sample_topics.to_jsonl(str(path))
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_jsonl_roundtrip(self, sample_topics, tmp_path):
        path = tmp_path / "topics.jsonl"
        sample_topics.to_jsonl(str(path))
        loaded = Topics[TRECTopic].read_jsonl(str(path))
        assert len(loaded.topics) == 2
        assert loaded.topics[0].title == "Aurora Borealis"
        assert loaded.topics[1].title == "Climate Change"

    def test_jsonl_preserves_topic_id(self, sample_topics, tmp_path):
        path = tmp_path / "topics.jsonl"
        sample_topics.to_jsonl(str(path))
        loaded = Topics[TRECTopic].read_jsonl(str(path))
        assert loaded.topics[0].topic_id == "topic-1"
        assert loaded.topics[1].topic_id == "topic-2"


@pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None,
    reason="pandas not installed (install with [evaluate] extra)",
)
class TestCsv:
    def test_writes_csv_with_header(self, sample_topics, tmp_path):
        path = tmp_path / "topics.csv"
        sample_topics.to_csv(str(path))
        content = path.read_text()
        assert "title" in content
        assert "Aurora Borealis" in content

    def test_csv_roundtrip(self, sample_topics, tmp_path):
        path = tmp_path / "topics.csv"
        sample_topics.to_csv(str(path))
        loaded = Topics[TRECTopic].from_csv(str(path))
        assert len(loaded.topics) == 2
        assert loaded.topics[0].title == "Aurora Borealis"
        assert loaded.topics[1].title == "Climate Change"

    def test_csv_preserves_topic_id(self, sample_topics, tmp_path):
        path = tmp_path / "topics.csv"
        sample_topics.to_csv(str(path))
        loaded = Topics[TRECTopic].from_csv(str(path))
        assert loaded.topics[0].topic_id == "topic-1"
        assert loaded.topics[1].topic_id == "topic-2"


@pytest.fixture
def sample_qrels():
    return Qrels[Qrel](
        qrels=[
            Qrel(query_id="301", doc_id="FBIS3-10082", relevance=0),
            Qrel(query_id="301", doc_id="FBIS3-10112", relevance=2),
            Qrel(query_id="302", doc_id="FBIS4-45678", relevance=1, iteration="1"),
        ]
    )


class TestToTrec:
    def test_writes_four_column_format(self, sample_qrels, tmp_path):
        path = tmp_path / "qrels.txt"
        sample_qrels.to_trec(str(path))
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "301 0 FBIS3-10082 0"
        assert lines[1] == "301 0 FBIS3-10112 2"
        assert lines[2] == "302 1 FBIS4-45678 1"

    def test_one_line_per_qrel(self, sample_qrels, tmp_path):
        path = tmp_path / "qrels.txt"
        sample_qrels.to_trec(str(path))
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == len(sample_qrels.qrels)


class TestReadTrec:
    def test_reads_four_column_file(self, tmp_path):
        path = tmp_path / "qrels.txt"
        path.write_text("301 0 FBIS3-10082 0\n301 0 FBIS3-10112 2\n")
        loaded = Qrels[Qrel].read_trec(str(path))
        assert len(loaded.qrels) == 2
        assert loaded.qrels[0].query_id == "301"
        assert loaded.qrels[0].doc_id == "FBIS3-10082"
        assert loaded.qrels[0].relevance == 0
        assert loaded.qrels[1].relevance == 2

    def test_reads_three_column_file(self, tmp_path):
        path = tmp_path / "qrels.txt"
        path.write_text("301 FBIS3-10082 0\n301 FBIS3-10112 2\n")
        loaded = Qrels[Qrel].read_trec(str(path))
        assert len(loaded.qrels) == 2
        assert loaded.qrels[0].iteration == "0"

    def test_skips_blank_lines_and_comments(self, tmp_path):
        path = tmp_path / "qrels.txt"
        path.write_text("# comment\n\n301 0 FBIS3-10082 0\n")
        loaded = Qrels[Qrel].read_trec(str(path))
        assert len(loaded.qrels) == 1

    def test_trec_roundtrip(self, sample_qrels, tmp_path):
        path = tmp_path / "qrels.txt"
        sample_qrels.to_trec(str(path))
        loaded = Qrels[Qrel].read_trec(str(path))
        assert len(loaded.qrels) == len(sample_qrels.qrels)
        for orig, loaded_qrel in zip(sample_qrels.qrels, loaded.qrels):
            assert loaded_qrel.query_id == orig.query_id
            assert loaded_qrel.doc_id == orig.doc_id
            assert loaded_qrel.relevance == orig.relevance
            assert loaded_qrel.iteration == orig.iteration
