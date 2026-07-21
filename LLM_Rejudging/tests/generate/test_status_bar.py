import unittest
from unittest.mock import MagicMock, patch
from topic_gen.generate import Generator


class TestStatusBar(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        with (
            patch("topic_gen.generate.os.path.exists", return_value=True),
            patch("topic_gen.generate.load_prompt", return_value=MagicMock()),
        ):
            self.generator = Generator(
                llm=self.mock_llm, prompt="dummy_path", parse=False
            )
        self.generator.chain = MagicMock()

    def test_generate_uses_thread_map(self):
        """Test that generate() calls thread_map for batch processing."""

        # Prepare inputs
        items = ["1", "2", "3"]
        kwargs = {"topic": list(range(3))}

        # Patch thread_map where it is used (in topic_gen.generate)
        with patch("topic_gen.generate.thread_map") as mock_thread_map:
            # thread_map returns a list of results
            mock_thread_map.return_value = ["result1", "result2", "result3"]

            results = self.generator.generate(item_ids=items, **kwargs)

            # Verify thread_map was called
            mock_thread_map.assert_called_once()

            # Verify arguments: first arg should be the function (self._safe_invoke),
            # second is the inputs list
            args, call_kwargs = mock_thread_map.call_args
            self.assertEqual(args[0], self.generator._safe_invoke)
            self.assertEqual(len(args[1]), 3)  # 3 items to process
            self.assertEqual(call_kwargs.get("desc"), "Generating Topics (dummy_path)")

            # Verify results
            self.assertEqual(len(results), 3)
            self.assertEqual(results, ["result1", "result2", "result3"])

    def test_generate_respects_max_workers(self):
        """Test that generate() passes max_workers to thread_map."""

        self.generator.config = {"max_workers": 5}
        items = ["1"]
        kwargs = {"topic": [1]}

        with patch("topic_gen.generate.thread_map") as mock_thread_map:
            mock_thread_map.return_value = ["res"]
            self.generator.generate(item_ids=items, **kwargs)

            _, call_kwargs = mock_thread_map.call_args
            self.assertEqual(call_kwargs.get("max_workers"), 5)

    def test_invoke_wrapper_handles_exception(self):
        """Test _safe_invoke returns exception on failure."""

        input_data = {"topic": 1}
        error = ValueError("oops")

        # Mock chain.invoke to raise error
        self.generator.chain.invoke.side_effect = error

        result = self.generator._safe_invoke(input_data)
        self.assertEqual(result, error)
