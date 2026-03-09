"""
Contained tests for the data curation pipeline.
No network calls, no HuggingFace push, no full dataset loading.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.pricer.items import Item
from src.pricer.pipeline.data_curation_orchestration import (
    deduplicate,
    resample,
    run,
    parse_args,
    DATASET_NAMES,
    DEFAULT_SIZE,
    DEFAULT_USERNAME,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


def make_item(
    title: str, full: str, price: float, category: str = "Electronics"
) -> Item:
    return Item(title=title, category=category, price=price, full=full)


# ---------------------------------------------------------------------------
# 1. Module-level constants
# ---------------------------------------------------------------------------
class TestConstants(unittest.TestCase):
    def test_dataset_names_non_empty(self):
        self.assertGreater(len(DATASET_NAMES), 0)

    def test_default_size_positive(self):
        self.assertGreater(DEFAULT_SIZE, 0)

    def test_splits_are_ordered(self):
        self.assertLess(TRAIN_SPLIT, VAL_SPLIT)


# ---------------------------------------------------------------------------
# 2. Deduplication logic
# ---------------------------------------------------------------------------
class TestDeduplicate(unittest.TestCase):
    def _make_items(self):
        return [
            make_item("Widget A", "Full text A", 9.99),
            make_item("Widget A", "Full text A", 9.99),  # exact duplicate
            make_item("Widget B", "Full text A", 19.99),  # same full, different title
            make_item("Widget C", "Full text C", 29.99),  # unique
        ]

    def test_removes_duplicate_titles(self):
        items = self._make_items()
        result = deduplicate(items)
        titles = [i.title for i in result]
        self.assertEqual(len(titles), len(set(titles)), "Duplicate titles remain")

    def test_removes_duplicate_full_text(self):
        items = self._make_items()
        result = deduplicate(items)
        fulls = [i.full for i in result]
        self.assertEqual(len(fulls), len(set(fulls)), "Duplicate full texts remain")

    def test_unique_items_survive(self):
        items = [
            make_item("Alpha", "Text alpha", 5.0),
            make_item("Beta", "Text beta", 10.0),
        ]
        result = deduplicate(items)
        self.assertEqual(len(result), 2)

    def test_empty_list(self):
        self.assertEqual(deduplicate([]), [])

    def test_single_item(self):
        items = [make_item("Solo", "Only one", 1.0)]
        result = deduplicate(items)
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# 3. Resampling logic
# ---------------------------------------------------------------------------
class TestResample(unittest.TestCase):
    def _make_pool(self, n=200):
        items = []
        categories = [
            "Automotive",
            "Electronics",
            "Tools_and_Home_Improvement",
            "Appliances",
        ]
        for i in range(n):
            items.append(
                make_item(
                    title=f"Product {i}",
                    full=f"Full text for product number {i}",
                    price=float((i % 100) + 1),
                    category=categories[i % len(categories)],
                )
            )
        return items

    def test_output_size(self):
        pool = self._make_pool(200)
        result = resample(pool, size=50)
        self.assertEqual(len(result), 50)

    def test_all_items_come_from_pool(self):
        pool = self._make_pool(200)
        result = resample(pool, size=50)
        pool_titles = {i.title for i in pool}
        for item in result:
            self.assertIn(item.title, pool_titles)

    def test_no_duplicates_in_result(self):
        pool = self._make_pool(200)
        result = resample(pool, size=50)
        titles = [i.title for i in result]
        self.assertEqual(
            len(titles), len(set(titles)), "Resample produced duplicate items"
        )

    def test_deterministic_with_seed(self):
        pool = self._make_pool(200)
        r1 = resample(pool, size=50)
        r2 = resample(pool, size=50)
        self.assertEqual([i.title for i in r1], [i.title for i in r2])

    def test_size_equals_pool(self):
        # Pool size == requested size: all items are selected (with weight clipping)
        pool = self._make_pool(50)
        result = resample(pool, size=50)
        self.assertEqual(len(result), 50)
        # Every item in the pool should appear exactly once
        result_titles = sorted(i.title for i in result)
        pool_titles = sorted(i.title for i in pool)
        self.assertEqual(result_titles, pool_titles)


# ---------------------------------------------------------------------------
# 4. CLI argument parsing
# ---------------------------------------------------------------------------
class TestArgParsing(unittest.TestCase):
    def test_defaults(self):
        with patch("sys.argv", ["prog"]):
            args = parse_args()
        self.assertEqual(args.size, DEFAULT_SIZE)
        self.assertFalse(args.push_to_hub)
        self.assertEqual(args.username, DEFAULT_USERNAME)

    def test_push_flag(self):
        with patch("sys.argv", ["prog", "--push-to-hub"]):
            args = parse_args()
        self.assertTrue(args.push_to_hub)

    def test_custom_size(self):
        with patch("sys.argv", ["prog", "--size", "1000"]):
            args = parse_args()
        self.assertEqual(args.size, 1000)

    def test_custom_username(self):
        with patch("sys.argv", ["prog", "--username", "testuser"]):
            args = parse_args()
        self.assertEqual(args.username, "testuser")


# ---------------------------------------------------------------------------
# 5. run() orchestration (mocked I/O)
# ---------------------------------------------------------------------------
class TestRunOrchestration(unittest.TestCase):
    def _make_pool(self, n=300):
        items = []
        categories = ["Automotive", "Electronics", "Tools_and_Home_Improvement"]
        for i in range(n):
            items.append(
                make_item(
                    title=f"Unique Item {i}",
                    full=f"Unique full text for item {i} with enough content",
                    price=float((i % 99) + 1),
                    category=categories[i % len(categories)],
                )
            )
        return items

    @patch("src.pricer.pipeline.data_curation_orchestration.load_dotenv")
    @patch("src.pricer.pipeline.data_curation_orchestration.load_all")
    def test_run_no_push_returns_sample(self, mock_load_all, mock_dotenv):
        pool = self._make_pool(300)
        mock_load_all.return_value = pool

        result = run(size=50, push=False)

        self.assertEqual(len(result), 50)
        mock_load_all.assert_called_once_with(DATASET_NAMES)

    @patch("src.pricer.pipeline.data_curation_orchestration.load_dotenv")
    @patch("src.pricer.pipeline.data_curation_orchestration.push_to_hub")
    @patch("src.pricer.pipeline.data_curation_orchestration.load_all")
    def test_run_with_push_calls_push(self, mock_load_all, mock_push, mock_dotenv):
        pool = self._make_pool(300)
        mock_load_all.return_value = pool

        run(size=50, push=True, username="testuser")

        mock_push.assert_called_once()
        call_args = mock_push.call_args
        self.assertEqual(call_args[1]["username"], "testuser")

    @patch("src.pricer.pipeline.data_curation_orchestration.load_dotenv")
    @patch("src.pricer.pipeline.data_curation_orchestration.push_to_hub")
    @patch("src.pricer.pipeline.data_curation_orchestration.load_all")
    def test_run_no_push_never_calls_push(self, mock_load_all, mock_push, mock_dotenv):
        mock_load_all.return_value = self._make_pool(300)
        run(size=50, push=False)
        mock_push.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Item.push_to_hub split sizes
# ---------------------------------------------------------------------------
class TestSplitSizes(unittest.TestCase):
    def test_split_indices_cover_full_sample(self):
        sample_size = 100
        train = list(range(sample_size))[:80]
        val = list(range(sample_size))[80:90]
        test = list(range(sample_size))[90:]
        self.assertEqual(len(train) + len(val) + len(test), sample_size)

    def test_actual_split_constants(self):
        # Verify the notebook's intended split proportions are preserved
        self.assertEqual(TRAIN_SPLIT, 800_000)
        self.assertEqual(VAL_SPLIT, 810_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
