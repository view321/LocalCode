import unittest

from lru import LruCache


class TestLruCache(unittest.TestCase):
    def test_capacity_validation(self):
        with self.assertRaises(ValueError):
            LruCache(0)

    def test_basic_put_get(self):
        c = LruCache(2)
        c.put("a", 1)
        self.assertEqual(c.get("a"), 1)
        self.assertIsNone(c.get("missing"))
        self.assertEqual(c.get("missing", 42), 42)
        self.assertEqual(len(c), 1)

    def test_evicts_least_recently_used(self):
        c = LruCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)  # evicts "a" (oldest)
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.get("b"), 2)
        self.assertEqual(c.get("c"), 3)
        self.assertEqual(len(c), 2)

    def test_get_refreshes_recency(self):
        c = LruCache(2)
        c.put("a", 1)
        c.put("b", 2)
        self.assertEqual(c.get("a"), 1)  # "a" is now the most recent
        c.put("c", 3)                    # evicts "b", not "a"
        self.assertEqual(c.get("a"), 1)
        self.assertIsNone(c.get("b"))
        self.assertEqual(c.get("c"), 3)

    def test_put_existing_refreshes_and_updates(self):
        c = LruCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("a", 10)  # update refreshes "a"
        c.put("c", 3)   # evicts "b"
        self.assertEqual(c.get("a"), 10)
        self.assertIsNone(c.get("b"))
        self.assertEqual(len(c), 2)

    def test_capacity_one_churn(self):
        c = LruCache(1)
        c.put("a", 1)
        c.put("b", 2)
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.get("b"), 2)
        self.assertEqual(len(c), 1)

    def test_long_sequence_keeps_size_bounded(self):
        c = LruCache(3)
        for i in range(50):
            c.put(i, i * i)
        self.assertEqual(len(c), 3)
        self.assertEqual(c.get(49), 49 * 49)
        self.assertEqual(c.get(47), 47 * 47)
        self.assertIsNone(c.get(0))


if __name__ == "__main__":
    unittest.main()
