import unittest

from dotenv_parser import parse


class TestParse(unittest.TestCase):
    def test_basic_pairs(self):
        self.assertEqual(parse("A=1\nB=two"), {"A": "1", "B": "two"})

    def test_blank_lines_and_comments_ignored(self):
        self.assertEqual(parse("# top\n\nA=1\n   # indented comment\n"), {"A": "1"})

    def test_export_prefix(self):
        self.assertEqual(parse("export PATH=/usr/bin"), {"PATH": "/usr/bin"})

    def test_whitespace_stripped(self):
        self.assertEqual(parse("  KEY  =  value  "), {"KEY": "value"})

    def test_double_quotes_keep_hash_and_spaces(self):
        self.assertEqual(parse('MSG="hello # world"'), {"MSG": "hello # world"})

    def test_single_quotes_keep_inner_whitespace(self):
        self.assertEqual(parse("MSG='  spaced  '"), {"MSG": "  spaced  "})

    def test_unquoted_trailing_comment_dropped(self):
        self.assertEqual(parse("VAL=abc # note"), {"VAL": "abc"})

    def test_hash_without_space_is_kept(self):
        self.assertEqual(parse("URL=http://x/y#frag"), {"URL": "http://x/y#frag"})

    def test_line_without_equals_ignored(self):
        self.assertEqual(parse("JUNK\nA=1"), {"A": "1"})

    def test_empty_value(self):
        self.assertEqual(parse("K="), {"K": ""})

    def test_quoted_empty_value(self):
        self.assertEqual(parse('K=""'), {"K": ""})

    def test_later_assignment_wins(self):
        self.assertEqual(parse("A=1\nA=2"), {"A": "2"})

    def test_empty_input(self):
        self.assertEqual(parse(""), {})

    def test_value_containing_equals(self):
        self.assertEqual(parse("Q=a=b=c"), {"Q": "a=b=c"})


if __name__ == "__main__":
    unittest.main()
