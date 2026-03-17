from __future__ import annotations

import unittest

from src.agents.paper_parsing_agent import PaperParsingAgent


class PaperParsingAgentTests(unittest.TestCase):
    def test_prepare_text_removes_references_and_joins_wrapped_lines(self) -> None:
        raw_text = (
            "[Page 1]\n"
            "A Journal Header\n"
            "1 Introduction\n"
            "This is a para-\n"
            "graph about boiling point prediction.\n"
            "\n"
            "References\n"
            "[1] Example citation\n"
        )

        prepared = PaperParsingAgent._prepare_text(raw_text)

        self.assertIn("1 Introduction", prepared)
        self.assertIn("This is a paragraph about boiling point prediction.", prepared)
        self.assertNotIn("References", prepared)
        self.assertNotIn("Example citation", prepared)

    def test_rule_based_markdown_preserves_page_markers_and_headings(self) -> None:
        prepared = (
            "[Page 2]\n"
            "2 Methods\n"
            "We used Random Forest.\n"
            "- first item\n"
        )

        markdown = PaperParsingAgent._rule_based_markdown(prepared)

        self.assertIn("<!-- [Page 2] -->", markdown)
        self.assertIn("## 2 Methods", markdown)
        self.assertIn("We used Random Forest.", markdown)
        self.assertIn("- first item", markdown)


if __name__ == "__main__":
    unittest.main()
