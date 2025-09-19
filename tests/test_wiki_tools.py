from unittest.mock import Mock, patch
import unittest
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from simple_or_agent.tools.wiki import (
    make_wikipedia_search_tool,
    make_wikipedia_summary_tool,
    make_wikipedia_page_content_tool,
    make_wikipedia_page_section_text_tool,
    make_wikipedia_page_categories_tool,
    make_wikipedia_page_links_tool,
    make_wikipedia_geosearch_tool,
    make_wikipedia_set_language_tool
)


class TestWikiTools(unittest.TestCase):
    """Test suite for Wikipedia tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_page_data = {
            "title": "Python (programming language)",
            "content": "Python is a high-level programming language...",
            "summary": "Python is a high-level, general-purpose programming language.",
            "categories": ["Programming languages", "Python programming language"],
            "links": ["Guido van Rossum", "Object-oriented programming", "Functional programming"],
            "section_content": "Python features a dynamic type system..."
        }

    def test_wikipedia_search_tool_success(self):
        """Test successful Wikipedia search."""
        tool = make_wikipedia_search_tool()

        with patch('wikipedia.search') as mock_search:
            mock_search.return_value = [
                "Python (programming language)",
                "Monty Python",
                "Pythonidae"
            ]

            result = tool.handler({"query": "Python", "results": 5})

            self.assertEqual(result["query"], "Python")
            self.assertEqual(len(result["results"]), 3)
            self.assertIn("Python (programming language)", result["results"])
            mock_search.assert_called_once_with(query="Python", results=5)

    def test_wikipedia_search_tool_empty_query(self):
        """Test Wikipedia search with empty query."""
        tool = make_wikipedia_search_tool()

        result = tool.handler({"query": ""})
        self.assertEqual(result["error"], "empty_query")

    def test_wikipedia_search_tool_default_results(self):
        """Test Wikipedia search with default results count."""
        tool = make_wikipedia_search_tool()

        with patch('wikipedia.search') as mock_search:
            mock_search.return_value = ["Result 1", "Result 2"]

            result = tool.handler({"query": "test"})

            mock_search.assert_called_once_with(query="test", results=10)

    def test_wikipedia_summary_tool_success(self):
        """Test successful Wikipedia summary retrieval."""
        tool = make_wikipedia_summary_tool()

        with patch('wikipedia.summary') as mock_summary:
            mock_summary.return_value = "Python is a high-level, general-purpose programming language."

            result = tool.handler({"title": "Python (programming language)", "sentences": 3})

            self.assertEqual(result["title"], "Python (programming language)")
            self.assertEqual(result["summary"], "Python is a high-level, general-purpose programming language.")
            mock_summary.assert_called_once_with(title="Python (programming language)", sentences=3)

    def test_wikipedia_summary_tool_default_sentences(self):
        """Test Wikipedia summary with default sentences."""
        tool = make_wikipedia_summary_tool()

        with patch('wikipedia.summary') as mock_summary:
            mock_summary.return_value = "Summary text"

            result = tool.handler({"title": "Python"})

            mock_summary.assert_called_once_with(title="Python", sentences=0)

    def test_wikipedia_summary_tool_empty_title(self):
        """Test Wikipedia summary with empty title."""
        tool = make_wikipedia_summary_tool()

        result = tool.handler({"title": ""})
        self.assertEqual(result["error"], "empty_title")

    def test_wikipedia_page_content_tool_success(self):
        """Test successful Wikipedia page content retrieval."""
        tool = make_wikipedia_page_content_tool()

        mock_page = Mock()
        mock_page.content = "Python is a high-level programming language..."

        with patch('wikipedia.page') as mock_page_func:
            mock_page_func.return_value = mock_page

            result = tool.handler({"title": "Python (programming language)"})

            self.assertEqual(result["title"], "Python (programming language)")
            self.assertEqual(result["content"], "Python is a high-level programming language...")
            mock_page_func.assert_called_once_with(title="Python (programming language)")

    def test_wikipedia_page_section_text_tool_success(self):
        """Test successful Wikipedia section text retrieval."""
        tool = make_wikipedia_page_section_text_tool()

        mock_page = Mock()
        mock_page.section.return_value = "Python features a dynamic type system..."

        with patch('wikipedia.page') as mock_page_func:
            mock_page_func.return_value = mock_page

            result = tool.handler({
                "title": "Python (programming language)",
                "section_title": "Features"
            })

            self.assertEqual(result["title"], "Python (programming language)")
            self.assertEqual(result["section_title"], "Features")
            self.assertEqual(result["section_text"], "Python features a dynamic type system...")
            mock_page_func.assert_called_once_with(title="Python (programming language)")
            mock_page.section.assert_called_once_with("Features")

    def test_wikipedia_page_section_text_tool_not_found(self):
        """Test Wikipedia section text when section not found."""
        tool = make_wikipedia_page_section_text_tool()

        mock_page = Mock()
        mock_page.section.return_value = None

        with patch('wikipedia.page') as mock_page_func:
            mock_page_func.return_value = mock_page

            result = tool.handler({
                "title": "Python (programming language)",
                "section_title": "Nonexistent Section"
            })

            self.assertEqual(result["error"], "section_not_found")

    def test_wikipedia_page_section_text_tool_empty_inputs(self):
        """Test Wikipedia section text with empty inputs."""
        tool = make_wikipedia_page_section_text_tool()

        result = tool.handler({"title": "", "section_title": "Features"})
        self.assertEqual(result["error"], "empty_title_or_section_title")

        result = tool.handler({"title": "Python", "section_title": ""})
        self.assertEqual(result["error"], "empty_title_or_section_title")

    def test_wikipedia_page_categories_tool_success(self):
        """Test successful Wikipedia page categories retrieval."""
        tool = make_wikipedia_page_categories_tool()

        mock_page = Mock()
        mock_page.categories = ["Programming languages", "Python programming language", "High-level programming languages"]

        with patch('wikipedia.page') as mock_page_func:
            mock_page_func.return_value = mock_page

            result = tool.handler({"title": "Python (programming language)"})

            self.assertEqual(result["title"], "Python (programming language)")
            self.assertEqual(len(result["categories"]), 3)
            self.assertIn("Programming languages", result["categories"])

    def test_wikipedia_page_links_tool_success(self):
        """Test successful Wikipedia page links retrieval."""
        tool = make_wikipedia_page_links_tool()

        mock_page = Mock()
        mock_page.links = ["Guido van Rossum", "Object-oriented programming", "Functional programming"]

        with patch('wikipedia.page') as mock_page_func:
            mock_page_func.return_value = mock_page

            result = tool.handler({"title": "Python (programming language)"})

            # The tool directly returns the links list, not a dictionary
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)
            self.assertIn("Guido van Rossum", result)

    def test_wikipedia_geosearch_tool_success(self):
        """Test successful Wikipedia geosearch."""
        tool = make_wikipedia_geosearch_tool()

        with patch('wikipedia.geosearch') as mock_geosearch:
            mock_geosearch.return_value = [
                "Statue of Liberty",
                "Ellis Island",
                "One World Trade Center"
            ]

            result = tool.handler({
                "latitude": 40.6892,
                "longitude": -74.0445,
                "radius": 1000,
                "results": 5
            })

            self.assertEqual(result["latitude"], 40.6892)
            self.assertEqual(result["longitude"], -74.0445)
            self.assertEqual(result["radius"], 1000)
            self.assertEqual(len(result["results"]), 3)
            mock_geosearch.assert_called_once_with(
                latitude=40.6892,
                longitude=-74.0445,
                radius=1000,
                results=5
            )

    def test_wikipedia_geosearch_tool_default_values(self):
        """Test Wikipedia geosearch with default values."""
        tool = make_wikipedia_geosearch_tool()

        with patch('wikipedia.geosearch') as mock_geosearch:
            mock_geosearch.return_value = ["Result 1"]

            result = tool.handler({
                "latitude": 40.6892,
                "longitude": -74.0445
            })

            mock_geosearch.assert_called_once_with(
                latitude=40.6892,
                longitude=-74.0445,
                radius=1000,
                results=10
            )

    def test_wikipedia_geosearch_tool_invalid_coordinates(self):
        """Test Wikipedia geosearch with invalid coordinates."""
        tool = make_wikipedia_geosearch_tool()

        # Invalid latitude
        result = tool.handler({"latitude": "invalid", "longitude": -74.0445})
        self.assertEqual(result["error"], "invalid_latitude_or_longitude")

        # Invalid longitude
        result = tool.handler({"latitude": 40.6892, "longitude": "invalid"})
        self.assertEqual(result["error"], "invalid_latitude_or_longitude")

        # Missing coordinates
        result = tool.handler({"latitude": 40.6892})
        self.assertEqual(result["error"], "invalid_latitude_or_longitude")

    def test_wikipedia_geosearch_tool_radius_bounds(self):
        """Test Wikipedia geosearch radius validation."""
        tool = make_wikipedia_geosearch_tool()

        # Radius too small
        result = tool.handler({
            "latitude": 40.6892,
            "longitude": -74.0445,
            "radius": 5
        })
        self.assertEqual(result["error"], "radius_out_of_bounds")

        # Radius too large
        result = tool.handler({
            "latitude": 40.6892,
            "longitude": -74.0445,
            "radius": 20000
        })
        self.assertEqual(result["error"], "radius_out_of_bounds")

    def test_wikipedia_set_language_tool_success(self):
        """Test successful Wikipedia language setting."""
        tool = make_wikipedia_set_language_tool()

        with patch('wikipedia.set_lang') as mock_set_lang:
            result = tool.handler({"prefix": "es"})

            self.assertEqual(result["message"], "Wikipedia language set to 'es'")
            mock_set_lang.assert_called_once_with("es")

    def test_wikipedia_set_language_tool_empty_prefix(self):
        """Test Wikipedia language setting with empty prefix."""
        tool = make_wikipedia_set_language_tool()

        result = tool.handler({"prefix": ""})
        self.assertEqual(result["error"], "empty_prefix")

    def test_wikipedia_tools_error_handling(self):
        """Test error handling across all Wikipedia tools."""
        # Test Wikipedia API errors
        tools_to_test = [
            (make_wikipedia_search_tool, {"query": "test"}),
            (make_wikipedia_summary_tool, {"title": "test"}),
            (make_wikipedia_page_content_tool, {"title": "test"}),
            (make_wikipedia_page_section_text_tool, {"title": "test", "section_title": "test"}),
            (make_wikipedia_page_categories_tool, {"title": "test"}),
            (make_wikipedia_set_language_tool, {"prefix": "es"}),
        ]

        for tool_func, args in tools_to_test:
            with self.subTest(tool=tool_func.__name__), \
                 patch('wikipedia.search' if 'search' in tool_func.__name__ else
                       'wikipedia.summary' if 'summary' in tool_func.__name__ else
                       'wikipedia.page' if 'page' in tool_func.__name__ else
                       'wikipedia.set_lang') as mock_func:

                mock_func.side_effect = Exception("API Error")

                tool = tool_func()
                result = tool.handler(args)
                self.assertIn("error", result)

        # Test page links tool separately since it uses a lambda handler
        with patch('wikipedia.page') as mock_page_func:
            mock_page_func.side_effect = Exception("API Error")

            tool = make_wikipedia_page_links_tool()
            # The lambda handler will raise an exception, which we need to catch
            with self.assertRaises(Exception):
                tool.handler({"title": "test"})

        # Test geosearch tool separately - it doesn't raise exceptions but returns empty results
        with patch('wikipedia.geosearch') as mock_geosearch:
            mock_geosearch.side_effect = Exception("Geosearch API Error")

            tool = make_wikipedia_geosearch_tool()
            result = tool.handler({"latitude": 40.0, "longitude": -74.0})
            # Geosearch should handle the exception and return an error
            self.assertIn("error", result)

    def test_wikipedia_tools_missing_required_parameters(self):
        """Test missing required parameters across all tools."""
        tools_with_required_params = [
            (make_wikipedia_search_tool, ["query"]),
            (make_wikipedia_summary_tool, ["title"]),
            (make_wikipedia_page_content_tool, ["title"]),
            (make_wikipedia_page_section_text_tool, ["title", "section_title"]),
            (make_wikipedia_page_categories_tool, ["title"]),
            (make_wikipedia_geosearch_tool, ["latitude", "longitude"]),
            (make_wikipedia_set_language_tool, ["prefix"]),
        ]

        for tool_func, _ in tools_with_required_params:
            with self.subTest(tool=tool_func.__name__):
                tool = tool_func()
                result = tool.handler({})
                self.assertIn("error", result)

        # Test page links tool separately since it uses a lambda handler
        tool = make_wikipedia_page_links_tool()
        # The lambda handler will raise a KeyError for missing 'title'
        with self.assertRaises(KeyError):
            tool.handler({})

    def test_wikipedia_page_links_tool_handler_consistency(self):
        """Test that page links tool uses consistent handler pattern."""
        tool = make_wikipedia_page_links_tool()

        mock_page = Mock()
        mock_page.links = ["Link 1", "Link 2"]

        with patch('wikipedia.page') as mock_page_func:
            mock_page_func.return_value = mock_page

            result = tool.handler({"title": "Test Page"})

            # The tool directly returns the links list
            self.assertIsInstance(result, list)
            self.assertEqual(result, ["Link 1", "Link 2"])

    def test_wikipedia_tools_parameter_types(self):
        """Test parameter type validation."""
        # Test geosearch with string coordinates (should convert to float)
        tool = make_wikipedia_geosearch_tool()

        with patch('wikipedia.geosearch') as mock_geosearch:
            mock_geosearch.return_value = ["Result"]

            result = tool.handler({
                "latitude": "40.6892",
                "longitude": "-74.0445"
            })

            # Should convert strings to floats successfully
            self.assertIn("results", result)


if __name__ == "__main__":
    # Run the tests
    unittest.main()