from unittest.mock import Mock, patch
import unittest
import sys
import os

# Import for integration tests - mark as optional in case ddgs is not available
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from simple_or_agent.tools.web_search_v2 import (
    make_duckduckgo_text_search_tool,
    make_duckduckgo_news_search_tool,
    make_duckduckgo_image_search_tool,
    make_duckduckgo_answers_search_tool,
    make_duckduckgo_maps_search_tool
)


class TestWebSearchTools(unittest.TestCase):
    """Test suite for web search tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_text_results = [
            {"title": "Python Programming", "href": "https://example.com/python", "body": "Python is a programming language"},
            {"title": "Python Tutorial", "href": "https://example.com/tutorial", "body": "Learn Python programming"}
        ]

        self.mock_news_results = [
            {"title": "Latest Python Update", "href": "https://example.com/news1", "body": "Python 3.12 released"},
            {"title": "Python in AI", "href": "https://example.com/news2", "body": "Python dominates AI field"}
        ]

        self.mock_image_results = [
            {"title": "Python Logo", "image": "https://example.com/python.png", "thumbnail": "https://example.com/thumb.png"},
            {"title": "Python Code", "image": "https://example.com/code.png", "thumbnail": "https://example.com/code_thumb.png"}
        ]

        self.mock_answer_results = [
            {"text": "Python is a high-level programming language", "topic": "Python"}
        ]

        self.mock_maps_results = [
            {"title": "Python Institute", "address": "123 Code St", "phone": "555-1234"},
            {"title": "Python Cafe", "address": "456 Programming Ave", "phone": "555-5678"}
        ]

    def test_duckduckgo_text_search_tool_success(self):
        """Test successful DuckDuckGo text search."""
        tool = make_duckduckgo_text_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.text.return_value = self.mock_text_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({
                "keywords": "Python programming",
                "region": "us-en",
                "timelimit": "w",
                "max_results": 5
            })

            self.assertIn("results", result)
            self.assertEqual(len(result["results"]), 2)
            mock_ddgs.text.assert_called_once_with(
                "Python programming",
                region="us-en",
                timelimit="w",
                max_results=5
            )

    def test_duckduckgo_text_search_tool_defaults(self):
        """Test DuckDuckGo text search with default parameters."""
        tool = make_duckduckgo_text_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.text.return_value = self.mock_text_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "Python"})

            mock_ddgs.text.assert_called_once_with(
                "Python",
                region="wt-wt",
                timelimit=None,
                max_results=10
            )

    def test_duckduckgo_text_search_tool_no_results(self):
        """Test DuckDuckGo text search with no results."""
        tool = make_duckduckgo_text_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.text.return_value = []

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "nonexistent"})

            self.assertEqual(result["results"], "No results found.")

    def test_duckduckgo_news_search_tool_success(self):
        """Test successful DuckDuckGo news search."""
        tool = make_duckduckgo_news_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.news.return_value = self.mock_news_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({
                "keywords": "Python news",
                "region": "uk-en",
                "timelimit": "d",
                "max_results": 8
            })

            self.assertIn("results", result)
            self.assertEqual(len(result["results"]), 2)
            mock_ddgs.news.assert_called_once_with(
                "Python news",
                region="uk-en",
                timelimit="d",
                max_results=8
            )

    def test_duckduckgo_news_search_tool_defaults(self):
        """Test DuckDuckGo news search with default parameters."""
        tool = make_duckduckgo_news_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.news.return_value = self.mock_news_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "Python"})

            mock_ddgs.news.assert_called_once_with(
                "Python",
                region="wt-wt",
                timelimit=None,
                max_results=15
            )

    def test_duckduckgo_news_search_tool_no_results(self):
        """Test DuckDuckGo news search with no results."""
        tool = make_duckduckgo_news_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.news.return_value = []

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "old news"})

            self.assertEqual(result["results"], "No news found.")

    def test_duckduckgo_image_search_tool_success(self):
        """Test successful DuckDuckGo image search."""
        tool = make_duckduckgo_image_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.images.return_value = self.mock_image_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({
                "keywords": "Python logo",
                "region": "us-en",
                "size": "Large",
                "max_results": 3
            })

            self.assertIn("results", result)
            self.assertEqual(len(result["results"]), 2)
            mock_ddgs.images.assert_called_once_with(
                "Python logo",
                region="us-en",
                size="Large",
                max_results=3
            )

    def test_duckduckgo_image_search_tool_defaults(self):
        """Test DuckDuckGo image search with default parameters."""
        tool = make_duckduckgo_image_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.images.return_value = self.mock_image_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "Python"})

            mock_ddgs.images.assert_called_once_with(
                "Python",
                region="wt-wt",
                size=None,
                max_results=10
            )

    def test_duckduckgo_image_search_tool_no_results(self):
        """Test DuckDuckGo image search with no results."""
        tool = make_duckduckgo_image_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.images.return_value = []

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "rare image"})

            self.assertEqual(result["results"], "No images found.")

    def test_duckduckgo_answers_search_tool_success(self):
        """Test successful DuckDuckGo answers search."""
        tool = make_duckduckgo_answers_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.text.return_value = self.mock_answer_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "What is Python?"})

            self.assertIn("results", result)
            self.assertEqual(len(result["results"]), 1)
            mock_ddgs.text.assert_called_once_with("What is Python?", max_results=3)

    def test_duckduckgo_answers_search_tool_no_results(self):
        """Test DuckDuckGo answers search with no results."""
        tool = make_duckduckgo_answers_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.text.return_value = []

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "unanswerable question"})

            self.assertEqual(result["results"], "No direct answer found.")

    def test_duckduckgo_maps_search_tool_success(self):
        """Test successful DuckDuckGo maps search."""
        tool = make_duckduckgo_maps_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.text.return_value = self.mock_maps_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({
                "keywords": "Python training",
                "place": "San Francisco, CA",
                "max_results": 3
            })

            self.assertIn("results", result)
            self.assertEqual(len(result["results"]), 2)
            mock_ddgs.text.assert_called_once_with(
                "Python training San Francisco, CA",
                max_results=3
            )

    def test_duckduckgo_maps_search_tool_defaults(self):
        """Test DuckDuckGo maps search with default parameters."""
        tool = make_duckduckgo_maps_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.text.return_value = self.mock_maps_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "Python"})

            mock_ddgs.text.assert_called_once_with(
                "Python",
                max_results=5
            )

    def test_duckduckgo_maps_search_tool_no_results(self):
        """Test DuckDuckGo maps search with no results."""
        tool = make_duckduckgo_maps_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.text.return_value = []

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "nonexistent location"})

            self.assertEqual(result["results"], "No locations found.")

    def test_web_search_tools_error_handling(self):
        """Test error handling across all web search tools."""
        tools_to_test = [
            (make_duckduckgo_text_search_tool, {"keywords": "test"}),
            (make_duckduckgo_news_search_tool, {"keywords": "test"}),
            (make_duckduckgo_image_search_tool, {"keywords": "test"}),
            (make_duckduckgo_answers_search_tool, {"keywords": "test"}),
            (make_duckduckgo_maps_search_tool, {"keywords": "test"}),
        ]

        for tool_func, args in tools_to_test:
            with self.subTest(tool=tool_func.__name__):
                mock_ddgs = Mock()
                mock_ddgs.text.side_effect = Exception("API Error")
                mock_ddgs.news.side_effect = Exception("API Error")
                mock_ddgs.images.side_effect = Exception("API Error")
                # answers now uses text method
                # maps now uses text method

                with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
                    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

                    tool = tool_func()
                    result = tool.handler(args)
                    self.assertIn("error", result)

    def test_web_search_tools_missing_required_parameters(self):
        """Test missing required parameters across all web search tools."""
        tools_to_test = [
            make_duckduckgo_text_search_tool,
            make_duckduckgo_news_search_tool,
            make_duckduckgo_image_search_tool,
            make_duckduckgo_answers_search_tool,
            make_duckduckgo_maps_search_tool,
        ]

        for tool_func in tools_to_test:
            with self.subTest(tool=tool_func.__name__):
                tool = tool_func()
                result = tool.handler({})
                self.assertIn("error", result)

    def test_duckduckgo_text_search_tool_parameter_validation(self):
        """Test parameter validation for text search tool."""
        tool = make_duckduckgo_text_search_tool()

        # Test with empty keywords
        result = tool.handler({"keywords": ""})
        self.assertIn("error", result)

        # Test with invalid max_results (should be handled gracefully)
        mock_ddgs = Mock()
        mock_ddgs.text.return_value = []

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "test", "max_results": -1})
            # Should still work, DDGS handles validation
            self.assertIn("results", result)

    def test_duckduckgo_news_search_tool_parameter_validation(self):
        """Test parameter validation for news search tool."""
        tool = make_duckduckgo_news_search_tool()

        # Test with invalid timelimit (should be handled gracefully)
        mock_ddgs = Mock()
        mock_ddgs.news.return_value = []

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "test", "timelimit": "invalid"})
            # Should still work, DDGS handles validation
            self.assertIn("results", result)

    def test_duckduckgo_image_search_tool_parameter_validation(self):
        """Test parameter validation for image search tool."""
        tool = make_duckduckgo_image_search_tool()

        # Test with invalid size (should be handled gracefully)
        mock_ddgs = Mock()
        mock_ddgs.images.return_value = []

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "test", "size": "invalid"})
            # Should still work, DDGS handles validation
            self.assertIn("results", result)

    def test_ddgs_context_manager_handling(self):
        """Test that DDGS is properly used as a context manager."""
        tool = make_duckduckgo_text_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.text.return_value = self.mock_text_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "test"})

            # Verify that __enter__ was called
            mock_ddgs_class.return_value.__enter__.assert_called_once()

    def test_search_results_structure(self):
        """Test that search results have the expected structure."""
        tool = make_duckduckgo_text_search_tool()

        mock_ddgs = Mock()
        mock_ddgs.text.return_value = self.mock_text_results

        with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

            result = tool.handler({"keywords": "test"})

            # Should return a dictionary with 'results' key
            self.assertIsInstance(result, dict)
            self.assertIn("results", result)

            # Results should be a list when found
            self.assertIsInstance(result["results"], list)

    def test_different_search_types_return_different_result_types(self):
        """Test that different search types handle results appropriately."""
        tools_and_results = [
            (make_duckduckgo_text_search_tool, self.mock_text_results),
            (make_duckduckgo_news_search_tool, self.mock_news_results),
            (make_duckduckgo_image_search_tool, self.mock_image_results),
            (make_duckduckgo_answers_search_tool, self.mock_answer_results),
            (make_duckduckgo_maps_search_tool, self.mock_maps_results),
        ]

        for tool_func, mock_results in tools_and_results:
            with self.subTest(tool=tool_func.__name__):
                mock_ddgs = Mock()
                # Set the appropriate method for each tool
                if "text" in tool_func.__name__:
                    mock_ddgs.text.return_value = mock_results
                elif "news" in tool_func.__name__:
                    mock_ddgs.news.return_value = mock_results
                elif "image" in tool_func.__name__:
                    mock_ddgs.images.return_value = mock_results
                elif "answers" in tool_func.__name__:
                    # answers now uses text method
                    mock_ddgs.text.return_value = mock_results
                elif "maps" in tool_func.__name__:
                    # maps now uses text method
                    mock_ddgs.text.return_value = mock_results

                with patch('simple_or_agent.tools.web_search_v2.DDGS') as mock_ddgs_class:
                    mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs

                    tool = tool_func()
                    result = tool.handler({"keywords": "test"})

                    self.assertIn("results", result)
                    if mock_results:
                        self.assertEqual(len(result["results"]), len(mock_results))


class TestWebSearchToolsIntegration(unittest.TestCase):
    """Integration tests for web search tools with actual API calls."""

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_duckduckgo_text_search_integration(self):
        """Test actual DuckDuckGo text search API."""
        tool = make_duckduckgo_text_search_tool()

        result = tool.handler({
            "keywords": "Python programming language",
            "max_results": 3
        })

        # Should return results (not an error)
        self.assertNotIn("error", result)
        self.assertIn("results", result)

        # Should not be the "no results" message
        self.assertNotEqual(result["results"], "No results found.")

        # Results should be a list
        self.assertIsInstance(result["results"], list)
        self.assertGreater(len(result["results"]), 0)

        # Each result should have basic structure
        for result_item in result["results"]:
            self.assertIsInstance(result_item, dict)
            # Text search results typically have title, href, body
            if "title" in result_item:
                self.assertIsInstance(result_item["title"], str)
                self.assertTrue(len(result_item["title"]) > 0)
            if "href" in result_item:
                self.assertIsInstance(result_item["href"], str)
                self.assertTrue(result_item["href"].startswith("http"))

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_duckduckgo_news_search_integration(self):
        """Test actual DuckDuckGo news search API."""
        tool = make_duckduckgo_news_search_tool()

        result = tool.handler({
            "keywords": "artificial intelligence",
            "max_results": 3
        })

        # Should return results (not an error)
        self.assertNotIn("error", result)
        self.assertIn("results", result)

        # Should not be the "no results" message
        self.assertNotEqual(result["results"], "No news found.")

        # Results should be a list
        self.assertIsInstance(result["results"], list)
        self.assertGreater(len(result["results"]), 0)

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_duckduckgo_image_search_integration(self):
        """Test actual DuckDuckGo image search API."""
        tool = make_duckduckgo_image_search_tool()

        result = tool.handler({
            "keywords": "cat",
            "max_results": 3
        })

        # Should return results (not an error)
        self.assertNotIn("error", result)
        self.assertIn("results", result)

        # Should not be the "no results" message
        self.assertNotEqual(result["results"], "No images found.")

        # Results should be a list
        self.assertIsInstance(result["results"], list)
        self.assertGreater(len(result["results"]), 0)

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_duckduckgo_answers_search_integration(self):
        """Test actual DuckDuckGo answers search API."""
        tool = make_duckduckgo_answers_search_tool()

        result = tool.handler({
            "keywords": "capital of France"
        })

        # Should return results (not an error)
        self.assertNotIn("error", result)
        self.assertIn("results", result)

        # Results should be a list or string
        if isinstance(result["results"], list):
            # If we get results, they should have content
            if len(result["results"]) > 0:
                first_result = result["results"][0]
                self.assertIsInstance(first_result, dict)
                # Answers typically have text
                if "text" in first_result:
                    self.assertIsInstance(first_result["text"], str)
                    self.assertTrue(len(first_result["text"]) > 0)

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_duckduckgo_maps_search_integration(self):
        """Test actual DuckDuckGo maps search API."""
        tool = make_duckduckgo_maps_search_tool()

        result = tool.handler({
            "keywords": "Eiffel Tower",
            "max_results": 3
        })

        # Should return results (not an error)
        self.assertNotIn("error", result)
        self.assertIn("results", result)

        # Should not be the "no results" message
        self.assertNotEqual(result["results"], "No locations found.")

        # Results should be a list
        self.assertIsInstance(result["results"], list)
        self.assertGreater(len(result["results"]), 0)

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_duckduckgo_text_search_with_region_integration(self):
        """Test text search with region parameter."""
        tool = make_duckduckgo_text_search_tool()

        result = tool.handler({
            "keywords": "weather",
            "region": "us-en",
            "max_results": 2
        })

        # Should return results
        self.assertNotIn("error", result)
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_duckduckgo_news_search_with_time_limit_integration(self):
        """Test news search with time limit parameter."""
        tool = make_duckduckgo_news_search_tool()

        result = tool.handler({
            "keywords": "technology",
            "timelimit": "d",  # Day limit
            "max_results": 2
        })

        # Should return results
        self.assertNotIn("error", result)
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_duckduckgo_image_search_with_size_integration(self):
        """Test image search with size parameter."""
        tool = make_duckduckgo_image_search_tool()

        result = tool.handler({
            "keywords": "nature",
            "size": "Large",
            "max_results": 2
        })

        # Should return results
        self.assertNotIn("error", result)
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_duckduckgo_maps_search_with_place_integration(self):
        """Test maps search with place parameter."""
        tool = make_duckduckgo_maps_search_tool()

        result = tool.handler({
            "keywords": "coffee shops",
            "place": "New York",
            "max_results": 2
        })

        # Should return results
        self.assertNotIn("error", result)
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_real_search_performance(self):
        """Test that real searches complete within reasonable time."""
        tool = make_duckduckgo_text_search_tool()

        import time
        start_time = time.time()

        result = tool.handler({
            "keywords": "machine learning",
            "max_results": 5
        })

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within 10 seconds
        self.assertLess(duration, 10, "Search took too long")

        # Should return valid results
        self.assertNotIn("error", result)
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_search_with_rare_query(self):
        """Test search with a rare query that might return few results."""
        tool = make_duckduckgo_text_search_tool()

        # Use a very specific query
        result = tool.handler({
            "keywords": "quantum computing 2024 specific algorithm",
            "max_results": 10
        })

        # Should not return an error even for rare queries
        self.assertNotIn("error", result)
        self.assertIn("results", result)

        # Results could be empty or have few items, but should handle gracefully
        if isinstance(result["results"], list):
            self.assertGreaterEqual(len(result["results"]), 0)
        else:
            # Should be the "no results" message
            self.assertIsInstance(result["results"], str)

    @unittest.skipUnless(DDGS_AVAILABLE, "DDGS library not available")
    def test_all_search_types_return_valid_data(self):
        """Test that all search types return valid data structures."""
        search_tools = [
            (make_duckduckgo_text_search_tool(), {"keywords": "test", "max_results": 2}),
            (make_duckduckgo_news_search_tool(), {"keywords": "test", "max_results": 2}),
            (make_duckduckgo_image_search_tool(), {"keywords": "test", "max_results": 2}),
            (make_duckduckgo_answers_search_tool(), {"keywords": "test"}),
            (make_duckduckgo_maps_search_tool(), {"keywords": "test", "max_results": 2}),
        ]

        for tool, args in search_tools:
            with self.subTest(tool=tool.name):
                result = tool.handler(args)

                # Should not return errors
                self.assertNotIn("error", result, f"{tool.name} returned an error")

                # Should have results key
                self.assertIn("results", result, f"{tool.name} missing results key")

                # Results should not be empty unless it's the "no results" message
                if isinstance(result["results"], list):
                    self.assertIsInstance(result["results"], list, f"{tool.name} results not a list")
                else:
                    self.assertIsInstance(result["results"], str, f"{tool.name} results not a string")


if __name__ == "__main__":
    # Run the tests
    unittest.main()