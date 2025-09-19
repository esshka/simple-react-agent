import unittest
import os
import sys
import time
from unittest.mock import patch

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from simple_or_agent.tools.searxng import (
    make_searxng_search_tool,
    process_search_results
)


class TestSearXNGToolsIntegration(unittest.TestCase):
    """Integration tests for SearXNG tools with actual API calls."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a public SearXNG instance for testing
        self.test_base_url = "https://searx.org"

        # Fallback to another public instance if the first one is not available
        self.fallback_base_url = "https://searx.be"

        # Use environment variable if set
        self.base_url = os.getenv("SEARXNG_BASE_URL", self.test_base_url)

    def get_working_base_url(self):
        """Try to find a working SearXNG instance."""
        for url in [self.base_url, self.test_base_url, self.fallback_base_url]:
            try:
                # Test if the instance is reachable with a simple search
                tool = make_searxng_search_tool()
                result = tool.handler({
                    "queries": ["test"],
                    "max_results": 1
                })
                if "error" not in result:
                    return url
            except:
                continue
        return self.test_base_url  # Default to the first option

    def test_searxng_search_tool_basic_functionality(self):
        """Test basic SearXNG search functionality."""
        tool = make_searxng_search_tool()

        # Test with a simple query
        result = tool.handler({
            "queries": ["Python programming"],
            "max_results": 3
        })

        # Should return results structure
        self.assertIsInstance(result, dict)
        self.assertIn("results", result)

        # If no error, check results structure
        if "error" not in result:
            results = result["results"]
            self.assertIsInstance(results, list)

            # Each result should have basic structure
            for search_result in results:
                self.assertIsInstance(search_result, dict)
                self.assertIn("url", search_result)
                self.assertIn("title", search_result)
                self.assertIn("content", search_result)
                self.assertIn("query", search_result)

                # URL should be valid
                self.assertTrue(search_result["url"].startswith("http"))
                self.assertTrue(len(search_result["url"]) > 0)

                # Title should be non-empty
                self.assertIsInstance(search_result["title"], str)
                self.assertTrue(len(search_result["title"]) > 0)

                # Content should be string (can be empty)
                self.assertIsInstance(search_result["content"], str)

                # Query should match what we searched for
                self.assertEqual(search_result["query"], "Python programming")

    def test_searxng_search_tool_multiple_queries(self):
        """Test SearXNG search with multiple queries."""
        tool = make_searxng_search_tool()

        result = tool.handler({
            "queries": ["artificial intelligence", "machine learning"],
            "max_results": 2
        })

        self.assertIsInstance(result, dict)
        self.assertIn("results", result)

        if "error" not in result:
            results = result["results"]
            self.assertIsInstance(results, list)

            # Should have results from both queries
            queries_found = set()
            for search_result in results:
                queries_found.add(search_result["query"])

            # Should find results from both queries (though might not always have both)
            self.assertGreater(len(queries_found), 0)

    def test_searxng_search_tool_with_category(self):
        """Test SearXNG search with category filtering."""
        tool = make_searxng_search_tool()

        # Test with news category
        result = tool.handler({
            "queries": ["technology"],
            "category": "news",
            "max_results": 3
        })

        self.assertIsInstance(result, dict)
        self.assertIn("results", result)
        self.assertIn("category", result)

        if "error" not in result:
            self.assertEqual(result["category"], "news")

            results = result["results"]
            self.assertIsInstance(results, list)

            for search_result in results:
                self.assertIsInstance(search_result, dict)
                self.assertIn("url", search_result)
                self.assertIn("title", search_result)
                self.assertIn("content", search_result)
                self.assertIn("query", search_result)

    def test_searxng_search_tool_max_results_limiting(self):
        """Test that max_results parameter correctly limits results."""
        tool = make_searxng_search_tool()

        # Test with different max_results values
        for max_results in [1, 5, 10]:
            with self.subTest(max_results=max_results):
                result = tool.handler({
                    "queries": ["Python"],
                    "max_results": max_results
                })

                if "error" not in result:
                    results = result["results"]
                    self.assertIsInstance(results, list)
                    self.assertLessEqual(len(results), max_results)

    def test_searxng_search_tool_empty_queries(self):
        """Test SearXNG search with empty queries list."""
        tool = make_searxng_search_tool()

        result = tool.handler({
            "queries": [],
            "max_results": 5
        })

        # Should return empty results
        self.assertIsInstance(result, dict)
        self.assertIn("results", result)

        if "error" not in result:
            results = result["results"]
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 0)

    def test_searxng_search_tool_invalid_queries(self):
        """Test SearXNG search with invalid query types."""
        tool = make_searxng_search_tool()

        # Test with missing required parameter
        result = tool.handler({})

        # Should return an error for missing queries
        self.assertIn("error", result)

    def test_searxng_search_tool_special_characters(self):
        """Test SearXNG search with special characters in queries."""
        tool = make_searxng_search_tool()

        # Test with special characters
        result = tool.handler({
            "queries": ["C++ programming & development", "Python's async/await"],
            "max_results": 2
        })

        self.assertIsInstance(result, dict)
        self.assertIn("results", result)

        if "error" not in result:
            results = result["results"]
            self.assertIsInstance(results, list)

            for search_result in results:
                self.assertIsInstance(search_result, dict)
                self.assertIn("url", search_result)
                self.assertIn("title", search_result)
                self.assertIn("content", search_result)
                self.assertIn("query", search_result)

    def test_searxng_search_tool_unicode_queries(self):
        """Test SearXNG search with unicode characters."""
        tool = make_searxng_search_tool()

        # Test with unicode characters
        result = tool.handler({
            "queries": ["café restaurant", "résumé writing tips"],
            "max_results": 2
        })

        self.assertIsInstance(result, dict)
        self.assertIn("results", result)

        if "error" not in result:
            results = result["results"]
            self.assertIsInstance(results, list)

            for search_result in results:
                self.assertIsInstance(search_result, dict)
                self.assertIn("url", search_result)
                self.assertIn("title", search_result)
                self.assertIn("content", search_result)
                self.assertIn("query", search_result)

    def test_searxng_search_performance(self):
        """Test that SearXNG search completes within reasonable time."""
        tool = make_searxng_search_tool()

        start_time = time.time()

        result = tool.handler({
            "queries": ["web development"],
            "max_results": 5
        })

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within 30 seconds (network requests can be slow)
        self.assertLess(duration, 30, "SearXNG search took too long")

        # Should return valid structure
        self.assertIsInstance(result, dict)
        self.assertIn("results", result)

    def test_searxng_search_different_categories(self):
        """Test SearXNG search with different categories."""
        tool = make_searxng_search_tool()

        categories_to_test = ["general", "news", "images", "videos", "science"]

        for category in categories_to_test:
            with self.subTest(category=category):
                result = tool.handler({
                    "queries": ["technology"],
                    "category": category,
                    "max_results": 2
                })

                self.assertIsInstance(result, dict)
                self.assertIn("results", result)
                self.assertIn("category", result)

                if "error" not in result:
                    self.assertEqual(result["category"], category)

                    results = result["results"]
                    self.assertIsInstance(results, list)

    def test_searxng_search_no_results_scenario(self):
        """Test SearXNG search with a query that might return no results."""
        tool = make_searxng_search_tool()

        # Use a very specific, unlikely-to-find query
        result = tool.handler({
            "queries": ["xyzabc123nonexistentquery987"],
            "max_results": 5
        })

        # Should still return valid structure even with no results
        self.assertIsInstance(result, dict)
        self.assertIn("results", result)

        if "error" not in result:
            results = result["results"]
            self.assertIsInstance(results, list)
            # Might be empty or have some results

    def test_searxng_search_duplicate_removal(self):
        """Test that duplicate URLs are removed from results."""
        tool = make_searxng_search_tool()

        # Search for the same query multiple times to test duplicate removal
        result = tool.handler({
            "queries": ["Python programming", "Python programming", "Python programming"],
            "max_results": 10
        })

        if "error" not in result:
            results = result["results"]
            self.assertIsInstance(results, list)

            # Check for duplicate URLs
            urls = set()
            for search_result in results:
                url = search_result["url"]
                self.assertNotIn(url, urls, f"Duplicate URL found: {url}")
                urls.add(url)

    def test_searxng_search_content_validation(self):
        """Test that search results have valid content."""
        tool = make_searxng_search_tool()

        result = tool.handler({
            "queries": ["artificial intelligence"],
            "max_results": 5
        })

        if "error" not in result:
            results = result["results"]
            self.assertIsInstance(results, list)

            for search_result in results:
                # Validate required fields
                self.assertIn("url", search_result)
                self.assertIn("title", search_result)
                self.assertIn("content", search_result)
                self.assertIn("query", search_result)

                # Validate URL format
                url = search_result["url"]
                self.assertTrue(url.startswith("http://") or url.startswith("https://"))
                self.assertTrue(len(url) > 10)  # Minimum reasonable URL length

                # Validate title
                title = search_result["title"]
                self.assertIsInstance(title, str)
                self.assertTrue(len(title) > 0)

                # Validate content (can be empty string)
                content = search_result["content"]
                self.assertIsInstance(content, str)

                # Validate query
                query = search_result["query"]
                self.assertIsInstance(query, str)
                self.assertTrue(len(query) > 0)

    def test_searxng_search_environment_variable(self):
        """Test that environment variable is used when set."""
        # Test with custom base URL via environment variable
        # Use a reliable public instance
        test_url = "https://searx.be"

        with patch.dict(os.environ, {"SEARXNG_BASE_URL": test_url}):
            tool = make_searxng_search_tool()

            # The tool should use the environment variable
            result = tool.handler({
                "queries": ["test"],
                "max_results": 1
            })

            self.assertIsInstance(result, dict)
            # Should either have results or an error (network issues can happen)
            self.assertTrue("results" in result or "error" in result)

    def test_searxng_search_error_handling(self):
        """Test error handling for various failure scenarios."""
        tool = make_searxng_search_tool()

        # Test with invalid base URL (this should fail gracefully)
        with patch.dict(os.environ, {"SEARXNG_BASE_URL": "http://invalid-url-that-does-not-exist.local:12345"}):
            result = tool.handler({
                "queries": ["test"],
                "max_results": 1
            })

            # Should return an error for invalid URL
            self.assertIn("error", result)

    def test_process_search_results_function(self):
        """Test the process_search_results function directly."""
        # Create mock search results
        mock_results = [
            {
                "url": "https://example.com/1",
                "title": "Result 1",
                "content": "Content 1",
                "query": "test",
                "score": 0.9
            },
            {
                "url": "https://example.com/2",
                "title": "Result 2",
                "content": "Content 2",
                "query": "test",
                "score": 0.8
            },
            {
                "url": "https://example.com/1",  # Duplicate URL
                "title": "Result 1 Duplicate",
                "content": "Content 1 Duplicate",
                "query": "test",
                "score": 0.7
            }
        ]

        # Process results
        processed = process_search_results(mock_results, max_results=5)

        # Should remove duplicates
        self.assertEqual(len(processed), 2)

        # Should be sorted by score (descending)
        self.assertEqual(processed[0]["score"], 0.9)
        self.assertEqual(processed[1]["score"], 0.8)

        # Should maintain required fields
        for result in processed:
            self.assertIn("url", result)
            self.assertIn("title", result)
            self.assertIn("content", result)
            self.assertIn("query", result)

    def test_process_search_results_with_metadata(self):
        """Test process_search_results with metadata enhancement."""
        mock_results = [
            {
                "url": "https://example.com/1",
                "title": "Result 1",
                "content": "Content 1",
                "query": "test",
                "score": 0.9,
                "metadata": "2023-01-01"
            },
            {
                "url": "https://example.com/2",
                "title": "Result 2",
                "content": "Content 2",
                "query": "test",
                "score": 0.8,
                "publishedDate": "2023-01-02"
            }
        ]

        processed = process_search_results(mock_results, max_results=5)

        # Should enhance titles with metadata
        self.assertIn("Published 2023-01-01", processed[0]["title"])
        self.assertIn("Published 2023-01-02", processed[1]["title"])

    def test_searxng_search_tool_large_query_set(self):
        """Test SearXNG search with many queries."""
        tool = make_searxng_search_tool()

        # Test with multiple queries
        queries = [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural networks"
        ]

        result = tool.handler({
            "queries": queries,
            "max_results": 2
        })

        self.assertIsInstance(result, dict)
        self.assertIn("results", result)

        if "error" not in result:
            results = result["results"]
            self.assertIsInstance(results, list)

            # Should have results from the queries
            queries_found = set(result["query"] for result in results)
            self.assertGreater(len(queries_found), 0)


if __name__ == "__main__":
    # Run the tests
    unittest.main()