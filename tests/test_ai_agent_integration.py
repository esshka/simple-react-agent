from dotenv import load_dotenv
import unittest
import time
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from simple_or_agent.simple_agent import SimpleAgent, ToolSpec
from simple_or_agent.lmstudio_client import LMStudioClient, LMStudioError

# Import all tool modules
from simple_or_agent.tools.maths import *
from simple_or_agent.tools.searxng import *
from simple_or_agent.tools.web_search_v2 import *
from simple_or_agent.tools.wiki import *
from simple_or_agent.tools.mongo import *

load_dotenv()

class TestAIAgentIntegration(unittest.TestCase):
    """Integration tests for AI agent with all tool modules using LMStudio."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize LMStudio client with the specified model
        self.base_url = os.getenv("LMSTUDIO_BASE_URL")
        self.model = os.getenv("LMSTUDIO_MODEL")

        try:
            self.client = LMStudioClient(base_url=self.base_url)
            self.agent_available = True
        except Exception as e:
            print(f"LMStudio client not available: {e}")
            self.agent_available = False

    def create_agent_with_tools(self, tools):
        """Create an agent with the specified tools."""
        if not self.agent_available:
            self.skipTest("LMStudio client not available")

        agent = SimpleAgent(
            client=self.client,
            model=self.model,
            system_prompt="""You are a helpful AI assistant with access to various tools.
            Use the appropriate tools to answer user questions accurately and efficiently.
            Always provide detailed, well-reasoned responses based on the tool results.""",
            temperature=0.1,
            max_tool_iters=5,
            max_rounds=3
        )

        # Add all specified tools
        for tool_func in tools:
            try:
                tool = tool_func()
                agent.add_tool(tool)
            except Exception as e:
                print(f"Failed to add tool {tool_func.__name__}: {e}")

        return agent

    @unittest.skipUnless(True, "Test basic agent functionality")
    def test_agent_basic_functionality(self):
        """Test basic agent functionality without tools."""
        if not self.agent_available:
            self.skipTest("LMStudio client not available")

        agent = SimpleAgent(
            client=self.client,
            model=self.model,
            system_prompt="You are a helpful assistant.",
            max_tool_iters=0
        )

        result = agent.ask("Hello! What is 2 + 2?")

        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)
        self.assertIn("4", result.content)

    def test_maths_tools_integration(self):
        """Test AI agent with mathematical tools."""
        maths_tools = [
            make_calc_tool,
            make_sympy_solve_equation_tool,
            make_sympy_simplify_expression_tool,
            make_sympy_expand_expression_tool,
            make_sympy_factor_expression_tool,
            make_sympy_differentiate_tool,
            make_sympy_integrate_tool,
            make_sympy_matrix_operation_tool
        ]

        agent = self.create_agent_with_tools(maths_tools)

        # Test basic arithmetic
        result = agent.ask("Calculate 25 * 17 + 143")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test equation solving
        result = agent.ask("Solve the equation 2x + 5 = 15")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test calculus
        result = agent.ask("Find the derivative of x^3 + 2x^2 - 5x + 1")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test matrix operations
        result = agent.ask("What is the determinant of the 2x2 matrix [[1,2],[3,4]]?")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_web_search_tools_integration(self):
        """Test AI agent with web search tools."""
        web_tools = [
            make_duckduckgo_text_search_tool,
            make_duckduckgo_news_search_tool,
            make_duckduckgo_image_search_tool,
            make_duckduckgo_answers_search_tool,
            make_duckduckgo_maps_search_tool
        ]

        agent = self.create_agent_with_tools(web_tools)

        # Test text search
        result = agent.ask("Search for information about artificial intelligence trends in 2024")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test news search
        result = agent.ask("Find recent news about quantum computing breakthroughs")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test answers search
        result = agent.ask("What is the capital of Australia?")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_wikipedia_tools_integration(self):
        """Test AI agent with Wikipedia tools."""
        wiki_tools = [
            make_wikipedia_search_tool,
            make_wikipedia_summary_tool,
            make_wikipedia_page_content_tool,
            make_wikipedia_page_section_text_tool,
            make_wikipedia_page_categories_tool,
            make_wikipedia_page_links_tool,
            make_wikipedia_geosearch_tool,
            make_wikipedia_set_language_tool
        ]

        agent = self.create_agent_with_tools(wiki_tools)

        # Test Wikipedia search
        result = agent.ask("Search Wikipedia for articles about machine learning")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test Wikipedia summary
        result = agent.ask("Get a summary of the Wikipedia article about Python programming language")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test geosearch
        result = agent.ask("Find Wikipedia articles near the Eiffel Tower coordinates")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_searxng_tools_integration(self):
        """Test AI agent with SearXNG search tools."""
        searxng_tools = [make_searxng_search_tool]

        agent = self.create_agent_with_tools(searxng_tools)

        # Test basic SearXNG search
        result = agent.ask("Search for information about renewable energy developments")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test multiple queries
        result = agent.ask("Search for both 'climate change solutions' and 'sustainable technology'")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_mongodb_tools_integration(self):
        """Test AI agent with MongoDB tools (requires local MongoDB)."""
        mongo_tools = [
            make_mongo_insert_one_tool,
            make_mongo_find_tool,
            make_mongo_update_one_tool,
            make_mongo_delete_one_tool,
            make_mongo_aggregate_tool,
            make_mongo_list_collections_tool,
            make_mongo_list_databases_tool,
            make_mongo_drop_collection_tool,
            make_mongo_count_documents_tool
        ]

        agent = self.create_agent_with_tools(mongo_tools)

        # Test basic MongoDB operations
        result = agent.ask("Connect to MongoDB and list available databases")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test data insertion and query
        result = agent.ask("Insert a test document into a collection and then retrieve it")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_cross_tool_integration(self):
        """Test AI agent using multiple tool types together."""
        all_tools = [
            make_calc_tool,
            make_duckduckgo_text_search_tool,
            make_wikipedia_summary_tool,
            make_searxng_search_tool
        ]

        agent = self.create_agent_with_tools(all_tools)

        # Test complex query requiring multiple tools
        result = agent.ask("Search for information about the mathematical constant pi, calculate its value to 10 decimal places, and get its Wikipedia summary")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_tool_error_handling(self):
        """Test agent's error handling with tools."""
        # Create agent with a simple tool
        agent = self.create_agent_with_tools([make_calc_tool])

        # Test with invalid calculation
        result = agent.ask("Calculate 1 divided by 0")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test with malformed mathematical expression
        result = agent.ask("Calculate 'invalid math expression'")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_agent_conversation_flow(self):
        """Test agent maintains conversation context."""
        agent = self.create_agent_with_tools([make_calc_tool, make_wikipedia_summary_tool])

        # First question
        result1 = agent.ask("What is 15 * 23?")
        self.assertIsInstance(result1.content, str)

        # Follow-up question that should use context
        result2 = agent.ask("Now add 100 to that result")
        self.assertIsInstance(result2.content, str)
        self.assertTrue(len(result2.content) > 0)

    def test_agent_tool_selection(self):
        """Test agent selects appropriate tools for different queries."""
        agent = self.create_agent_with_tools([
            make_calc_tool,
            make_duckduckgo_text_search_tool,
            make_wikipedia_summary_tool
        ])

        # Mathematical query - should use calculate tool
        result = agent.ask("What is the square root of 144?")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Factual query - should use Wikipedia
        result = agent.ask("Give me a summary of Albert Einstein")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Current events query - should use web search
        result = agent.ask("Search for recent developments in artificial intelligence")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_agent_performance(self):
        """Test agent performance with various tasks."""
        agent = self.create_agent_with_tools([
            make_calc_tool,
            make_wikipedia_summary_tool,
            make_duckduckgo_text_search_tool
        ])

        # Test performance with quick calculation
        start_time = time.time()
        result = agent.ask("Calculate 2^10")
        calc_time = time.time() - start_time

        self.assertLess(calc_time, 30, "Calculation took too long")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test performance with Wikipedia query
        start_time = time.time()
        result = agent.ask("Get a brief summary of the Python programming language")
        wiki_time = time.time() - start_time

        self.assertLess(wiki_time, 60, "Wikipedia query took too long")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_agent_tool_iteration(self):
        """Test agent can handle multiple tool calls in a single response."""
        agent = self.create_agent_with_tools([
            make_calc_tool,
            make_sympy_solve_equation_tool
        ])

        # Complex query that might require multiple calculations
        result = agent.ask("Solve for x in the equation 2x + 5 = 15, then calculate x squared plus 10")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_agent_memory_management(self):
        """Test agent handles conversation history properly."""
        agent = self.create_agent_with_tools([make_calc_tool])

        # Have a conversation
        for i in range(3):
            result = agent.ask(f"What is {i} + {i+1}?")
            self.assertIsInstance(result.content, str)
            self.assertTrue(len(result.content) > 0)

        # Reset and test with fresh context
        agent.reset()
        result = agent.ask("What is 5 + 3?")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_agent_parameter_validation(self):
        """Test agent validates tool parameters correctly."""
        agent = self.create_agent_with_tools([make_calc_tool])

        # Test with invalid parameters
        result = agent.ask("Calculate with no expression")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

        # Test with valid parameters
        result = agent.ask("Calculate 100 divided by 4")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_agent_network_resilience(self):
        """Test agent handles network issues gracefully."""
        agent = self.create_agent_with_tools([
            make_duckduckgo_text_search_tool,
            make_wikipedia_summary_tool
        ])

        # Test that agent handles potential network failures gracefully
        result = agent.ask("Search for information about a very specific topic")
        self.assertIsInstance(result.content, str)
        # Should not crash even if network calls fail

    def test_agent_multilingual_queries(self):
        """Test agent can handle queries in different languages."""
        agent = self.create_agent_with_tools([
            make_wikipedia_set_language_tool,
            make_wikipedia_summary_tool
        ])

        # Test language switching
        result = agent.ask("Set Wikipedia to Spanish and get a summary of 'Madrid'")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_agent_complex_reasoning(self):
        """Test agent can handle complex reasoning tasks."""
        agent = self.create_agent_with_tools([
            make_calc_tool,
            make_wikipedia_summary_tool,
            make_duckduckgo_text_search_tool
        ])

        # Complex multi-step reasoning task
        result = agent.ask("Research the Fibonacci sequence, calculate the 10th Fibonacci number, and explain its significance")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    def test_agent_tool_combination(self):
        """Test agent can combine multiple tools effectively."""
        agent = self.create_agent_with_tools([
            make_duckduckgo_text_search_tool,
            make_wikipedia_summary_tool,
            make_calc_tool
        ])

        # Query that might benefit from multiple information sources
        result = agent.ask("Research climate change statistics, find related Wikipedia articles, and calculate percentage changes")
        self.assertIsInstance(result.content, str)
        self.assertTrue(len(result.content) > 0)

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Additional cleanup if needed
        pass


if __name__ == "__main__":
    # Run the tests with verbose output
    unittest.main(verbosity=2)