"""
Tools module for the simple-react-agent project.

This module provides access to various tool specifications that can be used
with AI agents for different functionalities including mathematics, web search,
Wikipedia access, MongoDB operations, and more.
"""

__version__ = "1.0.0"
__author__ = "simple-react-agent project"

# Import all available tool modules
from simple_or_agent.tools.maths import *
from simple_or_agent.tools.mongo import *
from simple_or_agent.tools.wiki import *
from simple_or_agent.tools.web_search_v2 import *
from simple_or_agent.tools.searxng import *
from simple_or_agent.tools.toolkits import *

# Define tool categories for organized access
MATHS_TOOLS = [
    make_calc_tool,
    make_sympy_solve_equation_tool,
    make_sympy_simplify_expression_tool,
    make_sympy_expand_expression_tool,
    make_sympy_factor_expression_tool,
    make_sympy_differentiate_tool,
    make_sympy_integrate_tool,
    make_sympy_matrix_operation_tool,
]

DATABASE_TOOLS = [
    make_mongo_insert_one_tool,
    make_mongo_find_one_tool,
    make_mongo_find_tool,
    make_mongo_update_one_tool,
    make_mongo_delete_one_tool,
    make_mongo_aggregate_tool,
    make_mongo_count_documents_tool,
    make_mongo_list_databases_tool,
    make_mongo_list_collections_tool,
    make_mongo_drop_collection_tool,
]

KNOWLEDGE_TOOLS = [
    make_wikipedia_search_tool,
    make_wikipedia_summary_tool,
    make_wikipedia_page_content_tool,
    make_wikipedia_page_section_text_tool,
    make_wikipedia_page_categories_tool,
    make_wikipedia_page_links_tool,
    make_wikipedia_geosearch_tool,
    make_wikipedia_set_language_tool,
]

WEB_SEARCH_TOOLS = [
    make_duckduckgo_text_search_tool,
    make_duckduckgo_news_search_tool,
    make_duckduckgo_image_search_tool,
    make_duckduckgo_answers_search_tool,
    make_duckduckgo_maps_search_tool,
]

META_SEARCH_TOOLS = [
    make_searxng_search_tool,
]

# All tools combined
ALL_TOOLS = (
    MATHS_TOOLS +
    DATABASE_TOOLS +
    KNOWLEDGE_TOOLS +
    WEB_SEARCH_TOOLS +
    META_SEARCH_TOOLS
)

# Tool registry for easy lookup
TOOL_REGISTRY = {}
for tool_func in ALL_TOOLS:
    try:
        tool = tool_func()
        TOOL_REGISTRY[tool.name] = tool
    except Exception:
        # Skip tools that can't be instantiated (e.g., missing dependencies)
        pass


def get_tool_by_name(name: str):
    """Get a tool specification by its name.

    Args:
        name: The name of the tool to retrieve

    Returns:
        ToolSpec instance or None if not found
    """
    return TOOL_REGISTRY.get(name)


def list_all_tools():
    """List all available tool names.

    Returns:
        List of tool names
    """
    return list(TOOL_REGISTRY.keys())


def get_tools_by_category(category: str):
    """Get all tools in a specific category.

    Args:
        category: One of 'maths', 'database', 'knowledge', 'web_search', 'meta_search'

    Returns:
        List of ToolSpec instances
    """
    category_map = {
        'maths': MATHS_TOOLS,
        'database': DATABASE_TOOLS,
        'knowledge': KNOWLEDGE_TOOLS,
        'web_search': WEB_SEARCH_TOOLS,
        'meta_search': META_SEARCH_TOOLS,
    }

    tool_funcs = category_map.get(category, [])
    tools = []

    for tool_func in tool_funcs:
        try:
            tool = tool_func()
            tools.append(tool)
        except Exception:
            continue

    return tools


def create_tools_dict(tools_list=None):
    """Create a dictionary of tool specifications for agent use.

    Args:
        tools_list: Optional list of tool functions. If None, uses all tools.

    Returns:
        Dictionary mapping tool names to ToolSpec instances
    """
    if tools_list is None:
        tools_list = ALL_TOOLS

    tools_dict = {}

    for tool_func in tools_list:
        try:
            tool = tool_func()
            tools_dict[tool.name] = tool
        except Exception as e:
            print(f"Warning: Failed to create tool {tool_func.__name__}: {e}")

    return tools_dict


def get_tool_info():
    """Get information about all available tools.

    Returns:
        Dictionary with tool information organized by category
    """
    return {
        'categories': {
            'maths': {
                'description': 'Mathematical computation and symbolic algebra tools',
                'tools': [tool_func.__name__ for tool_func in MATHS_TOOLS]
            },
            'database': {
                'description': 'Database operations and data persistence tools',
                'tools': [tool_func.__name__ for tool_func in DATABASE_TOOLS]
            },
            'knowledge': {
                'description': 'Knowledge base and reference tools',
                'tools': [tool_func.__name__ for tool_func in KNOWLEDGE_TOOLS]
            },
            'web_search': {
                'description': 'Web search and information retrieval tools',
                'tools': [tool_func.__name__ for tool_func in WEB_SEARCH_TOOLS]
            },
            'meta_search': {
                'description': 'Meta-search engine tools',
                'tools': [tool_func.__name__ for tool_func in META_SEARCH_TOOLS]
            }
        },
        'total_tools': len(TOOL_REGISTRY),
        'tool_names': list(TOOL_REGISTRY.keys())
    }


# Export all tool creation functions
__all__ = [
    # Toolkits
    'MathsToolkit',
    'DatabaseToolkit',
    'WikiToolkit',
    'WebSearchToolkit',
    'MetaSearchToolkit',

    # Maths tools
    'make_calc_tool',
    'make_sympy_solve_equation_tool',
    'make_sympy_simplify_expression_tool',
    'make_sympy_expand_expression_tool',
    'make_sympy_factor_expression_tool',
    'make_sympy_differentiate_tool',
    'make_sympy_integrate_tool',
    'make_sympy_matrix_operation_tool',

    # MongoDB tools
    'make_mongo_insert_one_tool',
    'make_mongo_find_one_tool',
    'make_mongo_find_tool',
    'make_mongo_update_one_tool',
    'make_mongo_delete_one_tool',
    'make_mongo_aggregate_tool',
    'make_mongo_count_documents_tool',
    'make_mongo_list_databases_tool',
    'make_mongo_list_collections_tool',
    'make_mongo_drop_collection_tool',

    # Wikipedia tools
    'make_wikipedia_search_tool',
    'make_wikipedia_summary_tool',
    'make_wikipedia_page_content_tool',
    'make_wikipedia_page_section_text_tool',
    'make_wikipedia_page_categories_tool',
    'make_wikipedia_page_links_tool',
    'make_wikipedia_geosearch_tool',
    'make_wikipedia_set_language_tool',

    # Web search tools
    'make_duckduckgo_text_search_tool',
    'make_duckduckgo_news_search_tool',
    'make_duckduckgo_image_search_tool',
    'make_duckduckgo_answers_search_tool',
    'make_duckduckgo_maps_search_tool',

    # SearXNG tools
    'make_searxng_search_tool',
    'searxng_search',
    'searxng_search_async',

    # Tool categories
    'MATHS_TOOLS',
    'DATABASE_TOOLS',
    'KNOWLEDGE_TOOLS',
    'WEB_SEARCH_TOOLS',
    'META_SEARCH_TOOLS',
    'ALL_TOOLS',

    # Utility functions
    'get_tool_by_name',
    'list_all_tools',
    'get_tools_by_category',
    'create_tools_dict',
    'get_tool_info',
    'TOOL_REGISTRY',
]

# Version and metadata
__all__ += ['__version__', '__author__']
