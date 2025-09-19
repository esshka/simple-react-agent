from simple_or_agent.tools.maths import *
from simple_or_agent.tools.mongo import *
from simple_or_agent.tools.wiki import *
from simple_or_agent.tools.web_search_v2 import *
from simple_or_agent.tools.searxng import *

class BaseToolkit:
    """A base class for toolkits to avoid repetitive code."""
    TOOL_FACTORIES = []

    def __init__(self):
        """Initializes the toolkit by creating all the tool instances."""
        self.tools = [factory() for factory in self.TOOL_FACTORIES]
        # The f-string provides a nice confirmation message when a toolkit is created
        print(f"✅ {self.__class__.__name__} initialized with {len(self.tools)} tools.")

    def __iter__(self):
        """Makes the toolkit instance iterable."""
        return iter(self.tools)
    
class MathsToolkit(BaseToolkit):
    """Toolkit for mathematical operations and symbolic algebra."""
    TOOL_FACTORIES = [
        make_calc_tool,
        make_sympy_solve_equation_tool,
        make_sympy_simplify_expression_tool,
        make_sympy_expand_expression_tool,
        make_sympy_factor_expression_tool,
        make_sympy_differentiate_tool,
        make_sympy_integrate_tool,
        make_sympy_matrix_operation_tool,
    ]

class DatabaseToolkit(BaseToolkit):
    """Toolkit for all MongoDB database operations."""
    TOOL_FACTORIES = [
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

class WikiToolkit(BaseToolkit):
    """Toolkit for querying Wikipedia."""
    TOOL_FACTORIES = [
        make_wikipedia_search_tool,
        make_wikipedia_summary_tool,
        make_wikipedia_page_content_tool,
        make_wikipedia_page_section_text_tool,
        make_wikipedia_page_categories_tool,
        make_wikipedia_page_links_tool,
        make_wikipedia_geosearch_tool,
        make_wikipedia_set_language_tool,
    ]

class WebSearchToolkit(BaseToolkit):
    """Toolkit for performing web searches with DuckDuckGo."""
    TOOL_FACTORIES = [
        make_duckduckgo_text_search_tool,
        make_duckduckgo_news_search_tool,
        make_duckduckgo_image_search_tool,
        make_duckduckgo_answers_search_tool,
        make_duckduckgo_maps_search_tool,
    ]

class MetaSearchToolkit(BaseToolkit):
    """Toolkit for performing meta-searches with SearXNG."""
    TOOL_FACTORIES = [
        make_searxng_search_tool,
    ]

def print_toolkits():
    maths = MathsToolkit()
    database = DatabaseToolkit()
    wiki = WikiToolkit()
    web_search = WebSearchToolkit()
    meta_search = MetaSearchToolkit()

    for toolkit in [maths, database, wiki, web_search, meta_search]:
        print(f"{toolkit.__class__.__name__}")
        for tool in toolkit:
            print(f"    {tool.name}")

__all__ = [
    'MathsToolkit',
    'DatabaseToolkit',
    'WikiToolkit',
    'WebSearchToolkit',
    'MetaSearchToolkit',
]

if __name__ == "__main__":
    print_toolkits()
