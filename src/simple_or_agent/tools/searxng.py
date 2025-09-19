from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import aiohttp
import asyncio
import json
import os

from simple_or_agent.simple_agent import ToolSpec

load_dotenv()

async def fetch_search_results(
    session: aiohttp.ClientSession,
    base_url: str,
    query: str,
    category: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Fetches search results for a single query asynchronously.

    Args:
        session: The aiohttp session to use for the request.
        base_url: The base URL for the SearXNG instance.
        query: The search query.
        category: The category of the search query.

    Returns:
        A list of search result dictionaries.

    Raises:
        Exception: If the request to SearXNG fails.
    """
    query_params = {
        "q": query,
        "safesearch": "0",
        "format": "json",
        "language": "en",
    }

    if category:
        query_params["categories"] = category

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    url = f"{base_url}/search"

    async with session.get(url, params=query_params, headers=headers) as response:

        if response.status != 200:
            response_text = await response.text()
            raise Exception(f"Failed to fetch search results for query '{query}': {response.status} {response.reason}")

        data = await response.json()
        results = data.get("results", [])

        # Add the query to each result
        for result in results:
            result["query"] = query

        return results

def process_search_results(
    all_results: List[Dict[str, Any]],
    category: Optional[str] = None,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Processes and filters search results.

    Args:
        all_results: List of all search result dictionaries.
        category: Optional category to filter results by.
        max_results: Maximum number of results to return.

    Returns:
        Filtered and processed list of search results.
    """
    # Sort the combined results by score in descending order
    sorted_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)

    # Remove duplicates while preserving order
    seen_urls = set()
    unique_results = []
    for result in sorted_results:
        if "content" not in result or "title" not in result or "url" not in result or "query" not in result:
            continue
        if result["url"] not in seen_urls:
            unique_results.append(result)
            if "metadata" in result:
                result["title"] = f"{result['title']} - (Published {result['metadata']})"
            if "publishedDate" in result and result["publishedDate"]:
                result["title"] = f"{result['title']} - (Published {result['publishedDate']})"
            seen_urls.add(result["url"])

    # Filter results to include only those with the correct category if it is set
    if category:
        filtered_results = [result for result in unique_results if result.get("category") == category]
    else:
        filtered_results = unique_results

    return filtered_results[:max_results]

async def searxng_search_async(
    queries: List[str],
    base_url: str,
    category: Optional[str] = None,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Runs SearXNG search asynchronously with the given parameters.

    Args:
        queries: List of search queries.
        base_url: The base URL for the SearXNG instance.
        category: Optional category to search in.
        max_results: Maximum number of results to return.

    Returns:
        Search results in dictionary format.

    Raises:
        Exception: If the request to SearXNG fails.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_search_results(session, base_url, query, category) for query in queries]
        results = await asyncio.gather(*tasks)

    all_results = [item for sublist in results for item in sublist]

    # Process and filter results
    final_results = process_search_results(all_results, category, max_results)

    return {
        "results": [
            {
                "url": result["url"],
                "title": result["title"],
                "content": result.get("content"),
                "query": result["query"]
            }
            for result in final_results
        ],
        "category": category,
    }

def searxng_search(
    queries: List[str],
    base_url: str,
    category: Optional[str] = None,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Runs SearXNG search synchronously with the given parameters.

    This method creates an event loop in a separate thread to run the asynchronous operations.

    Args:
        queries: List of search queries.
        base_url: The base URL for the SearXNG instance.
        category: Optional category to search in.
        max_results: Maximum number of results to return.

    Returns:
        Search results in dictionary format.

    Raises:
        Exception: If the request to SearXNG fails.
    """
    with ThreadPoolExecutor() as executor:
        return executor.submit(asyncio.run, searxng_search_async(queries, base_url, category, max_results)).result()

def make_searxng_search_tool() -> ToolSpec:
    """Performs web search using SearXNG."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            queries = args["queries"]
            category = args.get("category")
            max_results = args.get("max_results", 10)

            base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")
            results = searxng_search(queries, base_url, category, max_results)
            return results
        except Exception as e:
            return {"error": f"searxng_search failed: {e}"}

    return ToolSpec(
        name="searxng_search",
        description="Performs web search using SearXNG with support for multiple queries and categories.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries to execute."
                },
                "category": {
                    "type": "string",
                    "description": "Optional category to filter search results (e.g., 'news', 'images', 'videos')."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return per query (default: 10)."
                }
            },
            "required": ["queries"],
        },
    )

__all__ = [
    "make_searxng_search_tool",
]

if __name__ == "__main__":
    # Example usage
    results = searxng_search(
        queries=["china current population", "capital of china"],
        base_url="http://localhost:8080",
        max_results=6
    )

    print(json.dumps(results, indent=2))
