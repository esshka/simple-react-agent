from typing import Dict, Any
from ddgs import DDGS

from simple_or_agent.simple_agent import ToolSpec

# --- Core Search Tools ---

def make_duckduckgo_text_search_tool() -> ToolSpec:
    """Performs a general web search for text results."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    args["keywords"],
                    region=args.get("region", "wt-wt"),
                    timelimit=args.get("timelimit"),
                    max_results=args.get("max_results", 10),
                )
                return {"results": results or "No results found."}
        except Exception as e:
            return {"error": f"DuckDuckGo text search failed: {e}"}

    return ToolSpec(
        name="duckduckgo_text_search",
        description="Search DuckDuckGo for text-based web results. Useful for finding information, articles, and answers to general questions.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "The search query or keywords."
                },
                "region": {
                    "type": "string",
                    "description": "Region for the search, e.g., 'us-en', 'uk-en', 'de-de'. Default is 'wt-wt' (worldwide)."
                },
                "timelimit": {
                    "type": "string",
                    "description": "Filter results by time. Options: 'd' (day), 'w' (week), 'm' (month), 'y' (year)."
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of results to return. Defaults to 10."
                },
            },
            "required": ["keywords"],
        },
    )

def make_duckduckgo_news_search_tool() -> ToolSpec:
    """Searches for news articles."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            with DDGS() as ddgs:
                results = ddgs.news(
                    args["keywords"],
                    region=args.get("region", "wt-wt"),
                    timelimit=args.get("timelimit"),
                    max_results=args.get("max_results", 15),
                )
                return {"results": results or "No news found."}
        except Exception as e:
            return {"error": f"DuckDuckGo news search failed: {e}"}

    return ToolSpec(
        name="duckduckgo_news_search",
        description="Search DuckDuckGo for the latest news articles. Ideal for current events and recent information.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "The topic or keywords to search for in the news."
                },
                "region": {
                    "type": "string",
                    "description": "Region for the search, e.g., 'us-en', 'uk-en'. Default is 'wt-wt'."
                },
                "timelimit": {
                    "type": "string",
                    "description": "Filter news by time. Options: 'd' (day), 'w' (week), 'm' (month)."
                },
                 "max_results": {
                    "type": "integer",
                    "description": "The maximum number of news articles to return. Defaults to 15."
                },
            },
            "required": ["keywords"],
        },
    )

def make_duckduckgo_image_search_tool() -> ToolSpec:
    """Searches for images."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            with DDGS() as ddgs:
                results = ddgs.images(
                    args["keywords"],
                    region=args.get("region", "wt-wt"),
                    size=args.get("size"),
                    max_results=args.get("max_results", 10),
                )
                return {"results": results or "No images found."}
        except Exception as e:
            return {"error": f"DuckDuckGo image search failed: {e}"}

    return ToolSpec(
        name="duckduckgo_image_search",
        description="Search DuckDuckGo for images. Use this to find pictures, diagrams, or visual representations.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "The keywords for the image search."
                },
                "region": {
                    "type": "string",
                    "description": "Region for the search. Default is 'wt-wt'."
                },
                "size": {
                    "type": "string",
                    "description": "Filter by image size. Options: 'Small', 'Medium', 'Large', 'Wallpaper'."
                },
                 "max_results": {
                    "type": "integer",
                    "description": "The maximum number of images to return. Defaults to 10."
                },
            },
            "required": ["keywords"],
        },
    )


# --- Specialized Search Tools ---

def make_duckduckgo_answers_search_tool() -> ToolSpec:
    """Searches for direct, concise answers (Instant Answers)."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            with DDGS() as ddgs:
                # Use text search with limited results for quick answers
                results = ddgs.text(
                    args["keywords"],
                    max_results=3  # Limit results for quick answers
                )
                return {"results": results or "No direct answer found."}
        except Exception as e:
            return {"error": f"DuckDuckGo answers search failed: {e}"}

    return ToolSpec(
        name="duckduckgo_answers_search",
        description="Search for a direct, concise answer to a question (like an infobox or a quick definition). Best for simple, factual queries.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "The question or keywords to find a direct answer for (e.g., 'capital of Canada')."
                }
            },
            "required": ["keywords"],
        },
    )

def make_duckduckgo_maps_search_tool() -> ToolSpec:
    """Performs a search for geographical locations or directions."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            with DDGS() as ddgs:
                # Use text search for location-based queries
                search_query = args["keywords"]
                if args.get("place"):
                    search_query = f"{search_query} {args['place']}"

                results = ddgs.text(
                    search_query,
                    max_results=args.get("max_results", 5),
                )
                return {"results": results or "No locations found."}
        except Exception as e:
            return {"error": f"DuckDuckGo maps search failed: {e}"}

    return ToolSpec(
        name="duckduckgo_maps_search",
        description="Search for geographical information, such as addresses, locations, or directions between places.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "The search query, e.g., 'Eiffel Tower' or 'pizza near me'."
                },
                "place": {
                    "type": "string",
                    "description": "A specific city or region to narrow down the search, e.g., 'Paris, France'."
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of map results to return. Defaults to 5."
                },
            },
            "required": ["keywords"],
        },
    )

__all__ = [
    "make_duckduckgo_text_search_tool",
    "make_duckduckgo_news_search_tool",
    "make_duckduckgo_image_search_tool",
    "make_duckduckgo_answers_search_tool",
    "make_duckduckgo_maps_search_tool",
]
