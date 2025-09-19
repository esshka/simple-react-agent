from typing import Dict, Any
import wikipedia

from simple_or_agent.simple_agent import ToolSpec

# --- Wikipedia tools ---

# Tool for wikipedia.search
def make_wikipedia_search_tool() -> ToolSpec:
    """Search Wikipedia for a query and return a list of matching page titles."""

    def handler(args: Dict[str, Any]) -> Any:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "empty_query"}
        results = int(args.get("results", 10) or 10)
        try:
            search_results = wikipedia.search(query=query, results=results)
            return {"query": query, "results": search_results}
        except Exception as e:
            return {"error": f"wikipedia_search failed: {e}", "query": query}

    return ToolSpec(
        name="wikipedia_search",
        description="Search Wikipedia for a query and return a list of matching page titles.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The term or phrase to search for on Wikipedia."
                },
                "results": {
                    "type": "integer",
                    "description": "The maximum number of results to return. Defaults to 10."
                }
            },
            "required": ["query"],
        }
    )

# Tool for wikipedia.summary
def make_wikipedia_summary_tool() -> ToolSpec:
    """Get a plain text summary of a Wikipedia page."""

    def handler(args: Dict[str, Any]) -> Any:
        title = str(args.get("title", "")).strip()
        if not title:
            return {"error": "empty_title"}
        sentences = int(args.get("sentences", 0) or 0)
        try:
            summary = wikipedia.summary(title=title, sentences=sentences)
            return {"title": title, "summary": summary}
        except Exception as e:
            return {"error": f"wikipedia_summary failed: {e}", "title": title}

    return ToolSpec(
        name="wikipedia_summary",
        description="Get a plain text summary of a Wikipedia page.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The exact title of the Wikipedia page."
                },
                "sentences": {
                    "type": "integer",
                    "description": "The number of sentences to include in the summary. If 0, returns the default introductory summary."
                }
            },
            "required": ["title"],
        }
    )

# Tool to get the full content of a page
def make_wikipedia_page_content_tool() -> ToolSpec:
    """Retrieve the full plain text content of a Wikipedia page, excluding tables and images."""

    def handler(args: Dict[str, Any]) -> Any:
        title = str(args.get("title", "")).strip()
        if not title:
            return {"error": "empty_title"}
        try:
            page = wikipedia.page(title=title)
            content = page.content
            return {"title": title, "content": content}
        except Exception as e:
            return {"error": f"wikipedia_page_content failed: {e}", "title": title}

    return ToolSpec(
        name="wikipedia_get_page_content",
        description="Retrieve the full plain text content of a Wikipedia page, excluding tables and images.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The exact title of the page to retrieve content from."
                }
            },
            "required": ["title"],
        }
    )

# Tool to get the text of a specific section
def make_wikipedia_page_section_text_tool() -> ToolSpec:
    """Get the plain text content of a specific section from a Wikipedia page."""

    def handler(args: Dict[str, Any]) -> Any:
        title = str(args.get("title", "")).strip()
        section_title = str(args.get("section_title", "")).strip()
        if not title or not section_title:
            return {"error": "empty_title_or_section_title"}
        try:
            section_text = wikipedia.page(title=title).section(section_title)
            if section_text is None:
                return {"error": "section_not_found", "title": title, "section_title": section_title}
            return {"title": title, "section_title": section_title, "section_text": section_text}
        except Exception as e:
            return {"error": f"wikipedia_page_section_text failed: {e}", "title": title, "section_title": section_title}

    return ToolSpec(
        name="wikipedia_get_page_section_text",
        description="Get the plain text content of a specific section from a Wikipedia page.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The exact title of the page."
                },
                "section_title": {
                    "type": "string",
                    "description": "The exact title of the section to retrieve."
                }
            },
            "required": ["title", "section_title"],
        }
    )

# Tool to get page categories
def make_wikipedia_page_categories_tool() -> ToolSpec:
    """List the categories a Wikipedia page belongs to."""

    def handler(args: Dict[str, Any]) -> Any:
        title = str(args.get("title", "")).strip()
        if not title:
            return {"error": "empty_title"}
        try:
            categories = wikipedia.page(title=title).categories
            return {"title": title, "categories": categories}
        except Exception as e:
            return {"error": f"wikipedia_page_categories failed: {e}", "title": title}

    return ToolSpec(
        name="wikipedia_get_page_categories",
        description="List the categories a Wikipedia page belongs to.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the page."
                }
            },
            "required": ["title"],
        }
    )

# Tool to get internal page links
def make_wikipedia_page_links_tool() -> ToolSpec:
    """List the titles of all Wikipedia pages linked from a given page."""

    def handler(args: Dict[str, Any]) -> Any:
        title = str(args.get("title", "")).strip()
        if not title:
            return {"error": "empty_title"}
        try:
            links = wikipedia.page(title=title).links
            return {"title": title, "links": links}
        except Exception as e:
            return {"error": f"wikipedia_page_links failed: {e}", "title": title}

    return ToolSpec(
        name="wikipedia_get_page_links",
        description="List the titles of all Wikipedia pages linked from a given page.",
        handler=lambda params: wikipedia.page(title=params['title']).links,
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the page."
                }
            },
            "required": ["title"],
        }
    )

# Tool for wikipedia.geosearch
def make_wikipedia_geosearch_tool() -> ToolSpec:
    """Find Wikipedia articles geographically near a given latitude and longitude."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            latitude = float(args.get("latitude"))
            longitude = float(args.get("longitude"))
        except (TypeError, ValueError):
            return {"error": "invalid_latitude_or_longitude"}
        radius = int(args.get("radius", 1000) or 1000)
        if radius < 10 or radius > 10000:
            return {"error": "radius_out_of_bounds"}
        results = int(args.get("results", 10) or 10)
        try:
            search_results = wikipedia.geosearch(latitude=latitude, longitude=longitude, radius=radius, results=results)
            return {"latitude": latitude, "longitude": longitude, "radius": radius, "results": search_results}
        except Exception as e:
            return {"error": f"wikipedia_geosearch failed: {e}", "latitude": latitude, "longitude": longitude}

    return ToolSpec(
        name="wikipedia_geosearch",
        description="Find Wikipedia articles geographically near a given latitude and longitude.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "The latitude for the geographic search."
                },
                "longitude": {
                    "type": "number",
                    "description": "The longitude for the geographic search."
                },
                "radius": {
                    "type": "integer",
                    "description": "Search radius in meters (must be between 10 and 10000). Defaults to 1000."
                },
                "results": {
                    "type": "integer",
                    "description": "The maximum number of results to return. Defaults to 10."
                }
            },
            "required": ["latitude", "longitude"],
        }
    )

# Tool for wikipedia.set_lang
def make_wikipedia_set_language_tool() -> ToolSpec:
    """Set the language for all subsequent Wikipedia searches and article retrievals."""

    def handler(args: Dict[str, Any]) -> Any:
        prefix = str(args.get("prefix", "")).strip()
        if not prefix:
            return {"error": "empty_prefix"}
        try:
            wikipedia.set_lang(prefix)
            return {"message": f"Wikipedia language set to '{prefix}'"}
        except Exception as e:
            return {"error": f"wikipedia_set_lang failed: {e}", "prefix": prefix}

    return ToolSpec(
        name="wikipedia_set_language",
        description="Set the language for all subsequent Wikipedia searches and article retrievals (e.g., 'en' for English, 'de' for German).",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "The two-letter ISO language code for the desired Wikipedia language."
                }
            },
            "required": ["prefix"],
        }
    )

__all__ = [
    "make_wikipedia_search_tool",
    "make_wikipedia_summary_tool",
    "make_wikipedia_page_content_tool",
    "make_wikipedia_page_section_text_tool",
    "make_wikipedia_page_categories_tool",
    "make_wikipedia_page_links_tool",
    "make_wikipedia_geosearch_tool",
    "make_wikipedia_set_language_tool",
]
