from typing import Any, Dict, List
from urllib.parse import urlparse
from typing import Dict, Any

from simple_or_agent.simple_agent import ToolSpec

# --- DuckDuckGo search and page fetch tools ---

SERP_HOSTS = {
    "bing.com", "www.bing.com",
    "google.com", "www.google.com",
    "duckduckgo.com", "www.duckduckgo.com",
    "search.yahoo.com", "yahoo.com", "www.yahoo.com",
    "startpage.com", "www.startpage.com",
    "yandex.com", "www.yandex.com", "yandex.ru", "www.yandex.ru",
    "baidu.com", "www.baidu.com",
}

def _is_serp(url: str) -> bool:
        try:
            p = urlparse(url)
            host = (p.hostname or "").lower()
            if not host:
                return False
            if host in SERP_HOSTS:
                return True
            if any(seg in (p.path or "").lower() for seg in ("/search", "/html", "/lite")) and (
                host.endswith("google.com") or host.endswith("bing.com") or host.endswith("duckduckgo.com") or host.endswith("yahoo.com")
            ):
                return True
            return False
        except Exception:
            return False

def make_web_search_tool(default_region: str, default_time: str | None, default_max: int) -> ToolSpec:
    """DuckDuckGo search (via ddgs). Returns compact results."""
    SERP_HOSTS = {
        "bing.com", "www.bing.com",
        "google.com", "www.google.com",
        "duckduckgo.com", "www.duckduckgo.com",
        "search.yahoo.com", "yahoo.com", "www.yahoo.com",
        "startpage.com", "www.startpage.com",
        "yandex.com", "www.yandex.com", "yandex.ru", "www.yandex.ru",
        "baidu.com", "www.baidu.com",
    }

    def handler(args: Dict[str, Any]) -> Any:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "empty_query"}

        # Use the new ddgs package exclusively
        try:
            from ddgs import DDGS  # type: ignore
        except Exception as e:
            return {"error": f"ddgs_import_failed: {e}"}

        region = str(args.get("region", default_region) or default_region)
        timelimit = args.get("time", default_time)
        max_results = int(args.get("max_results", default_max) or default_max)
        max_results = max(3, min(max_results, 15))

        out: List[Dict[str, Any]] = []
        try:
            with DDGS() as ddg:
                for r in ddg.text(query, region=region, safesearch="moderate", timelimit=timelimit, max_results=max_results):
                    url = r.get("href")
                    if not url or _is_serp(url):
                        # Skip search engine result pages; they are not sources
                        continue
                    out.append({
                        "title": r.get("title"),
                        "url": url,
                        "snippet": r.get("body"),
                    })
        except Exception as e:
            return {"error": f"ddgs_search_failed: {e}", "query": query}

        # If filtering removed everything, still return empty list (let model refine query)
        return {"query": query, "results": out[:max_results]}

    params: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "minimum": 3, "maximum": 15, "default": default_max},
            "region": {"type": "string", "default": default_region},
            "time": {"type": ["string", "null"], "enum": [None, "d", "w", "m", "y"], "default": default_time},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    return ToolSpec(name="web_search", description="Search the web via ddgs (DuckDuckGo) and return top results (title, url, snippet)", parameters=params, handler=handler)


def make_fetch_page_tool(default_max_chars: int) -> ToolSpec:
    """HTTP GET a URL and extract readable text using BeautifulSoup."""
    UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

    def _extract_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup  # local import
        except Exception:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        parts: List[str] = []
        # Prefer headings and paragraphs to keep it compact
        for sel in ["h1", "h2", "h3", "p", "li"]:
            for el in soup.select(sel):
                text = (el.get_text(" ", strip=True) or "").strip()
                if text:
                    parts.append(text)
        text = "\n".join(parts)
        # Collapse whitespace
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    def handler(args: Dict[str, Any]) -> Any:
        import requests
        url = str(args.get("url", "")).strip()
        if not url:
            return {"error": "empty_url"}
        # Block fetching search engine result pages to avoid garbage content
        if _is_serp(url):
            return {"error": "blocked_serp_url", "url": url}
        max_chars = int(args.get("max_chars", default_max_chars) or default_max_chars)
        max_chars = max(1000, min(max_chars, 15000))
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": UA})
            resp.raise_for_status()
        except Exception as e:
            return {"error": f"fetch_failed: {e}", "url": url}

        title = None
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(resp.text, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            title = None

        text = _extract_text(resp.text)
        if len(text) > max_chars:
            text = text[:max_chars]
        return {"url": url, "title": title, "text": text, "length": len(text)}

    params: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Page URL to fetch"},
            "max_chars": {"type": "integer", "minimum": 1000, "maximum": 15000, "default": default_max_chars},
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    return ToolSpec(name="fetch_page", description="Fetch a web page and return cleaned text (capped by max_chars)", parameters=params, handler=handler)

__all__ = [
    "make_web_search_tool",
    "make_fetch_page_tool",
]
