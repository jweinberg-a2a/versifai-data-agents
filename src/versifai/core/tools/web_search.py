"""
Tool: web_search

Searches the web for data documentation, field definitions, and data source
metadata when local documentation is insufficient.

Uses a simple approach via requests to fetch publicly available data
documentation pages from known portals configured in ProjectConfig.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import requests

from versifai.core.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from versifai.data_agents.engineer.config import ProjectConfig


class WebSearchTool(BaseTool):
    def __init__(self, cfg: ProjectConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            raise ValueError("cfg is required. See examples/ for sample configurations.")
        self._cfg = cfg

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for data source documentation, field definitions, "
            "data dictionaries, and methodology information. Use this when local "
            "documentation files are missing or insufficient to understand the "
            "data fields. Returns relevant text content from the search results."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Be specific about the data source. "
                        "Example: 'data dictionary field definitions county level'"
                    ),
                },
                "url": {
                    "type": "string",
                    "description": (
                        "Optionally, a specific URL to fetch and extract text from. "
                        "If provided, the query parameter is ignored and this URL is fetched directly."
                    ),
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return. Default 10000.",
                    "default": 10000,
                },
            },
            "required": ["query"],
        }

    def _execute(  # type: ignore[override]
        self, query: str, url: str | None = None, max_chars: int = 10000, **kwargs
    ) -> ToolResult:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        if url:
            return self._fetch_url(url, headers, max_chars)
        else:
            return self._search_and_fetch(query, headers, max_chars)

    def _fetch_url(self, url: str, headers: dict, max_chars: int) -> ToolResult:
        """Fetch a specific URL and extract text content."""
        try:
            resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as e:
            return ToolResult(success=False, error=f"Failed to fetch {url}: {e}")

        content = self._extract_text(resp.text)
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[... TRUNCATED ...]"

        return ToolResult(
            success=True,
            data={
                "url": url,
                "content": content,
                "content_length": len(content),
            },
            summary=f"Fetched {len(content)} chars from {url}",
        )

    def _search_and_fetch(self, query: str, headers: dict, max_chars: int) -> ToolResult:
        """
        Search for documentation using known data portal URLs from config.

        Since we can't use a search API directly, we maintain a registry of
        known documentation URLs (in ProjectConfig.documentation_urls) and
        try to match the query against them.
        """
        known_sources = self._cfg.documentation_urls

        # Try to match the query to a known source
        query_lower = query.lower()
        matched_urls = []
        for key, urls in known_sources.items():
            if key in query_lower:
                matched_urls.extend(urls)

        results = []

        # Fetch matched URLs
        for url in matched_urls[:3]:
            try:
                resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
                resp.raise_for_status()
                content = self._extract_text(resp.text)
                results.append(
                    {
                        "url": url,
                        "content": content[: max_chars // max(len(matched_urls), 1)],
                    }
                )
            except requests.RequestException as e:
                results.append({"url": url, "error": str(e)})

        if not results:
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": [],
                    "suggestion": (
                        "No matching documentation URLs found in the registry. "
                        "Try using the 'url' parameter with a specific documentation URL, "
                        "or ask the human operator for the documentation location."
                    ),
                },
                summary=f"No documentation found for query: {query}",
            )

        return ToolResult(
            success=True,
            data={
                "query": query,
                "results": results,
                "urls_checked": [r["url"] for r in results],
            },
            summary=f"Searched {len(results)} documentation sources for: {query}",
        )

    def _extract_text(self, html_content: str) -> str:
        """Extract readable text from HTML content."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script/style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Fallback: strip tags with regex
            text = re.sub(r"<[^>]+>", " ", html_content)
            text = re.sub(r"\s+", " ", text).strip()

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)
