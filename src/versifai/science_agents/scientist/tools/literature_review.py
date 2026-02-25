"""
Tool: review_literature

Fetches and analyzes published research to compare with the agent's own
findings. Supports searching for research articles, reading web pages
and reports, and structuring published findings for comparison.

Delegates page fetching to the core WebScraperTool, which handles
JavaScript-rendered SPAs, PDFs, and standard HTML pages.

This gives the DataScientistAgent the ability to ground its analysis
in existing peer-reviewed and policy research, strengthening the
evidence base and identifying novel contributions.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

import requests

from versifai.core.tools.base import BaseTool, ToolResult
from versifai.core.tools.web_scraper import WebScraperTool

if TYPE_CHECKING:
    from versifai.science_agents.scientist.config import ResearchConfig

logger = logging.getLogger("agent.scientist.literature")


class LiteratureReviewTool(BaseTool):
    """
    Searches for and reads published research for comparison with
    the agent's own analysis findings.

    Delegates web fetching to WebScraperTool for JS rendering support.

    Operations:
    - search: Search for published research on a topic
    - fetch_article: Fetch and extract text from a URL (supports JS-rendered pages)
    - compare_findings: Structure a comparison between own and published findings
    """

    def __init__(
        self,
        cfg: "ResearchConfig | None" = None,
        web_scraper: WebScraperTool | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            raise ValueError("cfg is required. See examples/ for sample configurations.")
        self._cfg = cfg
        self._scraper = web_scraper or WebScraperTool()

    @property
    def name(self) -> str:
        return "review_literature"

    @property
    def description(self) -> str:
        return (
            "Search for, read, and analyze published research to compare with your findings. "
            "Operations:\n"
            "- 'search': Search for published research on a topic. Returns article titles, "
            "URLs, and snippets from web search results.\n"
            "- 'fetch_article': Fetch and extract text content from a research article URL. "
            "Handles JavaScript-rendered pages (React SPAs), PDFs, and standard HTML.\n"
            "- 'compare_findings': Structure a comparison between your findings and published "
            "results. Provide your finding and the published finding to get a structured comparison.\n\n"
            "Use this to validate your results against existing literature, identify where your "
            "findings align with or diverge from published research, and strengthen your evidence base.\n\n"
            "For navigating multi-page research sites (like USRDS ADR), use 'scrape_web' tool "
            "with discover_site first to map the site, then fetch_article here to read specific pages."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation: 'search', 'fetch_article', or 'compare_findings'.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "For 'search': search terms (e.g., 'medicare advantage geographic disparities SVI'). "
                        "For 'fetch_article': the URL to fetch."
                    ),
                },
                "own_finding": {
                    "type": "string",
                    "description": "For 'compare_findings': your own quantitative finding to compare.",
                },
                "published_finding": {
                    "type": "string",
                    "description": "For 'compare_findings': the published result to compare against.",
                },
                "source_title": {
                    "type": "string",
                    "description": "For 'compare_findings': title/citation of the published source.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "For 'search': maximum number of results to return (default 10).",
                },
            },
            "required": ["operation"],
        }

    def _execute(
        self,
        operation: str = "",
        query: str = "",
        own_finding: str = "",
        published_finding: str = "",
        source_title: str = "",
        max_results: int = 10,
        **kwargs,
    ) -> ToolResult:
        if not operation:
            return ToolResult(success=False, error="Missing 'operation'.")

        dispatch = {
            "search": self._search,
            "fetch_article": self._fetch_article,
            "compare_findings": self._compare_findings,
        }

        handler = dispatch.get(operation)
        if not handler:
            return ToolResult(
                success=False,
                error=f"Unknown operation '{operation}'. Use: {list(dispatch.keys())}",
            )

        return handler(
            query=query,
            own_finding=own_finding,
            published_finding=published_finding,
            source_title=source_title,
            max_results=max_results,
        )

    # ------------------------------------------------------------------
    # Search for published research
    # ------------------------------------------------------------------

    def _search(self, query, max_results, **kwargs) -> ToolResult:
        if not query:
            return ToolResult(success=False, error="'query' is required for search.")

        results = []

        # Try known reference URLs from config first
        reference_matches = self._match_references(query)
        if reference_matches:
            results.append({
                "source": "configured_references",
                "articles": reference_matches,
            })

        # Search via DuckDuckGo HTML (no API key needed)
        try:
            web_results = self._search_duckduckgo(query, max_results)
            if web_results:
                results.append({
                    "source": "web_search",
                    "articles": web_results,
                })
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            results.append({
                "source": "web_search",
                "error": str(e),
            })

        # Search Google Scholar
        try:
            scholar_results = self._search_scholar(query, max_results)
            if scholar_results:
                results.append({
                    "source": "google_scholar",
                    "articles": scholar_results,
                })
        except Exception as e:
            logger.warning(f"Scholar search failed: {e}")

        total_articles = sum(
            len(r.get("articles", [])) for r in results if "articles" in r
        )

        if total_articles == 0:
            return ToolResult(
                success=True,
                data={"query": query, "results": results, "total": 0},
                summary=(
                    f"No results found for '{query}'. Try different search terms, "
                    "or use fetch_article with a specific URL."
                ),
            )

        return ToolResult(
            success=True,
            data={"query": query, "results": results, "total": total_articles},
            summary=f"Found {total_articles} articles for '{query}'.",
        )

    def _match_references(self, query: str) -> list[dict]:
        """Check if query matches any configured research references."""
        matches = []
        query_lower = query.lower()
        for ref in self._cfg.research_references:
            # Check if any keywords match
            keyword_match = any(kw.lower() in query_lower for kw in ref.keywords)
            title_match = any(word in ref.title.lower() for word in query_lower.split() if len(word) > 3)
            if keyword_match or title_match:
                entry = {"title": ref.title, "description": ref.description}
                if ref.url:
                    entry["url"] = ref.url
                matches.append(entry)
        return matches

    def _search_duckduckgo(self, query: str, max_results: int) -> list[dict]:
        """Search DuckDuckGo HTML for research articles."""
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query + ' research study')}"
        headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}

        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        articles = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for result in soup.select(".result")[:max_results]:
                title_el = result.select_one(".result__title a")
                snippet_el = result.select_one(".result__snippet")
                if title_el:
                    articles.append({
                        "title": title_el.get_text(strip=True),
                        "url": title_el.get("href", ""),
                        "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                    })
        except ImportError:
            # BeautifulSoup not available — extract with regex
            for match in re.finditer(r'class="result__title".*?<a.*?href="(.*?)".*?>(.*?)</a>', resp.text, re.DOTALL):
                articles.append({
                    "title": re.sub(r"<.*?>", "", match.group(2)).strip(),
                    "url": match.group(1),
                })
                if len(articles) >= max_results:
                    break

        return articles

    def _search_scholar(self, query: str, max_results: int) -> list[dict]:
        """Search Google Scholar for academic papers."""
        url = f"https://scholar.google.com/scholar?q={quote_plus(query)}&hl=en"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []

        articles = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for result in soup.select(".gs_ri")[:max_results]:
                title_el = result.select_one(".gs_rt a")
                snippet_el = result.select_one(".gs_rs")
                meta_el = result.select_one(".gs_a")
                if title_el:
                    articles.append({
                        "title": title_el.get_text(strip=True),
                        "url": title_el.get("href", ""),
                        "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                        "citation": meta_el.get_text(strip=True) if meta_el else "",
                    })
        except ImportError:
            pass

        return articles

    # ------------------------------------------------------------------
    # Fetch and extract article content (delegates to WebScraperTool)
    # ------------------------------------------------------------------

    def _fetch_article(self, query, **kwargs) -> ToolResult:
        if not query:
            return ToolResult(success=False, error="'query' (URL) is required for fetch_article.")

        # Delegate to the shared web scraper (handles JS rendering, PDFs, etc.)
        return self._scraper.execute(operation="fetch_page", url=query)

    # ------------------------------------------------------------------
    # Compare findings
    # ------------------------------------------------------------------

    def _compare_findings(
        self, own_finding, published_finding, source_title, **kwargs,
    ) -> ToolResult:
        if not own_finding:
            return ToolResult(success=False, error="'own_finding' is required.")
        if not published_finding:
            return ToolResult(success=False, error="'published_finding' is required.")

        comparison = {
            "own_finding": own_finding,
            "published_finding": published_finding,
            "source": source_title or "Unknown source",
            "comparison_framework": {
                "direction": (
                    "Determine: Do both findings point in the same direction? "
                    "(e.g., both show higher SVI → lower penetration)"
                ),
                "magnitude": (
                    "Compare effect sizes. Is your measured effect similar, larger, or smaller "
                    "than the published result?"
                ),
                "methodology": (
                    "Note methodological differences. Did the published study use different "
                    "geographic units, time periods, or statistical methods?"
                ),
                "population": (
                    "Compare populations. Does the published study cover the same geographic "
                    "scope and demographic groups?"
                ),
                "implications": (
                    "What does the comparison mean? If findings align, this strengthens "
                    "the evidence. If they diverge, explain why (different methodology, "
                    "time period, population, or a genuine new insight)."
                ),
            },
            "template": (
                f"Our analysis found: {own_finding}\n\n"
                f"Published research ({source_title or 'source'}) found: {published_finding}\n\n"
                "These findings [ALIGN/DIVERGE] because [REASON]. "
                "This [STRENGTHENS/CHALLENGES] our thesis that [THESIS ELEMENT]."
            ),
        }

        return ToolResult(
            success=True,
            data=comparison,
            summary=f"Comparison structured between own finding and '{source_title or 'published source'}'.",
        )
