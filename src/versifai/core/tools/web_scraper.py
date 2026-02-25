"""
Tool: scrape_web

Core utility for crawling and scraping web content, including JavaScript-
rendered Single Page Applications (SPAs).

Capabilities:
  - discover_site: Parse sitemap.xml to map out a site's full structure
  - fetch_page: Fetch a single page with JS rendering support
  - extract_tables: Extract HTML tables from a page as structured data

Rendering strategy (for JS-heavy sites):
  1. Try raw HTTP request first (fast, works for server-rendered pages)
  2. Detect if the response is a JS shell (React/Vue/Angular loader)
  3. If JS shell detected, render with Playwright headless browser
  4. If Playwright unavailable, try Google's web cache as fallback

This tool is designed to be registered directly with any agent, or used
programmatically by other tools (e.g., LiteratureReviewTool).
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urlparse

import requests

from versifai.core.tools.base import BaseTool, ToolResult

logger = logging.getLogger("agent.core.web_scraper")

# Markers that indicate a page is a JS shell with no real content
_JS_SHELL_MARKERS = [
    "you need to enable javascript",
    "please enable javascript",
    "this app requires javascript",
    "loading...</",
    '<div id="root"></div>',
    '<div id="app"></div>',
    '<div id="__next"></div>',
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Maximum content length returned to the agent
_MAX_CONTENT_CHARS = 25000


class WebScraperTool(BaseTool):
    """
    Crawls and scrapes web content, with support for JavaScript-rendered
    sites via headless browser rendering.
    """

    def __init__(self) -> None:
        super().__init__()
        # In-memory cache for rendered pages (avoids re-fetching within a session)
        self._page_cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "scrape_web"

    @property
    def description(self) -> str:
        return (
            "Crawl and scrape web content, including JavaScript-rendered sites "
            "(React, Vue, Angular SPAs). Operations:\n"
            "- 'discover_site': Discover a site's structure from its sitemap.xml. "
            "Provide the site's base URL (e.g., 'https://example.gov'). Returns "
            "an organized map of all pages grouped by section.\n"
            "- 'fetch_page': Fetch and extract text content from a URL. Automatically "
            "detects JavaScript-rendered pages and uses a headless browser to render "
            "them. Works with HTML pages and PDFs.\n"
            "- 'extract_tables': Fetch a page and extract all HTML tables as "
            "structured data (list of rows). Useful for data-heavy report pages.\n\n"
            "Use discover_site first to map out a research site, then fetch_page "
            "to read specific chapters or sections of interest."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": (
                        "Operation: 'discover_site', 'fetch_page', or 'extract_tables'."
                    ),
                },
                "url": {
                    "type": "string",
                    "description": (
                        "For 'discover_site': the base URL of the site (e.g., 'https://usrds-adr.niddk.nih.gov'). "
                        "For 'fetch_page' / 'extract_tables': the full URL of the page to fetch."
                    ),
                },
                "section_filter": {
                    "type": "string",
                    "description": (
                        "For 'discover_site': optional filter to show only pages "
                        "matching this path segment (e.g., '2025' or 'end-stage-renal-disease')."
                    ),
                },
            },
            "required": ["operation", "url"],
        }

    def _execute(
        self,
        operation: str = "",
        url: str = "",
        section_filter: str = "",
        **kwargs,
    ) -> ToolResult:
        if not operation:
            return ToolResult(success=False, error="Missing 'operation'.")
        if not url:
            return ToolResult(success=False, error="Missing 'url'.")

        dispatch = {
            "discover_site": self._discover_site,
            "fetch_page": self._fetch_page,
            "extract_tables": self._extract_tables,
        }

        handler = dispatch.get(operation)
        if not handler:
            return ToolResult(
                success=False,
                error=f"Unknown operation '{operation}'. Use: {list(dispatch.keys())}",
            )

        return handler(url=url, section_filter=section_filter)

    # ------------------------------------------------------------------
    # Public helpers — for programmatic use by other tools
    # ------------------------------------------------------------------

    def fetch_page_text(self, url: str) -> tuple[bool, str]:
        """
        Convenience method for other tools to fetch a page's text content.

        Returns (success, text_or_error).
        """
        result = self._fetch_page(url=url)
        if result.success:
            return True, result.data.get("text", "")
        return False, result.error

    def discover_site_structure(self, base_url: str) -> tuple[bool, dict]:
        """
        Convenience method for other tools to get site structure.

        Returns (success, structure_dict_or_error).
        """
        result = self._discover_site(url=base_url)
        if result.success:
            return True, result.data
        return False, {}

    # ------------------------------------------------------------------
    # Operation: discover_site
    # ------------------------------------------------------------------

    def _discover_site(self, url: str, section_filter: str = "", **kw) -> ToolResult:
        """Parse sitemap.xml and return organized site structure."""
        base_url = url.rstrip("/")
        sitemap_url = f"{base_url}/sitemap.xml"

        try:
            resp = requests.get(sitemap_url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            return ToolResult(
                success=False,
                error=(
                    f"Could not fetch sitemap at {sitemap_url}: {e}\n\n"
                    f"Try fetching the homepage directly with fetch_page to find "
                    f"navigation links, or provide the direct sitemap URL."
                ),
            )

        # Parse the XML sitemap
        try:
            urls = self._parse_sitemap_xml(resp.text, base_url)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to parse sitemap XML: {e}",
            )

        if not urls:
            return ToolResult(
                success=True,
                data={"base_url": base_url, "total_pages": 0, "sections": {}},
                summary=f"Sitemap found at {sitemap_url} but contained no URLs.",
            )

        # Apply section filter
        if section_filter:
            filter_lower = section_filter.lower()
            urls = [u for u in urls if filter_lower in u["path"].lower()]

        # Organize by path segments into sections
        sections = self._organize_by_sections(urls, base_url)

        total = len(urls)
        section_count = len(sections)

        return ToolResult(
            success=True,
            data={
                "base_url": base_url,
                "sitemap_url": sitemap_url,
                "total_pages": total,
                "section_count": section_count,
                "sections": sections,
            },
            summary=(
                f"Discovered {total} pages in {section_count} sections "
                f"from {sitemap_url}"
                + (f" (filtered by '{section_filter}')" if section_filter else "")
                + "."
            ),
        )

    def _parse_sitemap_xml(self, xml_text: str, base_url: str) -> list[dict]:
        """Parse sitemap XML and extract URL entries."""
        # Strip UTF-8 BOM if present (may appear as \ufeff or as raw bytes decoded wrong)
        xml_text = xml_text.lstrip("\ufeff")
        if xml_text.startswith("\xef\xbb\xbf"):
            xml_text = xml_text[3:]
        # Also handle BOM decoded as ISO-8859-1
        if xml_text.startswith("ï»¿"):
            xml_text = xml_text[3:]

        # Handle XML namespaces (strip default namespace for simpler parsing)
        xml_text_clean = re.sub(r'\s+xmlns\s*=\s*"[^"]*"', "", xml_text, count=1)

        root = ET.fromstring(xml_text_clean)

        urls = []
        # Handle both <url> (urlset) and <sitemap> (sitemapindex)
        for url_el in root.iter():
            if url_el.tag in ("url", "sitemap"):
                loc_el = url_el.find("loc")
                if loc_el is not None and loc_el.text:
                    full_url = loc_el.text.strip()
                    parsed = urlparse(full_url)
                    path = parsed.path.strip("/")

                    lastmod_el = url_el.find("lastmod")
                    lastmod = lastmod_el.text.strip() if lastmod_el is not None and lastmod_el.text else None

                    # Derive a readable title from the URL path
                    slug = path.split("/")[-1] if path else ""
                    title = slug.replace("-", " ").replace("_", " ").title() if slug else base_url

                    urls.append({
                        "url": full_url,
                        "path": path,
                        "title": title,
                        "lastmod": lastmod,
                    })

        return urls

    def _organize_by_sections(self, urls: list[dict], base_url: str) -> dict:
        """Group URLs into hierarchical sections based on path segments."""
        sections: dict[str, list[dict]] = {}

        for entry in urls:
            parts = entry["path"].split("/")

            if len(parts) >= 2:
                # e.g., "2025/chronic-kidney-disease" -> section = "2025 / chronic-kidney-disease"
                section_key = f"{parts[0]} / {parts[1]}" if len(parts) >= 2 else parts[0]
            elif len(parts) == 1:
                section_key = parts[0]
            else:
                section_key = "(root)"

            if section_key not in sections:
                sections[section_key] = []

            sections[section_key].append({
                "title": entry["title"],
                "url": entry["url"],
                "lastmod": entry.get("lastmod"),
            })

        # Sort pages within each section by URL
        for section in sections.values():
            section.sort(key=lambda x: x["url"])

        return sections

    # ------------------------------------------------------------------
    # Operation: fetch_page
    # ------------------------------------------------------------------

    def _fetch_page(self, url: str, **kw) -> ToolResult:
        """Fetch a page, rendering JS if needed."""

        # Check cache first
        if url in self._page_cache:
            text = self._page_cache[url]
            return ToolResult(
                success=True,
                data={"url": url, "text": text, "content_length": len(text), "source": "cache"},
                summary=f"Page from cache ({len(text)} chars).",
            )

        # Step 1: Try raw HTTP fetch
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as e:
            return ToolResult(success=False, error=f"Failed to fetch {url}: {e}")

        content_type = resp.headers.get("content-type", "")

        # Handle PDFs
        if "pdf" in content_type:
            return self._extract_pdf(resp.content, url)

        # Step 2: Check if the HTML is a JS shell
        html = resp.text
        if self._is_js_shell(html):
            logger.info(f"JS shell detected at {url} — attempting browser render")

            # Try Playwright
            rendered_html = self._render_with_playwright(url)

            if rendered_html is None:
                # Try Google cache as fallback
                rendered_html = self._fetch_google_cache(url)

            if rendered_html is None:
                return ToolResult(
                    success=False,
                    error=(
                        f"Page at {url} is a JavaScript-rendered application (React/Vue/Angular SPA). "
                        f"Could not render it because:\n"
                        f"  - Playwright is not installed (pip install playwright && playwright install chromium)\n"
                        f"  - Google cache fallback did not have a cached version\n\n"
                        f"To fix: install Playwright on the cluster, or try a different URL "
                        f"that serves static HTML."
                    ),
                )

            html = rendered_html

        # Step 3: Extract text content
        text = self._extract_html_text(html)

        if not text or len(text) < 50:
            return ToolResult(
                success=True,
                data={"url": url, "text": text, "content_length": len(text), "source": "http"},
                summary=f"Page fetched but content is very short ({len(text)} chars). May need JS rendering.",
            )

        # Truncate
        if len(text) > _MAX_CONTENT_CHARS:
            text = text[:_MAX_CONTENT_CHARS] + "\n\n[... CONTENT TRUNCATED — full page is longer ...]"

        # Cache it
        self._page_cache[url] = text

        source = "browser" if self._is_js_shell(resp.text) else "http"

        return ToolResult(
            success=True,
            data={"url": url, "text": text, "content_length": len(text), "source": source},
            summary=f"Fetched page ({len(text)} chars, via {source}).",
        )

    # ------------------------------------------------------------------
    # Operation: extract_tables
    # ------------------------------------------------------------------

    def _extract_tables(self, url: str, **kw) -> ToolResult:
        """Fetch a page and extract HTML tables as structured data."""
        # First, get the rendered HTML
        html = self._get_rendered_html(url)
        if html is None:
            return ToolResult(success=False, error=f"Could not fetch or render {url}")

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return ToolResult(
                success=False,
                error="Table extraction requires beautifulsoup4. pip install beautifulsoup4",
            )

        soup = BeautifulSoup(html, "html.parser")
        tables = []

        for i, table_el in enumerate(soup.find_all("table")):
            # Extract headers
            headers = []
            thead = table_el.find("thead")
            if thead:
                for th in thead.find_all(["th", "td"]):
                    headers.append(th.get_text(strip=True))

            # Extract rows
            rows = []
            tbody = table_el.find("tbody") or table_el
            for tr in tbody.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells and cells != headers:
                    rows.append(cells)

            # If no explicit headers, use first row
            if not headers and rows:
                headers = rows.pop(0)

            # Get caption if present
            caption_el = table_el.find("caption")
            caption = caption_el.get_text(strip=True) if caption_el else f"Table {i + 1}"

            if headers or rows:
                tables.append({
                    "caption": caption,
                    "headers": headers,
                    "rows": rows[:100],  # Limit rows
                    "total_rows": len(rows),
                })

        if not tables:
            return ToolResult(
                success=True,
                data={"url": url, "tables": [], "count": 0},
                summary=f"No tables found on {url}.",
            )

        # Truncate if too many tables
        total_tables = len(tables)
        if total_tables > 20:
            tables = tables[:20]

        return ToolResult(
            success=True,
            data={"url": url, "tables": tables, "count": total_tables},
            summary=f"Extracted {total_tables} table(s) from {url}.",
        )

    # ------------------------------------------------------------------
    # JS shell detection
    # ------------------------------------------------------------------

    def _is_js_shell(self, html: str) -> bool:
        """
        Detect whether HTML is a JavaScript SPA shell with no real content.

        Checks for telltale signs: empty root divs, "enable JavaScript"
        messages, minimal text content relative to script content.
        """
        html_lower = html.lower()

        # Check for explicit JS-required markers
        for marker in _JS_SHELL_MARKERS:
            if marker in html_lower:
                return True

        # Heuristic: if the page has very little text but lots of script tags,
        # it's likely a JS shell
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove scripts and styles
            for tag in soup.find_all(["script", "style", "link", "meta"]):
                tag.decompose()

            visible_text = soup.get_text(strip=True)

            # If the visible text (excluding scripts) is very short, it's likely a shell
            if len(visible_text) < 200:
                return True

        except ImportError:
            # Without BS4, use a simpler heuristic
            text_only = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text_only = re.sub(r"<style[^>]*>.*?</style>", "", text_only, flags=re.DOTALL)
            text_only = re.sub(r"<[^>]+>", "", text_only)
            text_only = text_only.strip()
            if len(text_only) < 200:
                return True

        return False

    # ------------------------------------------------------------------
    # Rendering strategies
    # ------------------------------------------------------------------

    def _render_with_playwright(self, url: str) -> str | None:
        """
        Render a JS-heavy page using Playwright headless browser.

        Returns the rendered HTML string, or None if Playwright is unavailable.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.info("Playwright not installed — skipping browser render")
            return None

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=60000)

                # Wait a moment for any remaining dynamic content
                page.wait_for_timeout(2000)

                html = page.content()
                browser.close()

            logger.info(f"Playwright rendered {url} ({len(html)} chars)")
            return html

        except Exception as e:
            logger.warning(f"Playwright render failed for {url}: {e}")
            return None

    def _fetch_google_cache(self, url: str) -> str | None:
        """
        Try fetching a page from Google's web cache as a fallback
        when JS rendering is unavailable.

        Returns HTML string or None.
        """
        cache_url = (
            f"https://webcache.googleusercontent.com/search?"
            f"q=cache:{quote_plus(url)}&strip=0"
        )

        try:
            resp = requests.get(cache_url, headers=_HEADERS, timeout=15)
            if resp.status_code == 200 and not self._is_js_shell(resp.text):
                logger.info(f"Google cache hit for {url}")
                return resp.text
        except Exception as e:
            logger.debug(f"Google cache fetch failed: {e}")

        return None

    def _get_rendered_html(self, url: str) -> str | None:
        """Get fully rendered HTML for a URL, using browser if needed."""
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException:
            return None

        html = resp.text

        if self._is_js_shell(html):
            rendered = self._render_with_playwright(url)
            if rendered:
                return rendered
            rendered = self._fetch_google_cache(url)
            if rendered:
                return rendered
            return None

        return html

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def _extract_html_text(self, html: str) -> str:
        """Extract readable text from HTML, preserving structure."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove non-content elements
            for tag in soup.select(
                "script, style, nav, footer, header, aside, "
                ".sidebar, .menu, .navigation, .nav-links, "
                ".cookie-banner, .popup, [role='navigation']"
            ):
                tag.decompose()

            # Try to find the main content area
            main = (
                soup.select_one("article")
                or soup.select_one("main")
                or soup.select_one('[role="main"]')
                or soup.select_one(".content")
                or soup.select_one("#content")
                or soup.select_one(".article-body")
                or soup.select_one(".post-content")
                or soup.body
                or soup
            )

            # Preserve headings and paragraph structure
            parts = []
            for el in main.descendants:
                if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    level = int(el.name[1])
                    prefix = "#" * level
                    parts.append(f"\n{prefix} {el.get_text(strip=True)}\n")
                elif el.name == "p":
                    text = el.get_text(strip=True)
                    if text:
                        parts.append(text + "\n")
                elif el.name == "li":
                    text = el.get_text(strip=True)
                    if text:
                        parts.append(f"  - {text}")
                elif el.name == "figcaption":
                    text = el.get_text(strip=True)
                    if text:
                        parts.append(f"[Figure: {text}]")

            text = "\n".join(parts)

            # If structured extraction got too little, fall back to get_text
            if len(text.strip()) < 200:
                text = main.get_text(separator="\n", strip=True)

        except ImportError:
            # Regex fallback
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def _extract_pdf(self, content: bytes, url: str) -> ToolResult:
        """Extract text from a PDF document."""
        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            text_parts = []
            for page in reader.pages[:30]:
                text_parts.append(page.extract_text() or "")
            text = "\n".join(text_parts)

            if len(text) > _MAX_CONTENT_CHARS:
                text = text[:_MAX_CONTENT_CHARS] + "\n\n[... CONTENT TRUNCATED ...]"

            self._page_cache[url] = text

            return ToolResult(
                success=True,
                data={
                    "url": url,
                    "text": text,
                    "content_length": len(text),
                    "pages": len(reader.pages),
                    "source": "pdf",
                },
                summary=f"Extracted PDF ({len(reader.pages)} pages, {len(text)} chars).",
            )
        except ImportError:
            return ToolResult(
                success=False,
                error="PDF extraction requires 'pypdf'. Install with: pip install pypdf",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"PDF extraction failed: {e}")
