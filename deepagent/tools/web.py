"""Web tools: WebSearch (DuckDuckGo HTML) and WebFetch (fetch + to-markdown)."""
from __future__ import annotations

import html
import json
import re
import urllib.parse

import httpx

from .base import ToolRegistry, ToolResult, integer, schema, string

registry = ToolRegistry()

UA = "Mozilla/5.0 (X11; Linux x86_64) deepagent/0.1 (+coding-agent)"


@registry.add(
    "WebSearch",
    "Search the web and return a ranked list of results (title, url, snippet). "
    "Use it to find documentation, error explanations, package versions or "
    "current information beyond the model's training cutoff.",
    schema(
        ["query"],
        query=string("The search query."),
        limit=integer("Max results (default 8)."),
    ),
    read_only=True,
)
def _web_search(args: dict, ctx) -> ToolResult:
    query = str(args["query"]).strip()
    if not query:
        return ToolResult.error("query is empty.")
    limit = int(args.get("limit") or 8)
    try:
        results = _ddg(query, limit)
    except Exception as exc:
        return ToolResult.error(f"search failed: {exc}")
    if not results:
        return ToolResult.ok("No results.", summary="0 results")
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    return ToolResult.ok("\n".join(lines), summary=f"{len(results)} results for {query[:40]!r}")


def _ddg(query: str, limit: int) -> list[dict]:
    url = "https://html.duckduckgo.com/html/"
    with httpx.Client(timeout=30, headers={"User-Agent": UA}, follow_redirects=True) as c:
        r = c.post(url, data={"q": query})
        r.raise_for_status()
        body = r.text
    out: list[dict] = []
    for m in re.finditer(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', body, re.S
    ):
        href, title = m.group(1), _strip(m.group(2))
        if href.startswith("//duckduckgo.com/l/?"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = q.get("uddg", [href])[0]
        snippet = ""
        tail = body[m.end(): m.end() + 1200]
        sm = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', tail, re.S)
        if sm:
            snippet = _strip(sm.group(1))
        out.append({"title": title, "url": href, "snippet": snippet})
        if len(out) >= limit:
            break
    return out


@registry.add(
    "WebFetch",
    "Fetch a URL and return its main text content converted to Markdown. Use "
    "after WebSearch to read a specific page, or to fetch a known doc/API URL.",
    schema(
        ["url"],
        url=string("The http(s) URL to fetch."),
        max_chars=integer("Truncate the extracted text (default 20000)."),
    ),
    read_only=True,
)
def _web_fetch(args: dict, ctx) -> ToolResult:
    url = str(args["url"]).strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    max_chars = int(args.get("max_chars") or 20000)
    try:
        with httpx.Client(timeout=40, headers={"User-Agent": UA},
                          follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            text = r.text
    except Exception as exc:
        return ToolResult.error(f"fetch failed: {exc}")

    if "application/json" in ctype:
        try:
            text = json.dumps(r.json(), indent=2)[:max_chars]
            return ToolResult.ok(text, summary=f"json {url}")
        except Exception:
            pass
    if "text/html" in ctype or "<html" in text[:2000].lower():
        text = _html_to_md(text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... [truncated, {len(text)} chars]"
    return ToolResult.ok(text or "[empty page]", summary=f"fetched {url}")


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def _html_to_md(raw: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore
        from markdownify import markdownify  # type: ignore

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        return markdownify(str(main), heading_style="ATX")
    except Exception:
        # dependency-free fallback
        raw = re.sub(r"(?is)<(script|style).*?</\1>", "", raw)
        raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
        raw = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", raw)
        return _strip(raw)


__all__ = ["registry"]
