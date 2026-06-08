"""Scrape public information from centr-krasok.kz for the bot knowledge base.

The script crawls same-domain pages, extracts readable text and common product
fields, and writes both machine-readable JSONL and a compact text knowledge file.
It uses only dependencies already present in this project (`httpx`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse

import httpx


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://centr-krasok.kz/"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "scraped"
DEFAULT_KNOWLEDGE_FILE = ROOT / "data" / "company_knowledge.txt"

SKIP_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".rar",
    ".svg",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}

NOISE_PHRASES = {
    "в корзину",
    "previousnext",
    "посмотреть",
    "оставить заявку",
    "имя*",
    "телефон*",
}


@dataclass
class PageRecord:
    url: str
    title: str
    description: str
    headings: list[str]
    text: str
    product: dict[str, str | list[str]]
    links: list[str]


class ReadableHTMLParser(HTMLParser):
    """Small dependency-free parser for links, metadata, and visible text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.text_parts: list[str] = []
        self.headings: list[str] = []
        self.meta: dict[str, str] = {}
        self.title = ""
        self._skip_depth = 0
        self._current_heading: str | None = None
        self._heading_parts: list[str] = []
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {name.lower(): value or "" for name, value in attrs}

        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return

        if tag == "a" and attr.get("href"):
            self.links.append(attr["href"])
        elif tag == "meta":
            key = (attr.get("property") or attr.get("name") or "").lower()
            content = attr.get("content", "").strip()
            if key and content:
                self.meta[key] = content
        elif tag == "title":
            self._in_title = True
            self._title_parts = []
        elif tag in {"h1", "h2", "h3"}:
            self._current_heading = tag
            self._heading_parts = []
        elif tag in {"br", "p", "li", "div", "section", "article", "tr"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
            return

        if tag == "title":
            self._in_title = False
            self.title = " ".join(self._title_parts).strip()
        elif self._current_heading == tag:
            heading = clean_spaces(" ".join(self._heading_parts))
            if heading:
                self.headings.append(heading)
            self._current_heading = None
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return

        value = data.strip()
        if not value:
            return

        if self._in_title:
            self._title_parts.append(value)
        if self._current_heading:
            self._heading_parts.append(value)
        self.text_parts.append(value)


def clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_text(parts: Iterable[str], max_chars: int) -> str:
    lines: list[str] = []
    seen: set[str] = set()

    for part in parts:
        for raw_line in re.split(r"[\r\n]+", part):
            line = clean_spaces(raw_line)
            lower = line.lower()
            if len(line) < 3 or lower in NOISE_PHRASES or lower in seen:
                continue
            seen.add(lower)
            lines.append(line)

    text = "\n".join(lines)
    return text[:max_chars].rstrip()


def same_site_url(base_url: str, href: str) -> str | None:
    href = href.strip()
    if not href:
        return None

    absolute = urljoin(base_url, href)
    absolute, _fragment = urldefrag(absolute)
    parsed = urlparse(absolute)
    base = urlparse(base_url)

    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() != base.netloc.lower():
        return None
    if Path(parsed.path).suffix.lower() in SKIP_EXTENSIONS:
        return None

    path = parsed.path.rstrip("/") or "/"
    return parsed._replace(path=path, query="").geturl().rstrip("/")


def unique_matches(pattern: str, text: str) -> list[str]:
    values = {
        clean_spaces(match)
        for match in re.findall(pattern, text, flags=re.IGNORECASE)
        if clean_spaces(match)
    }
    return sorted(values)


def is_probable_product(url: str, text: str, title: str) -> bool:
    if not urlparse(url).path.startswith("/catalog/"):
        return False

    markers = ["артикул", "бренд", "остаток", "kzt", "тг"]
    haystack = f"{title}\n{text}".lower()
    sku_values = unique_matches(r"Артикул\s+([A-Za-zА-Яа-я0-9._/-]+)", text)
    return len(sku_values) == 1 and sum(marker in haystack for marker in markers) >= 3


def extract_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return clean_spaces(match.group(1)) if match else ""


def extract_product(
    url: str, text: str, title: str, headings: list[str]
) -> dict[str, str | list[str]]:
    if not is_probable_product(url, text, title):
        return {}

    generic_headings = {"популярные товары", "товары со скидкой", "каталог"}
    product_name = next(
        (heading for heading in headings if heading.lower() not in generic_headings),
        title,
    )
    prices = re.findall(r"\d(?:[\d \u00a0]*\d)?\s*(?:KZT|тг)", text, flags=re.IGNORECASE)
    sku_values = unique_matches(r"Артикул\s+([A-Za-zА-Яа-я0-9._/-]+)", text)
    brand_values = unique_matches(r"Бренд\s+([^\n]+)", text)
    stock = re.findall(r"Остаток\s+[^\n]+", text, flags=re.IGNORECASE)

    product: dict[str, str | list[str]] = {
        "name": clean_spaces(product_name),
        "sku": sku_values[0] if sku_values else "",
        "brand": brand_values[0] if len(brand_values) == 1 else "",
        "price": clean_spaces(prices[-1]) if prices else "",
        "stock": [clean_spaces(item) for item in stock],
    }
    return {key: value for key, value in product.items() if value}


def parse_page(url: str, html: str, max_text_chars: int) -> PageRecord:
    parser = ReadableHTMLParser()
    parser.feed(html)

    title = clean_spaces(parser.meta.get("og:title", "") or parser.title)
    description = clean_spaces(
        parser.meta.get("description", "") or parser.meta.get("og:description", "")
    )
    text = normalize_text(parser.text_parts, max_text_chars)
    links = sorted(
        {
            link
            for href in parser.links
            if (link := same_site_url(url, href)) is not None
        }
    )

    return PageRecord(
        url=url,
        title=title,
        description=description,
        headings=parser.headings[:12],
        text=text,
        product=extract_product(url, text, title, parser.headings),
        links=links,
    )


async def fetch(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[warn] skip {url}: {exc}")
        return None

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        return None
    return response.text


async def crawl(args: argparse.Namespace) -> list[PageRecord]:
    start_url = same_site_url(args.base_url, args.base_url) or args.base_url.rstrip("/")
    queue: deque[str] = deque([start_url])
    seen: set[str] = set()
    records: list[PageRecord] = []

    headers = {
        "User-Agent": (
            "centr-krasok-bot-scraper/1.0 "
            "(knowledge-base updater; contact info.online@abis.kz)"
        )
    }

    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        while queue and len(records) < args.max_pages:
            batch: list[str] = []
            while queue and len(batch) < args.concurrency and len(records) + len(batch) < args.max_pages:
                url = queue.popleft()
                if url in seen:
                    continue
                seen.add(url)
                batch.append(url)

            if not batch:
                continue

            pages = await asyncio.gather(*(fetch(client, url) for url in batch))
            for url, html in zip(batch, pages):
                if not html:
                    continue
                record = parse_page(url, html, args.max_text_chars)
                records.append(record)
                print(f"[ok] {len(records):>3}/{args.max_pages} {url}")

                for link in record.links:
                    if link not in seen and len(seen) + len(queue) < args.max_urls_seen:
                        queue.append(link)

            if args.delay:
                await asyncio.sleep(args.delay)

    return records


def write_jsonl(records: list[PageRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def common_text_lines(records: list[PageRecord]) -> set[str]:
    """Find site-wide boilerplate lines that should not dominate excerpts."""

    counts: dict[str, int] = {}
    for record in records:
        page_lines = {
            clean_spaces(line)
            for line in record.text.splitlines()
            if len(clean_spaces(line)) >= 3
        }
        for line in page_lines:
            counts[line] = counts.get(line, 0) + 1

    threshold = max(3, len(records) // 5)
    return {line for line, count in counts.items() if count >= threshold}


def is_low_value_line(line: str) -> bool:
    lower = line.lower()
    return (
        lower in NOISE_PHRASES
        or lower in {"рус", "русский", "қазақша", "english", "главная", "каталог"}
        or "ваш город" in lower
        or "выберите город" in lower
        or "время работы" in lower
        or re.fullmatch(r"\+?\d[\d\s()+-]{7,}", line) is not None
    )


def page_excerpt(record: PageRecord, common_lines: set[str], max_lines: int = 18) -> str:
    lines: list[str] = []
    seen: set[str] = set()

    for raw_line in record.text.splitlines():
        line = clean_spaces(raw_line)
        if len(line) < 4 or line in common_lines or is_low_value_line(line):
            continue
        lower = line.lower()
        if lower in seen:
            continue
        seen.add(lower)
        lines.append(line)
        if len(lines) >= max_lines:
            break

    if lines:
        return "\n".join(lines)
    return record.description


def write_knowledge(records: list[PageRecord], path: Path, source_url: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    product_records = [record for record in records if record.product]
    info_records = [record for record in records if not record.product]
    boilerplate_lines = common_text_lines(records)

    lines = [
        "# Центр Красок #1 — scraped knowledge",
        "",
        f"Источник: {source_url}",
        f"Дата сбора: {now}",
        f"Страниц собрано: {len(records)}",
        "",
    ]

    if product_records:
        lines.extend(["## Товары", ""])
        for record in product_records:
            product = record.product
            lines.append(f"### {product.get('name') or record.title}")
            if product.get("brand"):
                lines.append(f"- Бренд: {product['brand']}")
            if product.get("sku"):
                lines.append(f"- Артикул: {product['sku']}")
            if product.get("price"):
                lines.append(f"- Цена: {product['price']}")
            for stock_item in product.get("stock", []):
                lines.append(f"- {stock_item}")
            lines.append(f"- Ссылка: {record.url}")
            lines.append("")

    if info_records:
        lines.extend(["## Страницы сайта", ""])
        for record in info_records:
            title = record.title or (record.headings[0] if record.headings else record.url)
            excerpt = page_excerpt(record, boilerplate_lines, args_excerpt_line_count(record.text))
            lines.append(f"### {title}")
            lines.append(f"Ссылка: {record.url}")
            if excerpt:
                lines.append(excerpt)
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def args_excerpt_line_count(text: str) -> int:
    return 18 if len(text) < 4_000 else 24


def replace_knowledge(scraped_path: Path, knowledge_path: Path) -> None:
    backup_path = knowledge_path.with_suffix(
        f"{knowledge_path.suffix}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    )
    shutil.copy2(knowledge_path, backup_path)
    shutil.copy2(scraped_path, knowledge_path)
    print(f"[ok] backed up old knowledge to {backup_path}")
    print(f"[ok] updated bot knowledge file {knowledge_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape centr-krasok.kz into JSONL and a bot knowledge text file."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument("--max-urls-seen", type=int, default=600)
    parser.add_argument("--max-text-chars", type=int, default=8_000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--update-knowledge",
        action="store_true",
        help="Replace data/company_knowledge.txt with the scraped knowledge file.",
    )
    parser.add_argument("--knowledge-file", type=Path, default=DEFAULT_KNOWLEDGE_FILE)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = await crawl(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = args.output_dir / f"centr_krasok_pages_{timestamp}.jsonl"
    knowledge_path = args.output_dir / f"company_knowledge_scraped_{timestamp}.txt"

    write_jsonl(records, jsonl_path)
    write_knowledge(records, knowledge_path, args.base_url)

    print(f"[ok] wrote page data: {jsonl_path}")
    print(f"[ok] wrote knowledge text: {knowledge_path}")

    if args.update_knowledge:
        replace_knowledge(knowledge_path, args.knowledge_file)


if __name__ == "__main__":
    asyncio.run(main())
