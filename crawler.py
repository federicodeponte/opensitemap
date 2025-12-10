"""
Sitemap Crawler

Production-ready sitemap crawler with intelligent page classification.
Fetches and labels all URLs from a company's sitemap with automatic
page type detection.
"""

import defusedxml.ElementTree as ET
from typing import Optional, Dict, List
from datetime import datetime
from collections import OrderedDict
import httpx
from httpx import Timeout, Limits
import logging
from urllib.parse import urlparse
import re
import asyncio
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import SitemapPage, SitemapPageList, PageLabel

logger = logging.getLogger(__name__)


class SitemapCrawler:
    """
    Crawls company sitemap and auto-labels all URLs.

    Features:
    - Fetches sitemap from standard locations (/sitemap.xml, /sitemap_index.xml)
    - Handles recursive sitemap_index.xml with concurrent processing
    - Intelligent page type classification (10 page types)
    - LRU caching with configurable TTL
    - Rate limiting and comprehensive error handling
    - Security-first URL validation
    """

    # Default URL patterns for page type classification
    DEFAULT_PATTERNS = {
        "blog": [
            r"\/blog\/?",
            r"\/news\/?",
            r"\/articles\/?",
            r"\/posts\/?",
            r"\/insights\/?",
            r"\/stories\/?",
            r"\/updates\/?",
            r"\/press\/?",
        ],
        "product": [
            r"\/products?\/?",
            r"\/solutions?\/?",
            r"\/pricing\/?",
            r"\/features\/?",
            r"\/plans\/?",
            r"\/offerings?\/?",
            r"\/store\/?",
            r"\/shop\/?",
            r"\/catalog\/?",
            r"\/deals?\/?",
            r"\/inventory\/?",
        ],
        "service": [
            r"\/services?\/?",
            r"\/consulting\/?",
            r"\/agency\/?",
            r"\/professional-services\/?",
        ],
        "docs": [
            r"\/docs?\/?",
            r"\/documentation\/?",
            r"\/guides?\/?",
            r"\/tutorials?\/?",
            r"\/help\/?",
            r"\/kb\/?",
            r"\/knowledge-base\/?",
            r"\/faq\/?",
        ],
        "resource": [
            r"\/whitepapers?\/?",
            r"\/case-studies?\/?",
            r"\/templates?\/?",
            r"\/tools?\/?",
            r"\/calculators?\/?",
            r"\/webinars?\/?",
            r"\/videos?\/?",
            r"\/ebooks?\/?",
            r"\/reports?\/?",
        ],
        "company": [
            r"\/about\/?",
            r"\/about-us\/?",
            r"\/team\/?",
            r"\/careers?\/?",
            r"\/jobs?\/?",
            r"\/culture\/?",
            r"\/company\/?",
            r"\/who-we-are\/?",
            r"\/mission\/?",
            r"\/vision\/?",
            r"\/values?\/?",
            r"\/leadership\/?",
            r"\/newsroom\/?",
        ],
        "legal": [
            r"\/imprint\/?",
            r"\/impressum\/?",
            r"\/privacy\/?",
            r"\/privacy-policy\/?",
            r"\/terms?\/?",
            r"\/terms-of-service\/?",
            r"\/terms-of-use\/?",
            r"\/legal\/?",
            r"\/disclaimer\/?",
            r"\/cookies?\/?",
            r"\/data-protection\/?",
            r"\/gdpr\/?",
        ],
        "contact": [
            r"\/contact\/?",
            r"\/contact-us\/?",
            r"\/get-in-touch\/?",
            r"\/reach-us\/?",
            r"\/talk-to-us\/?",
            r"\/support\/?",
            r"\/customer-support\/?",
            r"\/help-desk\/?",
            r"\/email-us\/?",
        ],
        "landing": [
            r"\/campaigns?\/?",
            r"\/lp\/?",
            r"\/landing\/?",
            r"\/offers?\/?",
            r"\/promotions?\/?",
            r"\/deals?\/?",
            r"\/promos?\/?",
        ],
    }

    # Dangerous URL protocols that should be rejected for security
    DANGEROUS_PROTOCOLS = [
        'javascript:', 'file:', 'data:', 'vbscript:', 
        'about:', 'chrome:', 'chrome-extension:'
    ]

    def __init__(
        self,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        timeout: Optional[Timeout] = None,
        cache_ttl: int = 3600,
        max_urls: int = 10000,
        max_cache_size: int = 100,
    ):
        """
        Initialize SitemapCrawler.

        Args:
            custom_patterns: Custom URL patterns for classification
            timeout: HTTP timeout configuration 
            cache_ttl: Cache time-to-live in seconds
            max_urls: Maximum URLs to process
            max_cache_size: Maximum cache entries (LRU eviction)

        Raises:
            ValueError: If max_urls or max_cache_size is <= 0, or cache_ttl < 0
        """
        # Validate parameters
        if max_urls <= 0:
            raise ValueError(f"max_urls must be > 0, got {max_urls}")
        if max_cache_size <= 0:
            raise ValueError(f"max_cache_size must be > 0, got {max_cache_size}")
        if cache_ttl < 0:
            raise ValueError(f"cache_ttl must be >= 0, got {cache_ttl}")

        self.patterns = custom_patterns or self.DEFAULT_PATTERNS
        self.timeout = timeout or Timeout(
            connect=5.0,
            read=10.0,
            write=5.0,
            pool=5.0,
        )
        self.cache_ttl = cache_ttl
        self.max_urls = max_urls
        self.max_cache_size = max_cache_size
        
        # LRU cache with thread safety
        self._cache: OrderedDict[str, tuple[SitemapPageList, float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = asyncio.Lock()

    async def crawl(self, company_url: str) -> SitemapPageList:
        """
        Crawl company's sitemap and return labeled pages.

        Args:
            company_url: Company website URL

        Returns:
            SitemapPageList with classified pages

        Raises:
            ValueError: If company_url is invalid
        """
        start_time = time.time()
        
        # Normalize and validate URL
        company_url = company_url.rstrip('/')
        
        if not self._is_valid_url(company_url):
            raise ValueError(f"Invalid company_url: {company_url}")

        logger.info(f"Starting sitemap crawl for {company_url}")

        # Check cache (thread-safe)
        cache_key = f"{company_url}:{self.max_urls}"
        async with self._cache_lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    self._cache_hits += 1
                    self._cache.move_to_end(cache_key)  # LRU
                    duration = time.time() - start_time
                    logger.info(f"Cache hit: {result.count()} URLs in {duration:.2f}s")
                    return result
                else:
                    del self._cache[cache_key]  # Expired

            self._cache_misses += 1

        try:
            # Fetch all URLs from sitemap(s)
            urls = await self._fetch_all_urls(company_url)
            
            if not urls:
                return self._empty_sitemap(company_url)

            # Apply URL limit
            if len(urls) > self.max_urls:
                logger.warning(f"Limiting {len(urls)} URLs to {self.max_urls}")
                urls = urls[:self.max_urls]

            # Classify pages
            pages = []
            for url in urls:
                if self._is_valid_url(url):
                    page = self._classify_page(url)
                    pages.append(page)

            # Create result
            result = SitemapPageList(
                pages=pages,
                company_url=company_url,
                total_urls=len(urls),
                fetch_timestamp=datetime.now().isoformat()
            )

            # Cache result (thread-safe)
            async with self._cache_lock:
                self._cache[cache_key] = (result, time.time())
                self._cache.move_to_end(cache_key)
                
                # LRU eviction
                while len(self._cache) > self.max_cache_size:
                    self._cache.popitem(last=False)

            duration = time.time() - start_time
            logger.info(f"Crawl complete: {len(pages)} URLs in {duration:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Crawl failed for {company_url}: {e}")
            return self._empty_sitemap(company_url)

    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self._cache),
            "max_cache_size": self.max_cache_size
        }

    def clear_cache(self):
        """Clear the cache and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _empty_sitemap(self, company_url: str) -> SitemapPageList:
        """Create empty sitemap for error cases."""
        return SitemapPageList(
            pages=[],
            company_url=company_url,
            total_urls=0,
            fetch_timestamp=datetime.now().isoformat()
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
        reraise=True,
    )
    async def _fetch_all_urls(self, company_url: str) -> List[str]:
        """Fetch all URLs from sitemap(s) with retry logic."""
        all_urls = []
        sitemap_locations = [
            f"{company_url}/sitemap.xml",
            f"{company_url}/sitemap_index.xml",
            f"{company_url}/sitemap/sitemap.xml",
        ]

        # Try www variant
        parsed = urlparse(company_url)
        if parsed.netloc and not parsed.netloc.startswith("www."):
            base = f"{parsed.scheme}://www.{parsed.netloc}"
            sitemap_locations.extend([
                f"{base}/sitemap.xml",
                f"{base}/sitemap_index.xml", 
                f"{base}/sitemap/sitemap.xml",
            ])

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            limits=Limits(max_connections=5, max_keepalive_connections=2)
        ) as client:
            for sitemap_url in sitemap_locations:
                try:
                    await asyncio.sleep(0.5)  # Rate limiting
                    response = await client.get(sitemap_url)
                    response.raise_for_status()

                    root = ET.fromstring(response.content)
                    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

                    # Check for sitemap index
                    sitemaps = root.findall(".//sm:sitemap/sm:loc", ns)
                    if sitemaps:
                        # Concurrent sub-sitemap fetching
                        sub_urls = [elem.text for elem in sitemaps if elem.text]
                        tasks = [
                            self._fetch_sub_sitemap(client, url) 
                            for url in sub_urls
                        ]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        for result in results:
                            if isinstance(result, list):
                                all_urls.extend(result)
                        
                        if all_urls:
                            break
                    else:
                        # Regular sitemap
                        urls = self._extract_urls(response.content)
                        all_urls.extend(urls)
                        break

                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (404, 403, 401):
                        continue  # Try next location
                    if e.response.status_code in (503, 429, 500, 502, 504):
                        raise  # Retry
                    continue
                except (httpx.TimeoutException, ET.ParseError):
                    continue

        return list(set(all_urls))  # Deduplicate

    async def _fetch_sub_sitemap(self, client: httpx.AsyncClient, url: str) -> List[str]:
        """Fetch URLs from a sub-sitemap."""
        try:
            await asyncio.sleep(0.2)  # Rate limiting
            response = await client.get(url)
            response.raise_for_status()
            return self._extract_urls(response.content)
        except Exception as e:
            logger.debug(f"Sub-sitemap fetch failed {url}: {e}")
            return []

    @staticmethod
    def _extract_urls(content: bytes) -> List[str]:
        """Extract URLs from sitemap XML."""
        urls = []
        try:
            root = ET.fromstring(content)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            
            for url_elem in root.findall(".//sm:url/sm:loc", ns):
                if url_elem.text:
                    urls.append(url_elem.text)
        except ET.ParseError:
            pass
        
        return urls

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL for security and format."""
        if not url or not isinstance(url, str):
            return False
        
        try:
            # Security: Check for dangerous protocols
            url_lower = url.lower().strip()
            if any(url_lower.startswith(proto) for proto in self.DANGEROUS_PROTOCOLS):
                return False
            
            parsed = urlparse(url)
            return (
                parsed.scheme in ['http', 'https'] and
                parsed.netloc and
                '.' in parsed.netloc
            )
        except Exception:
            return False

    def _classify_page(self, url: str) -> SitemapPage:
        """Classify page based on URL patterns."""
        path = urlparse(url).path.lower()
        
        # Score each label
        scores: Dict[PageLabel, float] = {
            label: 0.0 for label in self.patterns.keys()  # type: ignore
        }
        scores["other"] = 0.1  # Default score

        for label, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, path):
                    scores[label] += 0.4

        # Best match
        best_label: PageLabel = max(scores, key=scores.get)  # type: ignore
        confidence = min(scores[best_label], 1.0)
        title = self._extract_title_from_url(url)

        return SitemapPage(
            url=url,
            label=best_label,
            title=title,
            path=path,
            confidence=confidence,
        )

    @staticmethod
    def _extract_title_from_url(url: str) -> str:
        """Extract human-readable title from URL."""
        path = urlparse(url).path
        parts = path.rstrip("/").split("/")
        
        # Get last non-empty part
        for part in reversed(parts):
            if part:
                # Convert slug to title
                title = part.replace("-", " ").replace("_", " ")
                return " ".join(word.capitalize() for word in title.split()) or "Untitled"
        
        return "Untitled"