"""
Command-line interface for OpenSitemap.

Provides a simple CLI for crawling and analyzing sitemaps from the command line.
"""

import argparse
import asyncio
import json
import sys
from typing import Optional

from . import SitemapCrawler, __version__


def format_results(sitemap_pages, output_format: str = "summary") -> str:
    """Format crawl results for display."""
    if output_format == "json":
        # Convert to JSON-serializable format
        data = {
            "company_url": sitemap_pages.company_url,
            "total_pages": sitemap_pages.count(),
            "fetch_timestamp": sitemap_pages.fetch_timestamp,
            "label_summary": sitemap_pages.label_summary(),
            "site_analysis": sitemap_pages.analyze_site_structure(),
            "pages": [
                {
                    "url": page.url,
                    "label": page.label,
                    "title": page.title,
                    "confidence": page.confidence
                }
                for page in sitemap_pages.pages
            ]
        }
        return json.dumps(data, indent=2)
    
    elif output_format == "urls":
        return "\n".join(sitemap_pages.get_all_urls())
    
    elif output_format == "blogs":
        blog_urls = sitemap_pages.get_blog_urls()
        return "\n".join(blog_urls) if blog_urls else "No blog URLs found"
    
    else:  # summary format
        summary = sitemap_pages.label_summary()
        analysis = sitemap_pages.analyze_site_structure()
        
        output = []
        output.append(f"🗺️  Sitemap Analysis for {sitemap_pages.company_url}")
        output.append("=" * 60)
        output.append(f"Total pages crawled: {sitemap_pages.count()}")
        output.append(f"Fetch timestamp: {sitemap_pages.fetch_timestamp}")
        output.append("")
        
        output.append("📊 Page Type Breakdown:")
        for page_type, count in summary.items():
            if count > 0:
                percentage = count / sitemap_pages.count() * 100
                output.append(f"  {page_type:12s}: {count:4d} pages ({percentage:5.1f}%)")
        
        output.append("")
        output.append("🎯 Site Analysis:")
        output.append(f"  Site type: {analysis['site_type']}")
        output.append(f"  Content volume: {analysis['content_volume']}")
        output.append(f"  Has blog: {'Yes' if analysis['has_blog'] else 'No'}")
        
        if analysis['content_focus']:
            output.append("")
            output.append("🔍 Content Focus:")
            for focus_type, percentage in analysis['content_focus'].items():
                if percentage > 5:  # Only show significant percentages
                    output.append(f"  {focus_type:15s}: {percentage:5.1f}%")
        
        return "\n".join(output)


async def crawl_sitemap(
    url: str,
    max_urls: int = 10000,
    timeout: int = 10,
    output_format: str = "summary",
    verbose: bool = False
) -> Optional[str]:
    """Crawl a sitemap and return formatted results."""
    import logging
    
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    try:
        crawler = SitemapCrawler(max_urls=max_urls)
        
        if verbose:
            print(f"Crawling sitemap for: {url}", file=sys.stderr)
        
        sitemap_pages = await crawler.crawl(url)
        
        if verbose:
            stats = crawler.get_cache_stats()
            print(f"Cache stats: {stats}", file=sys.stderr)
        
        return format_results(sitemap_pages, output_format)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return None


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenSitemap - Advanced sitemap crawler and URL classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  opensitemap https://stripe.com
  opensitemap https://example.com --format json --max-urls 500
  opensitemap https://site.com --format blogs --verbose
        """
    )
    
    parser.add_argument("url", help="Website URL to crawl")
    parser.add_argument(
        "--format", 
        choices=["summary", "json", "urls", "blogs"],
        default="summary",
        help="Output format (default: summary)"
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=10000,
        help="Maximum URLs to process (default: 10000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"opensitemap {__version__}"
    )
    
    args = parser.parse_args()
    
    # Run the async crawl function
    result = asyncio.run(crawl_sitemap(
        args.url,
        args.max_urls,
        args.timeout,
        args.format,
        args.verbose
    ))
    
    if result is None:
        return 1
    
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())