"""
OpenSitemap - Advanced sitemap crawler and URL classifier

A production-ready Python library for crawling and analyzing website sitemaps
with intelligent page classification. Perfect for SEO analysis, competitive
research, and content discovery.
"""

__version__ = "1.0.0"
__author__ = "Federico Deponte"
__email__ = "federico@isaacsecurity.com"
__license__ = "MIT"

from .crawler import SitemapCrawler
from .models import SitemapPage, SitemapPageList, PageLabel

__all__ = [
    "SitemapCrawler",
    "SitemapPage", 
    "SitemapPageList",
    "PageLabel",
    "__version__",
]