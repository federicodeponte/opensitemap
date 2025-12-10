"""
Sitemap Page Models

Data structures for representing sitemap pages with automatic classification.
"""

from typing import Optional, Literal, List, Dict
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

PageLabel = Literal[
    "blog",
    "product", 
    "service",
    "docs",
    "resource",
    "company",      # About, team, careers, culture
    "legal",        # Imprint, privacy, terms, legal
    "contact",      # Contact, support, help desk
    "landing",      # Landing pages, campaigns
    "other"
]


class SitemapPage(BaseModel):
    """
    Single URL entry from company's sitemap with automatic classification.

    Attributes:
        url: Full or relative URL from sitemap
        label: Auto-detected page type (blog, product, service, docs, resource, other)
        title: Page title (extracted from URL or metadata if available)
        path: URL path for pattern analysis
        confidence: Confidence score for label (0-1, where 1 = very confident)
    """

    url: str = Field(..., description="Full or relative URL from sitemap")
    label: PageLabel = Field(
        default="other",
        description="Auto-detected page type (blog, product, service, docs, resource, other)"
    )
    title: Optional[str] = Field(
        default=None,
        description="Page title (optional, can be extracted from URL or metadata)"
    )
    path: str = Field(..., description="URL path for pattern analysis")
    confidence: float = Field(
        default=0.5,
        description="Confidence score for label classification (0-1)",
        ge=0.0,
        le=1.0
    )

    def is_blog(self) -> bool:
        """Check if page is a blog."""
        return self.label == "blog"

    def is_blog_confident(self, min_confidence: float = 0.7) -> bool:
        """Check if page is a blog with minimum confidence threshold."""
        return self.is_blog() and self.confidence >= min_confidence

    def __repr__(self) -> str:
        """String representation."""
        return f"SitemapPage({self.url}, label={self.label}, confidence={self.confidence:.2f})"

    def __hash__(self) -> int:
        """Make hashable for deduplication."""
        return hash(self.url)

    def __eq__(self, other: object) -> bool:
        """Equality based on URL."""
        if not isinstance(other, SitemapPage):
            return NotImplemented
        return self.url == other.url


class SitemapPageList(BaseModel):
    """
    Collection of sitemap pages with analysis methods.

    Manages the complete labeled sitemap and provides filtering/access methods.
    """

    pages: List[SitemapPage] = Field(
        default_factory=list,
        description="List of pages from sitemap"
    )
    company_url: str = Field(..., description="Company URL from which sitemap was fetched")
    total_urls: int = Field(default=0, description="Total URLs in original sitemap")
    fetch_timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp when sitemap was fetched"
    )

    def get_blogs(self, min_confidence: float = 0.7) -> List[SitemapPage]:
        """Get all blog pages above confidence threshold."""
        return [page for page in self.pages if page.is_blog_confident(min_confidence)]

    def get_by_label(self, label: PageLabel, min_confidence: float = 0.0) -> List[SitemapPage]:
        """Get all pages with specific label."""
        return [
            page for page in self.pages
            if page.label == label and page.confidence >= min_confidence
        ]

    def get_blog_urls(self, min_confidence: float = 0.7) -> List[str]:
        """Get list of blog URLs."""
        return [page.url for page in self.get_blogs(min_confidence)]

    def get_all_urls(self) -> List[str]:
        """Get list of all URLs."""
        return [page.url for page in self.pages]

    def get_urls_by_label(self, label: PageLabel, min_confidence: float = 0.0) -> List[str]:
        """Get list of URLs by label."""
        return [page.url for page in self.get_by_label(label, min_confidence)]

    def deduplicate(self) -> "SitemapPageList":
        """Remove duplicate URLs, keeping first occurrence."""
        seen = set()
        unique_pages = []

        for page in self.pages:
            if page.url not in seen:
                unique_pages.append(page)
                seen.add(page.url)

        new_list = SitemapPageList(
            pages=unique_pages,
            company_url=self.company_url,
            total_urls=self.total_urls,
            fetch_timestamp=self.fetch_timestamp
        )
        return new_list

    def count(self) -> int:
        """Get total page count."""
        return len(self.pages)

    def count_by_label(self, label: PageLabel) -> int:
        """Get count of pages by label."""
        return len(self.get_by_label(label))

    def label_summary(self) -> Dict[PageLabel, int]:
        """Get summary of page counts by label."""
        summary: Dict[PageLabel, int] = {
            "blog": 0,
            "product": 0,
            "service": 0,
            "docs": 0,
            "resource": 0,
            "company": 0,
            "legal": 0,
            "contact": 0,
            "landing": 0,
            "other": 0
        }

        for page in self.pages:
            summary[page.label] += 1

        return summary

    def analyze_site_structure(self) -> Dict[str, any]:
        """
        Analyze site structure and content focus.
        
        Returns:
            Dictionary with site analysis including content focus percentages
            and site type classification.
        """
        page_summary = self.label_summary()
        total_pages = self.count()
        
        if total_pages == 0:
            return {
                "site_type": "unknown",
                "content_focus": {},
                "has_blog": False,
                "content_volume": "none"
            }
        
        # Calculate content focus percentages
        content_focus = {
            "content_heavy": (page_summary.get("blog", 0) + page_summary.get("resource", 0)) / total_pages * 100,
            "product_focus": page_summary.get("product", 0) / total_pages * 100,
            "service_focus": page_summary.get("service", 0) / total_pages * 100,
            "documentation": page_summary.get("docs", 0) / total_pages * 100
        }
        
        # Determine site type
        if content_focus["content_heavy"] > 30:
            site_type = "content_marketing"
        elif content_focus["product_focus"] > 40:
            site_type = "product_focused"
        elif content_focus["service_focus"] > 30:
            site_type = "service_focused"
        else:
            site_type = "corporate"
            
        return {
            "site_type": site_type,
            "content_focus": content_focus,
            "has_blog": page_summary.get("blog", 0) > 0,
            "content_volume": "high" if page_summary.get("blog", 0) > 20 else "medium" if page_summary.get("blog", 0) > 5 else "low"
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"SitemapPageList({len(self.pages)} pages, {self.label_summary()})"