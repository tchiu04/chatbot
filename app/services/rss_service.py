"""
RSS feed service for ingesting news articles.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from newspaper import Article

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class NewsArticle:
    """Data class for news articles."""
    
    def __init__(
        self,
        title: str,
        content: str,
        url: str,
        published_date: Optional[datetime] = None,
        source: Optional[str] = None,
        summary: Optional[str] = None,
        authors: Optional[List[str]] = None
    ):
        self.title = title
        self.content = content
        self.url = url
        self.published_date = published_date
        self.source = source
        self.summary = summary
        self.authors = authors or []
    
    def to_dict(self) -> Dict:
        """Convert article to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "source": self.source,
            "summary": self.summary,
            "authors": self.authors
        }
    
    @property
    def text_content(self) -> str:
        """Get combined text content for processing."""
        return f"{self.title}\n\n{self.content}"


class RSSService:
    """Service for fetching and processing RSS feeds."""
    
    def __init__(self):
        self.settings = get_settings()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"
        })
    
    async def fetch_feed_articles(self, feed_url: str, max_articles: int = 50) -> List[NewsArticle]:
        """
        Fetch articles from an RSS feed.
        
        Args:
            feed_url: RSS feed URL
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of NewsArticle objects
        """
        try:
            logger.info(f"Fetching RSS feed: {feed_url}")
            
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"RSS feed may have issues: {feed_url}")
            
            articles = []
            source_name = self._extract_source_name(feed_url, feed)
            
            # Process feed entries
            for entry in feed.entries[:max_articles]:
                try:
                    article = await self._process_feed_entry(entry, source_name)
                    if article:
                        articles.append(article)
                        
                except Exception as e:
                    logger.warning(f"Failed to process entry {entry.get('link', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from {feed_url}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []
    
    async def fetch_all_feeds(self, max_articles_per_feed: int = 50) -> List[NewsArticle]:
        """
        Fetch articles from all configured RSS feeds.
        
        Args:
            max_articles_per_feed: Maximum articles per feed
            
        Returns:
            List of all NewsArticle objects
        """
        all_articles = []
        
        for feed_url in self.settings.rss_feeds:
            articles = await self.fetch_feed_articles(feed_url, max_articles_per_feed)
            all_articles.extend(articles)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        logger.info(f"Fetched {len(unique_articles)} unique articles from {len(self.settings.rss_feeds)} feeds")
        return unique_articles
    
    async def _process_feed_entry(self, entry: Dict, source_name: str) -> Optional[NewsArticle]:
        """Process a single RSS feed entry."""
        try:
            # Extract basic information
            title = entry.get("title", "").strip()
            url = entry.get("link", "").strip()
            
            if not title or not url:
                return None
            
            # Extract published date
            published_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_date = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published_date = datetime(*entry.updated_parsed[:6])
            
            # Extract summary from RSS
            rss_summary = entry.get("summary", "").strip()
            
            # Try to extract full content
            content = await self._extract_article_content(url)
            
            # Use RSS summary if content extraction fails
            if not content and rss_summary:
                content = self._clean_html(rss_summary)
            
            if not content:
                logger.warning(f"No content extracted for {url}")
                return None
            
            # Extract authors
            authors = []
            if hasattr(entry, "author"):
                authors = [entry.author]
            elif hasattr(entry, "authors"):
                authors = [author.name for author in entry.authors if hasattr(author, "name")]
            
            return NewsArticle(
                title=title,
                content=content,
                url=url,
                published_date=published_date,
                source=source_name,
                summary=rss_summary[:500] if rss_summary else None,
                authors=authors
            )
            
        except Exception as e:
            logger.error(f"Failed to process feed entry: {e}")
            return None
    
    async def _extract_article_content(self, url: str) -> str:
        """Extract full article content from URL."""
        try:
            # Use newspaper3k for content extraction
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text:
                return article.text.strip()
            
            # Fallback to basic web scraping
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Try common content selectors
            content_selectors = [
                "article",
                ".article-content",
                ".post-content",
                ".entry-content",
                "[role='main']",
                "main"
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    text = content_elem.get_text(separator="\n", strip=True)
                    if len(text) > 200:  # Minimum content length
                        return text
            
            # Fallback to body text
            body_text = soup.get_text(separator="\n", strip=True)
            return body_text if len(body_text) > 200 else ""
            
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {e}")
            return ""
    
    def _extract_source_name(self, feed_url: str, feed: feedparser.FeedParserDict) -> str:
        """Extract source name from feed."""
        # Try to get from feed metadata
        if hasattr(feed.feed, "title"):
            return feed.feed.title
        
        # Extract from URL
        parsed_url = urlparse(feed_url)
        domain = parsed_url.netloc.lower()
        
        # Clean up common domain patterns
        domain = re.sub(r"^(www\.|feeds\.|rss\.)", "", domain)
        domain = re.sub(r"\.com$|\.org$|\.net$", "", domain)
        
        return domain.capitalize()
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract text."""
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n", strip=True)
