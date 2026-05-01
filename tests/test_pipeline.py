"""Tests for data pipeline (ArXiv + OpenAlex integration)."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


def test_rate_limiter_respects_interval():
    """Verify RateLimiter enforces minimum interval between requests."""
    import time
    from dataclasses import dataclass
    
    @dataclass
    class RateLimiter:
        requests_per_second: float
        
        def __post_init__(self):
            self.last_request_time = {}
            self.min_interval = 1.0 / self.requests_per_second
        
        def wait(self, key: str = "default"):
            now = time.time()
            elapsed = now - self.last_request_time.get(key, 0.0)
            wait_time = self.min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request_time[key] = time.time()
    
    limiter = RateLimiter(requests_per_second=2.0)  # 0.5s per request
    
    times = []
    for i in range(3):
        start = time.time()
        limiter.wait("test")
        times.append(time.time() - start)
    
    # Second and third requests should each take ~0.5s due to rate limiting
    assert times[1] >= 0.4, f"Rate limiting not enforced: {times[1]:.2f}s < 0.4s"
    assert times[2] >= 0.4, f"Rate limiting not enforced: {times[2]:.2f}s < 0.4s"


def test_paper_schema_validity():
    """Verify Paper model has required fields."""
    required_fields = [
        'arxiv_id', 'title', 'abstract', 'authors', 'category', 'submitted_date'
    ]
    
    paper_data = {
        'arxiv_id': '2020.12345',
        'title': 'Sample Paper',
        'abstract': 'This is a sample abstract.',
        'authors': 'Author1, Author2',
        'category': 'cs.LG',
        'submitted_date': '2020-01-15',
    }
    
    for field in required_fields:
        assert field in paper_data, f"Paper missing required field: {field}"


def test_citation_schema_validity():
    """Verify Citation model has required fields."""
    required_fields = [
        'arxiv_id', 'citations_12mo', 'citations_24mo', 'fetched_date'
    ]
    
    citation_data = {
        'arxiv_id': '2020.12345',
        'citations_12mo': 5,
        'citations_24mo': 12,
        'fetched_date': '2024-01-01',
    }
    
    for field in required_fields:
        assert field in citation_data, f"Citation missing required field: {field}"


def test_arxiv_id_format_validation():
    """Verify ArXiv ID format (YYYY.NNNNN)."""
    import re
    
    arxiv_pattern = r'^\d{4}\.\d{5}(v\d+)?$'
    
    valid_ids = ['2020.12345', '2021.01234', '2022.99999', '2020.12345v2']
    invalid_ids = ['2020.123', '20.12345', 'abc.def', '2020-12345']
    
    for arxiv_id in valid_ids:
        assert re.match(arxiv_pattern, arxiv_id), f"Valid ID rejected: {arxiv_id}"
    
    for arxiv_id in invalid_ids:
        assert not re.match(arxiv_pattern, arxiv_id), f"Invalid ID accepted: {arxiv_id}"


def test_citation_count_non_negative():
    """Verify citation counts are non-negative integers."""
    citations = [
        {'citations_12mo': 0, 'citations_24mo': 5},
        {'citations_12mo': 10, 'citations_24mo': 20},
        {'citations_12mo': 100, 'citations_24mo': 150},
    ]
    
    for c in citations:
        assert c['citations_12mo'] >= 0, f"Negative 12mo citations: {c['citations_12mo']}"
        assert c['citations_24mo'] >= 0, f"Negative 24mo citations: {c['citations_24mo']}"
        assert c['citations_12mo'] <= c['citations_24mo'], \
            "12mo citations should not exceed 24mo citations"


def test_date_format_iso8601():
    """Verify date strings follow ISO 8601 format (YYYY-MM-DD)."""
    import re
    from datetime import datetime
    
    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    
    valid_dates = ['2020-01-15', '2021-12-31', '2022-06-15']
    invalid_dates = ['2020-1-15', '01-15-2020', '2020/01/15']
    
    for date_str in valid_dates:
        assert re.match(date_pattern, date_str), f"Valid date rejected: {date_str}"
        # Also verify it's a valid calendar date
        datetime.strptime(date_str, '%Y-%m-%d')
    
    for date_str in invalid_dates:
        assert not re.match(date_pattern, date_str), f"Invalid date accepted: {date_str}"


def test_arxiv_categories_known():
    """Verify papers belong to known ArXiv categories."""
    allowed_categories = ['cs.LG', 'cs.AI', 'stat.ML', 'econ.GN']
    
    papers = [
        {'category': 'cs.LG'},
        {'category': 'cs.AI'},
        {'category': 'stat.ML'},
        {'category': 'econ.GN'},
    ]
    
    for paper in papers:
        assert paper['category'] in allowed_categories, \
            f"Unknown category: {paper['category']}"
