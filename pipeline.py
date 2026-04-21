#!/usr/bin/env python3
"""Data pipeline for ArXiv papers and citation counts."""

import sqlite3
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import urllib.parse
import xml.etree.ElementTree as ET

import requests
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "citations.db"
STATE_FILE = "pipeline_state.json"


@dataclass
class RateLimiter:
    """Simple rate limiter with per-second request tracking."""
    requests_per_second: float

    def __post_init__(self):
        self.last_request_time = {}
        self.min_interval = 1.0 / self.requests_per_second

    def wait(self, key: str = "default"):
        """Wait until we can make the next request."""
        now = time.time()
        if key not in self.last_request_time:
            self.last_request_time[key] = now
            return

        elapsed = now - self.last_request_time[key]
        wait_time = self.min_interval - elapsed
        if wait_time > 0:
            time.sleep(wait_time)

        self.last_request_time[key] = time.time()


def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            category TEXT,
            submitted_date TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS citations (
            arxiv_id TEXT PRIMARY KEY,
            citations_12mo INTEGER,
            citations_24mo INTEGER,
            fetched_date TEXT,
            FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized")


def load_state() -> Dict[str, Any]:
    """Load pipeline state from file."""
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"papers_fetched": 0, "citations_fetched": 0}


def save_state(state: Dict[str, Any]):
    """Save pipeline state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def get_existing_arxiv_ids() -> set:
    """Get set of arxiv_ids already in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT arxiv_id FROM papers")
    ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    return ids


def fetch_arxiv_papers(
    categories: List[str],
    start_date: str,
    end_date: str,
    target_count: int = 5000,
    rate_limiter: Optional[RateLimiter] = None,
) -> List[Dict[str, Any]]:
    """Fetch papers from ArXiv API."""
    if rate_limiter is None:
        rate_limiter = RateLimiter(1.0)

    existing_ids = get_existing_arxiv_ids()
    papers = []
    base_url = "https://export.arxiv.org/api/query?"

    headers = {
        "User-Agent": "ArXiv-Citation-Trajectory/1.0 (+https://github.com/a-kuo/arxiv-citation-trajectory)"
    }

    for category in categories:
        start_index = 0
        max_results = 100

        while len(papers) < target_count:
            params = {
                "search_query": f'cat:{category} AND submittedDate:[{start_date.replace("-", "")}000000 TO {end_date.replace("-", "")}235959]',
                "start": start_index,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }

            url = base_url + urllib.parse.urlencode(params)

            rate_limiter.wait("arxiv")
            try:
                response = requests.get(url, timeout=30, headers=headers)
                response.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"Error fetching ArXiv (category {category}, offset {start_index}): {e}")
                break

            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                logger.warning(f"Error parsing ArXiv XML (category {category}, offset {start_index}): {e}")
                break

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)

            if not entries:
                break

            for entry in entries:
                if len(papers) >= target_count:
                    break

                entry_id = entry.find("atom:id", ns)
                if entry_id is None:
                    continue
                arxiv_id = entry_id.text.split('/abs/')[-1]

                if arxiv_id in existing_ids:
                    logger.debug(f"Skipping existing paper: {arxiv_id}")
                    continue

                title_elem = entry.find("atom:title", ns)
                title = title_elem.text if title_elem is not None else ""

                summary_elem = entry.find("atom:summary", ns)
                summary = summary_elem.text if summary_elem is not None else ""

                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                authors_str = ", ".join(authors)

                published_elem = entry.find("atom:published", ns)
                submitted_date = published_elem.text[:10] if published_elem is not None else ""

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": summary,
                    "authors": authors_str,
                    "category": category,
                    "submitted_date": submitted_date,
                })

            start_index += max_results
            logger.info(f"Fetched {len(papers)} papers from {category} so far...")

    logger.info(f"Fetched {len(papers)} papers total from ArXiv")
    return papers


def store_papers(papers: List[Dict[str, Any]]):
    """Store papers in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for paper in papers:
        try:
            cursor.execute(
                """
                INSERT INTO papers (arxiv_id, title, abstract, authors, category, submitted_date)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    paper["arxiv_id"],
                    paper["title"],
                    paper["abstract"],
                    paper["authors"],
                    paper["category"],
                    paper["submitted_date"],
                ),
            )
        except sqlite3.IntegrityError:
            logger.debug(f"Paper {paper['arxiv_id']} already exists")

    conn.commit()
    conn.close()
    logger.info(f"Stored {len(papers)} papers in database")


def fetch_openalex_citations(
    arxiv_id: str,
    submitted_date: str,
    rate_limiter: Optional[RateLimiter] = None,
) -> Optional[Dict[str, int]]:
    """Fetch citation counts from OpenAlex API."""
    if rate_limiter is None:
        rate_limiter = RateLimiter(10.0)

    rate_limiter.wait("openalex")

    try:
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"arxiv:{arxiv_id}",
            "per_page": 1,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            logger.debug(f"No OpenAlex data for {arxiv_id}")
            return None

        work = data["results"][0]
        publication_date = work.get("publication_date")
        cited_by_count = work.get("cited_by_count", 0)

        if not publication_date:
            publication_date = submitted_date

        submission_dt = datetime.strptime(submitted_date, "%Y-%m-%d")
        citations_12mo = count_citations_until(work, submission_dt, 12)
        citations_24mo = count_citations_until(work, submission_dt, 24)

        return {
            "citations_12mo": citations_12mo,
            "citations_24mo": citations_24mo,
        }

    except requests.RequestException as e:
        logger.warning(f"Error fetching OpenAlex data for {arxiv_id}: {e}")
        return None


def count_citations_until(work: Dict, submission_dt: datetime, months: int) -> int:
    """Estimate citations within N months of submission."""
    cited_by_count = work.get("cited_by_count", 0)

    if not cited_by_count:
        return 0

    cutoff_dt = submission_dt + timedelta(days=30 * months)
    publication_date = work.get("publication_date", submission_dt.isoformat()[:10])

    if publication_date:
        try:
            pub_dt = datetime.strptime(publication_date[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            pub_dt = submission_dt
    else:
        pub_dt = submission_dt

    if pub_dt > cutoff_dt:
        return 0

    return cited_by_count


def fetch_all_citations(rate_limiter: Optional[RateLimiter] = None):
    """Fetch citations for all papers in database."""
    if rate_limiter is None:
        rate_limiter = RateLimiter(10.0)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.arxiv_id, p.submitted_date
        FROM papers p
        WHERE p.arxiv_id NOT IN (SELECT arxiv_id FROM citations)
    """)

    papers_to_fetch = cursor.fetchall()
    conn.close()

    logger.info(f"Fetching citations for {len(papers_to_fetch)} papers")

    for i, (arxiv_id, submitted_date) in enumerate(papers_to_fetch, 1):
        result = fetch_openalex_citations(arxiv_id, submitted_date, rate_limiter)

        if result:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO citations (arxiv_id, citations_12mo, citations_24mo, fetched_date)
                VALUES (?, ?, ?, ?)
                """,
                (
                    arxiv_id,
                    result["citations_12mo"],
                    result["citations_24mo"],
                    datetime.now().isoformat()[:10],
                ),
            )
            conn.commit()
            conn.close()

        if i % 100 == 0:
            logger.info(f"Fetched citations for {i}/{len(papers_to_fetch)} papers")

    logger.info("Citation fetching completed")


def generate_quality_report():
    """Generate data quality report."""
    conn = sqlite3.connect(DB_PATH)

    papers_df = pd.read_sql_query("SELECT * FROM papers", conn)
    citations_df = pd.read_sql_query("SELECT * FROM citations", conn)
    merged_df = papers_df.merge(citations_df, on="arxiv_id", how="left")

    conn.close()

    report = []
    report.append("\n" + "="*60)
    report.append("DATA QUALITY REPORT")
    report.append("="*60)

    report.append(f"\nTotal papers: {len(papers_df)}")
    report.append(f"Papers with citations data: {len(citations_df)}")
    if len(papers_df) > 0:
        report.append(f"Citation data coverage: {len(citations_df) / len(papers_df) * 100:.1f}%")

    report.append("\nNULL RATES:")
    for col in ["title", "abstract", "authors"]:
        if col in papers_df.columns:
            null_rate = papers_df[col].isna().sum() / len(papers_df) * 100
            report.append(f"  {col}: {null_rate:.2f}%")

    for col in ["citations_12mo", "citations_24mo"]:
        if col in merged_df.columns:
            null_rate = merged_df[col].isna().sum() / len(merged_df) * 100
            report.append(f"  {col}: {null_rate:.2f}%")

    report.append("\nCITATION DISTRIBUTION:")
    cit_12 = citations_df["citations_12mo"].dropna()
    cit_24 = citations_df["citations_24mo"].dropna()

    if len(cit_12) > 0:
        report.append(f"  12-month citations:")
        report.append(f"    Mean: {cit_12.mean():.2f}")
        report.append(f"    Median: {cit_12.median():.2f}")
        report.append(f"    P90: {cit_12.quantile(0.90):.2f}")
        report.append(f"    P99: {cit_12.quantile(0.99):.2f}")

    if len(cit_24) > 0:
        report.append(f"  24-month citations:")
        report.append(f"    Mean: {cit_24.mean():.2f}")
        report.append(f"    Median: {cit_24.median():.2f}")
        report.append(f"    P90: {cit_24.quantile(0.90):.2f}")
        report.append(f"    P99: {cit_24.quantile(0.99):.2f}")

    report.append("\nPAPERS PER CATEGORY:")
    if len(papers_df) > 0:
        category_counts = papers_df["category"].value_counts().sort_index()
        for category, count in category_counts.items():
            report.append(f"  {category}: {count}")

    report.append("\nDATE COVERAGE:")
    if len(papers_df) > 0:
        report.append(f"  Earliest: {papers_df['submitted_date'].min()}")
        report.append(f"  Latest: {papers_df['submitted_date'].max()}")

    report.append("\n" + "="*60 + "\n")

    report_text = "\n".join(report)
    print(report_text)

    with open("quality_report.txt", "w") as f:
        f.write(report_text)

    logger.info("Quality report saved to quality_report.txt")


def main():
    """Run the complete pipeline."""
    logger.info("Starting data pipeline")

    init_database()

    arxiv_limiter = RateLimiter(1.0)
    openalex_limiter = RateLimiter(10.0)

    categories = ["cs.LG", "cs.AI", "econ.GN", "stat.ML"]
    start_date = "2019-01-01"
    end_date = "2022-12-31"

    logger.info(f"Fetching papers from {start_date} to {end_date}")
    papers = fetch_arxiv_papers(
        categories,
        start_date,
        end_date,
        target_count=5000,
        rate_limiter=arxiv_limiter,
    )

    logger.info(f"Storing {len(papers)} papers")
    store_papers(papers)

    logger.info("Fetching citations from OpenAlex")
    fetch_all_citations(rate_limiter=openalex_limiter)

    logger.info("Generating quality report")
    generate_quality_report()

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
