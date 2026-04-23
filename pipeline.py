#!/usr/bin/env python3
"""Data pipeline for ArXiv papers and citation counts.

Database backend: PostgreSQL (primary), MySQL, or SQL Server via SQLAlchemy.
Set the DATABASE_URL environment variable to switch backends:

    PostgreSQL:  postgresql+psycopg2://user:pass@host:5432/citations
    MySQL:       mysql+pymysql://user:pass@host:3306/citations
    SQL Server:  mssql+pyodbc://user:pass@host/citations?driver=ODBC+Driver+17+for+SQL+Server
    SQLite:      sqlite:///citations.db   (demo/local fallback only)

If DATABASE_URL is not set, falls back to SQLite for local testing.
"""

import json
import logging
import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, String, Text,
    create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------

_DEFAULT_DB_URL = "sqlite:///citations.db"
STATE_FILE = "pipeline_state.json"


def get_engine(db_url: Optional[str] = None):
    """Create SQLAlchemy engine from DATABASE_URL env var or explicit url."""
    url = db_url or os.environ.get("DATABASE_URL", _DEFAULT_DB_URL)
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(url, connect_args=connect_args, echo=False)
    logger.info(f"Database engine: {engine.dialect.name} @ {engine.url.database}")
    return engine


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"
    arxiv_id = Column(String(64), primary_key=True)
    title = Column(Text)
    abstract = Column(Text)
    authors = Column(Text)
    category = Column(String(32))
    submitted_date = Column(String(16))
    citations = relationship("Citation", back_populates="paper", uselist=False)


class Citation(Base):
    __tablename__ = "citations"
    arxiv_id = Column(String(64), ForeignKey("papers.arxiv_id"), primary_key=True)
    citations_12mo = Column(Integer)
    citations_24mo = Column(Integer)
    fetched_date = Column(String(16))
    paper = relationship("Paper", back_populates="citations")


def init_database(engine=None):
    """Create tables if they don't exist."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database schema initialised")
    return engine


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

@dataclass
class RateLimiter:
    """Per-key rate limiter tracking last request time."""
    requests_per_second: float

    def __post_init__(self):
        self.last_request_time: Dict[str, float] = {}
        self.min_interval = 1.0 / self.requests_per_second

    def wait(self, key: str = "default"):
        now = time.time()
        elapsed = now - self.last_request_time.get(key, 0.0)
        wait_time = self.min_interval - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time[key] = time.time()


# ---------------------------------------------------------------------------
# Pipeline state (resume support)
# ---------------------------------------------------------------------------

def load_state() -> Dict[str, Any]:
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"papers_fetched": 0, "citations_fetched": 0}


def save_state(state: Dict[str, Any]):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def get_existing_arxiv_ids(engine) -> set:
    with Session(engine) as session:
        rows = session.execute(text("SELECT arxiv_id FROM papers")).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# ArXiv fetching
# ---------------------------------------------------------------------------

def fetch_arxiv_papers(
    categories: List[str],
    start_date: str,
    end_date: str,
    target_count: int = 5000,
    rate_limiter: Optional[RateLimiter] = None,
    engine=None,
) -> List[Dict[str, Any]]:
    """Fetch papers from ArXiv API. Skips papers already in the database."""
    if rate_limiter is None:
        rate_limiter = RateLimiter(1.0)
    if engine is None:
        engine = get_engine()

    existing_ids = get_existing_arxiv_ids(engine)
    papers: List[Dict[str, Any]] = []
    base_url = "https://export.arxiv.org/api/query?"
    headers = {"User-Agent": "ArXiv-Citation-Trajectory/1.0"}

    for category in categories:
        start_index = 0
        max_results = 100

        while len(papers) < target_count:
            params = {
                "search_query": (
                    f"cat:{category} AND submittedDate:"
                    f"[{start_date.replace('-','')}000000 TO {end_date.replace('-','')}235959]"
                ),
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
                logger.warning(f"ArXiv fetch error ({category}, offset {start_index}): {e}")
                break

            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                logger.warning(f"ArXiv XML parse error: {e}")
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
                arxiv_id = entry_id.text.split("/abs/")[-1]
                if arxiv_id in existing_ids:
                    continue

                title = getattr(entry.find("atom:title", ns), "text", "") or ""
                abstract = getattr(entry.find("atom:summary", ns), "text", "") or ""
                authors = ", ".join(
                    getattr(a.find("atom:name", ns), "text", "")
                    for a in entry.findall("atom:author", ns)
                )
                submitted_date = (getattr(entry.find("atom:published", ns), "text", "") or "")[:10]

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "category": category,
                    "submitted_date": submitted_date,
                })

            start_index += max_results
            logger.info(f"  [{category}] fetched {len(papers)} papers so far...")

    logger.info(f"Total papers fetched from ArXiv: {len(papers)}")
    return papers


def store_papers(papers: List[Dict[str, Any]], engine=None):
    """Upsert papers into the database."""
    if engine is None:
        engine = get_engine()
    stored = 0
    with Session(engine) as session:
        for p in papers:
            existing = session.get(Paper, p["arxiv_id"])
            if existing is None:
                session.add(Paper(**p))
                stored += 1
        session.commit()
    logger.info(f"Stored {stored} new papers ({len(papers) - stored} already existed)")


# ---------------------------------------------------------------------------
# OpenAlex citation fetching
# ---------------------------------------------------------------------------

def fetch_openalex_citations(
    arxiv_id: str,
    submitted_date: str,
    rate_limiter: Optional[RateLimiter] = None,
) -> Optional[Dict[str, int]]:
    if rate_limiter is None:
        rate_limiter = RateLimiter(10.0)

    rate_limiter.wait("openalex")
    try:
        response = requests.get(
            "https://api.openalex.org/works",
            params={"filter": f"arxiv:{arxiv_id}", "per_page": 1},
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None

        work = results[0]
        submission_dt = datetime.strptime(submitted_date, "%Y-%m-%d")
        return {
            "citations_12mo": _count_citations_within(work, submission_dt, months=12),
            "citations_24mo": _count_citations_within(work, submission_dt, months=24),
        }
    except requests.RequestException as e:
        logger.warning(f"OpenAlex fetch error for {arxiv_id}: {e}")
        return None


def _count_citations_within(work: Dict, submission_dt: datetime, months: int) -> int:
    cited_by_count = work.get("cited_by_count", 0)
    if not cited_by_count:
        return 0
    cutoff_dt = submission_dt + timedelta(days=30 * months)
    pub_str = work.get("publication_date") or submission_dt.isoformat()[:10]
    try:
        pub_dt = datetime.strptime(pub_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        pub_dt = submission_dt
    return 0 if pub_dt > cutoff_dt else cited_by_count


def fetch_all_citations(engine=None, rate_limiter: Optional[RateLimiter] = None):
    """Fetch and store citation counts for all papers not yet covered."""
    if engine is None:
        engine = get_engine()
    if rate_limiter is None:
        rate_limiter = RateLimiter(10.0)

    with Session(engine) as session:
        rows = session.execute(text(
            "SELECT p.arxiv_id, p.submitted_date FROM papers p "
            "WHERE p.arxiv_id NOT IN (SELECT arxiv_id FROM citations)"
        )).fetchall()

    logger.info(f"Fetching citations for {len(rows)} papers without citation data")

    for i, (arxiv_id, submitted_date) in enumerate(rows, 1):
        result = fetch_openalex_citations(arxiv_id, submitted_date, rate_limiter)
        if result:
            with Session(engine) as session:
                existing = session.get(Citation, arxiv_id)
                if existing is None:
                    session.add(Citation(
                        arxiv_id=arxiv_id,
                        citations_12mo=result["citations_12mo"],
                        citations_24mo=result["citations_24mo"],
                        fetched_date=datetime.now().isoformat()[:10],
                    ))
                    session.commit()
        if i % 100 == 0:
            logger.info(f"  Progress: {i}/{len(rows)} papers")

    logger.info("Citation fetching completed")


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def generate_quality_report(engine=None):
    if engine is None:
        engine = get_engine()

    papers_df = pd.read_sql("SELECT * FROM papers", engine)
    citations_df = pd.read_sql("SELECT * FROM citations", engine)
    merged_df = papers_df.merge(citations_df, on="arxiv_id", how="left")

    lines = [
        "\n" + "=" * 60,
        "DATA QUALITY REPORT",
        "=" * 60,
        f"\nTotal papers: {len(papers_df)}",
        f"Papers with citation data: {len(citations_df)}",
    ]
    if len(papers_df):
        lines.append(f"Coverage: {len(citations_df) / len(papers_df) * 100:.1f}%")

    lines.append("\nNULL RATES:")
    for col in ["title", "abstract", "authors", "citations_12mo", "citations_24mo"]:
        df = papers_df if col in papers_df.columns else merged_df
        null_pct = df[col].isna().sum() / len(df) * 100
        lines.append(f"  {col}: {null_pct:.2f}%")

    for label, col in [("12-month", "citations_12mo"), ("24-month", "citations_24mo")]:
        data = citations_df[col].dropna()
        if len(data):
            lines.extend([
                f"\n{label.upper()} CITATION DISTRIBUTION (n={len(data)}):",
                f"  Mean:   {data.mean():.2f}",
                f"  Median: {data.median():.2f}",
                f"  Std:    {data.std():.2f}",
                f"  P25:    {data.quantile(0.25):.2f}",
                f"  P75:    {data.quantile(0.75):.2f}",
                f"  P90:    {data.quantile(0.90):.2f}",
                f"  P99:    {data.quantile(0.99):.2f}",
            ])

    lines.append("\nPAPERS PER CATEGORY:")
    for cat, n in papers_df["category"].value_counts().sort_index().items():
        lines.append(f"  {cat}: {n}")

    lines.append("\nDATE COVERAGE:")
    lines.append(f"  Earliest: {papers_df['submitted_date'].min()}")
    lines.append(f"  Latest:   {papers_df['submitted_date'].max()}")
    lines.append("\n" + "=" * 60 + "\n")

    report_text = "\n".join(lines)
    print(report_text)
    Path("quality_report.txt").write_text(report_text)
    logger.info("Quality report saved to quality_report.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    engine = init_database()

    arxiv_limiter = RateLimiter(1.0)
    openalex_limiter = RateLimiter(10.0)

    papers = fetch_arxiv_papers(
        categories=["cs.LG", "cs.AI", "econ.GN", "stat.ML"],
        start_date="2019-01-01",
        end_date="2022-12-31",
        target_count=5000,
        rate_limiter=arxiv_limiter,
        engine=engine,
    )

    store_papers(papers, engine=engine)
    fetch_all_citations(engine=engine, rate_limiter=openalex_limiter)
    generate_quality_report(engine=engine)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
