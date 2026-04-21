#!/usr/bin/env python3
"""Demo pipeline with sample data to demonstrate functionality."""

import sqlite3
import pipeline
from datetime import datetime

sample_papers = [
    {
        "arxiv_id": "1904.01234",
        "title": "Attention is All You Need: Transformer Architecture for Machine Learning",
        "abstract": "We propose a new simple network architecture based on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "authors": "Ashish Vaswani, Noam Shazeer, Parmar Amirhossein",
        "category": "cs.LG",
        "submitted_date": "2019-01-15"
    },
    {
        "arxiv_id": "1906.05909",
        "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "abstract": "Language model pretraining has led to significant performance gains. We present a replication study of BERT pretraining.",
        "authors": "Yinhan Liu, Myle Ott, Naman Goyal",
        "category": "cs.CL",
        "submitted_date": "2019-06-26"
    },
    {
        "arxiv_id": "1909.11942",
        "title": "YOLOv3: An Incremental Improvement",
        "abstract": "We present some updates to YOLO! We made a bunch of little design changes to make it better.",
        "authors": "Joseph Redmon, Ali Farhadi",
        "category": "cs.CV",
        "submitted_date": "2019-09-09"
    },
    {
        "arxiv_id": "2001.04451",
        "title": "ViT: Vision Transformers",
        "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited.",
        "authors": "Alexei Dosovitskiy, Lucas Beyer, Koray Kavukcuoglu",
        "category": "cs.CV",
        "submitted_date": "2020-10-22"
    },
    {
        "arxiv_id": "2005.14165",
        "title": "GPT-3: Language Models are Few-Shot Learners",
        "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by scaling up language models.",
        "authors": "Tom B. Brown, Benjamin Mann, Nick Ryder",
        "category": "cs.CL",
        "submitted_date": "2020-05-28"
    }
]

sample_citations = [
    {"arxiv_id": "1904.01234", "citations_12mo": 45, "citations_24mo": 120},
    {"arxiv_id": "1906.05909", "citations_12mo": 32, "citations_24mo": 89},
    {"arxiv_id": "1909.11942", "citations_12mo": 28, "citations_24mo": 76},
    {"arxiv_id": "2001.04451", "citations_12mo": 156, "citations_24mo": 312},
    {"arxiv_id": "2005.14165", "citations_12mo": 234, "citations_24mo": 512},
]


def run_demo():
    """Run demo pipeline with sample data."""
    print("\n" + "="*60)
    print("ArXiv Citation Trajectory - Demo Pipeline")
    print("="*60)

    pipeline.init_database()

    print("\n1. Storing sample papers...")
    pipeline.store_papers(sample_papers)

    print("\n2. Storing citation data...")
    conn = sqlite3.connect(pipeline.DB_PATH)
    cursor = conn.cursor()

    for cit in sample_citations:
        cursor.execute(
            """
            INSERT INTO citations (arxiv_id, citations_12mo, citations_24mo, fetched_date)
            VALUES (?, ?, ?, ?)
            """,
            (cit["arxiv_id"], cit["citations_12mo"], cit["citations_24mo"], datetime.now().isoformat()[:10]),
        )

    conn.commit()
    conn.close()
    print(f"   Stored {len(sample_citations)} citation records")

    print("\n3. Generating quality report...\n")
    pipeline.generate_quality_report()

    print("Demo completed! Database contents:")
    conn = sqlite3.connect(pipeline.DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM papers")
    papers_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM citations")
    citations_count = cursor.fetchone()[0]

    conn.close()

    print(f"  - Papers: {papers_count}")
    print(f"  - Citation records: {citations_count}")
    print(f"\nDatabase file: {pipeline.DB_PATH}")


if __name__ == "__main__":
    run_demo()
