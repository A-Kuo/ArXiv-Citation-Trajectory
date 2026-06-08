#!/usr/bin/env python3
"""Quick test of pipeline functionality with small dataset."""

import sqlite3
import tempfile
import os
from datetime import datetime

import pipeline

# Test with temporary database
test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db').name
original_db = pipeline.DB_PATH
pipeline.DB_PATH = test_db

try:
    print("Testing database initialization...")
    pipeline.init_database()

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "papers" in tables, "papers table not created"
    assert "citations" in tables, "citations table not created"
    print("✓ Database initialization successful")

    print("\nTesting rate limiter...")
    limiter = pipeline.RateLimiter(10.0)
    import time
    start = time.time()
    for i in range(3):
        limiter.wait("test")
    elapsed = time.time() - start
    assert elapsed >= 0.2, f"Rate limiting not working: {elapsed}s"
    print(f"✓ Rate limiter working (3 requests took {elapsed:.2f}s)")

    print("\nTesting data storage...")
    test_papers = [
        {
            "arxiv_id": "2301.00001",
            "title": "Test Paper 1",
            "abstract": "This is a test abstract",
            "authors": "John Doe, Jane Smith",
            "category": "cs.LG",
            "submitted_date": "2023-01-01"
        },
        {
            "arxiv_id": "2301.00002",
            "title": "Test Paper 2",
            "abstract": "Another test abstract",
            "authors": "Alice Bob",
            "category": "cs.AI",
            "submitted_date": "2023-01-02"
        }
    ]

    pipeline.store_papers(test_papers)

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 2, f"Expected 2 papers, got {count}"
    print(f"✓ Paper storage working ({count} papers stored)")

    print("\nTesting existing ID detection...")
    existing = pipeline.get_existing_arxiv_ids()
    assert "2301.00001" in existing, "Existing paper not detected"
    assert "2301.00002" in existing, "Existing paper not detected"
    print(f"✓ Existing ID detection working ({len(existing)} papers found)")

    print("\nTesting duplicate prevention...")
    pipeline.store_papers(test_papers)  # Store same papers again

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 2, f"Duplicates were stored: {count} papers"
    print(f"✓ Duplicate prevention working (still {count} papers)")

    print("\n" + "="*50)
    print("All tests passed! Pipeline is functional.")
    print("="*50)

finally:
    pipeline.DB_PATH = original_db
    if os.path.exists(test_db):
        os.remove(test_db)
