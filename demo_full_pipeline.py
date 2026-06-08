#!/usr/bin/env python3
"""Full pipeline demo: Phase 1 (data fetching) + Phase 2 (feature engineering)."""

import os
import sqlite3

import pipeline
import feature_engineering


def run_full_demo():
    """Run complete pipeline demo."""
    print("\n" + "="*70)
    print("ARXIV CITATION TRAJECTORY - COMPLETE PIPELINE DEMO")
    print("="*70)

    print("\n" + "="*70)
    print("PHASE 1: DATA FETCHING")
    print("="*70)

    if os.path.exists("citations.db"):
        os.remove("citations.db")

    print("\n1. Initializing database...")
    pipeline.init_database()

    print("2. Loading sample data...")
    sample_papers = [
        {
            "arxiv_id": "1904.01234",
            "title": "Attention is All You Need: Transformer Architecture",
            "abstract": "We propose a new simple network architecture based on attention mechanisms. This survey covers recent advances in deep learning [1] and [2]. Our benchmark demonstrates $significant$ improvements.",
            "authors": "Ashish Vaswani, Noam Shazeer, Parmar Amirhossein",
            "category": "cs.LG",
            "submitted_date": "2019-01-15"
        },
        {
            "arxiv_id": "1906.05909",
            "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
            "abstract": "Language model pretraining has led to significant performance gains. We present a state-of-the-art replication study of BERT pretraining [3]. This novel approach is $important$.",
            "authors": "Yinhan Liu, Myle Ott, Naman Goyal",
            "category": "cs.CL",
            "submitted_date": "2019-06-26"
        },
        {
            "arxiv_id": "2001.04451",
            "title": "Vision Transformers: An Image is Worth 16x16 Words",
            "abstract": "While the Transformer architecture has become the de-facto standard for NLP, its applications to computer vision remain limited. We demonstrate that pure transformer-based architectures can also be competitive on image classification [1] [2] [3]. This benchmark shows novel results.",
            "authors": "Alexei Dosovitskiy, Lucas Beyer, Koray Kavukcuoglu",
            "category": "cs.CV",
            "submitted_date": "2020-10-22"
        },
        {
            "arxiv_id": "2005.14165",
            "title": "Language Models are Few-Shot Learners",
            "abstract": "Recent work has demonstrated substantial gains on many NLP tasks by scaling up language models. In this survey, we explore the capabilities of large autoregressive models [1] [2] [3] [4]. We propose novel evaluation metrics and benchmarks.",
            "authors": "Tom B. Brown, Benjamin Mann, Nick Ryder",
            "category": "cs.CL",
            "submitted_date": "2020-05-28"
        },
        {
            "arxiv_id": "2103.14030",
            "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
            "abstract": "This benchmark examines transformer architectures for vision. The survey of recent deep learning $developments$ [1] shows [2] novel approaches [3]. We propose state-of-the-art results.",
            "authors": "Alexei Dosovitskiy, Lucas Beyer, Alexander Kolesnikov",
            "category": "cs.CV",
            "submitted_date": "2020-10-22"
        },
    ]

    pipeline.store_papers(sample_papers)
    print(f"   ✓ Stored {len(sample_papers)} papers")

    print("3. Adding citation data...")
    conn = sqlite3.connect("citations.db")
    cursor = conn.cursor()
    citations_data = [
        ("1904.01234", 45, 120),
        ("1906.05909", 32, 89),
        ("2001.04451", 156, 312),
        ("2005.14165", 234, 512),
        ("2103.14030", 28, 67),
    ]
    for arxiv_id, c12, c24 in citations_data:
        cursor.execute(
            "INSERT INTO citations (arxiv_id, citations_12mo, citations_24mo, fetched_date) VALUES (?, ?, ?, ?)",
            (arxiv_id, c12, c24, "2024-01-01")
        )
    conn.commit()
    conn.close()
    print(f"   ✓ Added citation data for {len(citations_data)} papers")

    print("\n" + "="*70)
    print("PHASE 2: FEATURE ENGINEERING")
    print("="*70)

    print("\n1. Loading and filtering papers...")
    engineer = feature_engineering.FeatureEngineer()
    features, targets, arxiv_ids = engineer.run()

    print(f"\n2. Feature extraction complete:")
    print(f"   ✓ {features.shape[0]} papers × {features.shape[1]} features")
    print(f"   ✓ Text features: abstract_length, title_length, flesch_reading_ease,")
    print(f"                    equation_count, citation_count, TF-IDF")
    print(f"   ✓ Metadata features: author_count, submission_month/year,")
    print(f"                        category, keyword signals")

    print(f"\n3. Target variables:")
    print(f"   ✓ citations_24mo (regression): {targets['citations_24mo'].min():.0f}-{targets['citations_24mo'].max():.0f}")
    print(f"   ✓ citations_24mo_log (skew-corrected): {targets['citations_24mo_log'].min():.2f}-{targets['citations_24mo_log'].max():.2f}")
    print(f"   ✓ top_quartile (classification): {targets['top_quartile'].sum()} positives out of {len(targets)}")

    print(f"\n4. Saving outputs:")
    features.to_parquet("features.parquet", index=False)
    targets.to_parquet("targets.parquet", index=False)
    arxiv_ids.to_parquet("arxiv_ids.parquet", index=False)
    print(f"   ✓ features.parquet ({features.shape})")
    print(f"   ✓ targets.parquet ({targets.shape})")
    print(f"   ✓ arxiv_ids.parquet ({arxiv_ids.shape})")

    print(f"\n5. Generating report...")
    report = engineer.generate_report()
    with open("FEATURE_ENGINEERING_REPORT.md", "w") as f:
        f.write(report)
    print(f"   ✓ FEATURE_ENGINEERING_REPORT.md")

    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"\nDatabase: citations.db")
    print(f"  - Papers table: {len(sample_papers)} papers")
    print(f"  - Citations table: {len(citations_data)} citation records")
    print(f"\nFeature outputs:")
    print(f"  - features.parquet: ML-ready features")
    print(f"  - targets.parquet: regression + classification targets")
    print(f"  - arxiv_ids.parquet: paper identifiers")
    print(f"  - FEATURE_ENGINEERING_REPORT.md: complete documentation")
    print(f"\nNext steps:")
    print(f"  1. Train models using features and targets")
    print(f"  2. Use arxiv_ids for result interpretation")
    print(f"  3. Reference report for filtering/feature decisions")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    run_full_demo()
