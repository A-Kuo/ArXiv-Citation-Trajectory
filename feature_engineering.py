#!/usr/bin/env python3
"""Feature engineering pipeline for ArXiv papers."""

import sqlite3
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "citations.db"
OUTPUT_DIR = Path(".")


class FeatureEngineer:
    """Pipeline for feature engineering from ArXiv papers."""

    def __init__(self, db_path: str = DB_PATH, cutoff_date: Optional[datetime] = None):
        """Initialize feature engineer.

        Args:
            db_path: Path to SQLite database
            cutoff_date: Only include papers submitted before this date to ensure
                        they have at least 24 months of citation history.
                        Default: 24 months before today.
        """
        self.db_path = db_path
        self.cutoff_date = cutoff_date or (datetime.now() - timedelta(days=365*2))
        self.filtering_log = []
        self.feature_log = []
        self.target_log = []

    def log_filter(self, stage: str, reason: str, count: int, total: int):
        """Log filtering decisions."""
        pct = (count / total * 100) if total > 0 else 0
        msg = f"{stage}: {reason} - Filtered {count} papers ({pct:.1f}% of {total})"
        logger.info(msg)
        self.filtering_log.append(msg)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load papers and citations from database."""
        conn = sqlite3.connect(self.db_path)
        papers = pd.read_sql_query("SELECT * FROM papers", conn)
        citations = pd.read_sql_query("SELECT * FROM citations", conn)
        conn.close()

        logger.info(f"Loaded {len(papers)} papers and {len(citations)} citations")
        return papers, citations

    def merge_data(self, papers: pd.DataFrame, citations: pd.DataFrame) -> pd.DataFrame:
        """Merge papers and citations."""
        df = papers.merge(citations, on="arxiv_id", how="inner")
        logger.info(f"After join: {len(df)} papers with citation data")
        return df

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering decisions with documentation."""
        initial_count = len(df)
        self.filtering_log.append(f"\n### FILTERING DECISIONS ###\n")
        self.filtering_log.append(f"Initial dataset: {initial_count} papers\n")

        df["submitted_date"] = pd.to_datetime(df["submitted_date"])

        before_time_filter = len(df)
        df = df[df["submitted_date"] <= self.cutoff_date]
        self.log_filter(
            "Time-based cutoff",
            f"Papers submitted after {self.cutoff_date.date()} (need 24mo citation history)",
            before_time_filter - len(df),
            initial_count,
        )

        before_null_filter = len(df)
        df = df[df["citations_24mo"].notna()]
        self.log_filter(
            "Null citations",
            "Papers with missing citations_24mo (excluded - can't measure target)",
            before_null_filter - len(df),
            initial_count,
        )

        before_text_filter = len(df)
        df = df[(df["abstract"].notna()) & (df["title"].notna())]
        df = df[(df["abstract"].str.len() > 10) & (df["title"].str.len() > 3)]
        self.log_filter(
            "Invalid text",
            "Papers with missing/empty abstract or title",
            before_text_filter - len(df),
            initial_count,
        )

        self.filtering_log.append(f"\nFinal dataset: {len(df)} papers")
        self.filtering_log.append(f"Retention rate: {len(df) / initial_count * 100:.1f}%\n")

        logger.info(f"After filtering: {len(df)} papers")
        return df.reset_index(drop=True)

    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text-based features."""
        logger.info("Extracting text features...")
        self.feature_log.append("\n### TEXT FEATURES ###\n")

        features = pd.DataFrame(index=df.index)

        features["abstract_length"] = df["abstract"].str.split().str.len()
        self.feature_log.append(f"✓ abstract_length: word count (range: {features['abstract_length'].min()}-{features['abstract_length'].max()})\n")

        features["title_length"] = df["title"].str.split().str.len()
        self.feature_log.append(f"✓ title_length: word count (range: {features['title_length'].min()}-{features['title_length'].max()})\n")

        try:
            features["flesch_reading_ease"] = df["abstract"].apply(
                lambda x: textstat.flesch_reading_ease(x) if isinstance(x, str) else np.nan
            )
            self.feature_log.append(f"✓ flesch_reading_ease: readability score (range: {features['flesch_reading_ease'].min():.1f}-{features['flesch_reading_ease'].max():.1f})\n")
        except Exception as e:
            logger.warning(f"Error computing Flesch score: {e}")
            features["flesch_reading_ease"] = np.nan

        features["equation_count"] = df["abstract"].str.count(r'\$').fillna(0).astype(int)
        self.feature_log.append(f"✓ equation_count: $ symbols in abstract (max: {features['equation_count'].max()})\n")

        features["citation_count"] = df["abstract"].str.count(r'\[\d+\]').fillna(0).astype(int)
        self.feature_log.append(f"✓ citation_count: [digit] patterns in abstract (max: {features['citation_count'].max()})\n")

        combined_text = df["abstract"] + " " + df["title"]

        n_docs = len(combined_text)
        min_df_value = max(1, min(5, max(1, n_docs // 10)))

        tfidf = TfidfVectorizer(
            max_features=500,
            min_df=min_df_value,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
            max_df=1.0,
        )

        try:
            tfidf_array = tfidf.fit_transform(combined_text)
            feature_names = tfidf.get_feature_names_out()
            if len(feature_names) > 0:
                tfidf_df = pd.DataFrame(
                    tfidf_array.toarray(),
                    columns=[f"tfidf_{name}" for name in feature_names],
                    index=df.index,
                )
                features = pd.concat([features, tfidf_df], axis=1)
                self.feature_log.append(f"✓ TF-IDF: {len(feature_names)} features (unigrams + bigrams, min_df={min_df_value}, max_features=500)\n")
            else:
                self.feature_log.append(f"⚠ TF-IDF: No features generated (dataset too small)\n")
        except Exception as e:
            logger.warning(f"Error computing TF-IDF: {e}")
            self.feature_log.append(f"⚠ TF-IDF: Not computed - {e}\n")

        logger.info(f"Extracted {features.shape[1]} text features")
        return features

    def extract_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract metadata-based features."""
        logger.info("Extracting metadata features...")
        self.feature_log.append("\n### METADATA FEATURES ###\n")

        features = pd.DataFrame(index=df.index)

        features["author_count"] = df["authors"].str.split(",").str.len()
        self.feature_log.append(f"✓ author_count: number of authors (range: {features['author_count'].min()}-{features['author_count'].max()})\n")

        features["submission_month"] = df["submitted_date"].dt.month
        features["submission_year"] = df["submitted_date"].dt.year
        self.feature_log.append(f"✓ submission_month: month (1-12) for seasonality\n")
        self.feature_log.append(f"✓ submission_year: year for temporal trends\n")

        categories = df["category"].unique()
        for cat in sorted(categories):
            features[f"cat_{cat}"] = (df["category"] == cat).astype(int)
        self.feature_log.append(f"✓ Category one-hot encoding: {len(categories)} categories\n")

        keywords = ["survey", "benchmark", "state-of-the-art", "novel", "we propose"]
        combined_text = (df["title"] + " " + df["abstract"]).str.lower()

        for keyword in keywords:
            features[f"has_{keyword.replace('-', '_').replace(' ', '_')}"] = (
                combined_text.str.contains(re.escape(keyword), regex=True)
            ).astype(int)
        self.feature_log.append(f"✓ Keyword signals: {len(keywords)} high-signal words detected\n")

        logger.info(f"Extracted {features.shape[1]} metadata features")
        return features

    def construct_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct target variables."""
        logger.info("Constructing target variables...")
        self.target_log.append("\n### TARGET VARIABLES ###\n")

        targets = pd.DataFrame(index=df.index)

        targets["citations_24mo"] = df["citations_24mo"].astype(float)
        self.target_log.append(f"✓ citations_24mo: raw citation count (range: {targets['citations_24mo'].min():.0f}-{targets['citations_24mo'].max():.0f})\n")

        targets["citations_24mo_log"] = np.log1p(df["citations_24mo"])
        self.target_log.append(f"✓ citations_24mo_log: log1p transform to handle skew\n")

        p75 = df["citations_24mo"].quantile(0.75)
        targets["top_quartile"] = (df["citations_24mo"] > p75).astype(int)
        self.target_log.append(f"✓ top_quartile: binary (citations_24mo > {p75:.0f}), {targets['top_quartile'].sum()} positives ({targets['top_quartile'].sum()/len(targets)*100:.1f}%)\n")

        self.target_log.append(f"\nTarget distribution:\n")
        self.target_log.append(f"  Mean citations_24mo: {targets['citations_24mo'].mean():.2f}\n")
        self.target_log.append(f"  Median citations_24mo: {targets['citations_24mo'].median():.2f}\n")
        self.target_log.append(f"  Std citations_24mo: {targets['citations_24mo'].std():.2f}\n")
        self.target_log.append(f"  P75 (quartile threshold): {p75:.0f}\n")
        self.target_log.append(f"  P90: {df['citations_24mo'].quantile(0.90):.2f}\n")
        self.target_log.append(f"  P99: {df['citations_24mo'].quantile(0.99):.2f}\n")

        logger.info(f"Constructed {targets.shape[1]} target variables")
        return targets

    def build_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Build complete feature matrix with text and metadata features."""
        text_features = self.extract_text_features(df)
        metadata_features = self.extract_metadata_features(df)

        features = pd.concat([text_features, metadata_features], axis=1)
        logger.info(f"Complete feature matrix: {features.shape[0]} samples × {features.shape[1]} features")

        targets = self.construct_targets(df)

        arxiv_ids = pd.DataFrame({"arxiv_id": df["arxiv_id"].values}, index=df.index)

        return features, targets, arxiv_ids

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline")

        papers, citations = self.load_data()
        df = self.merge_data(papers, citations)
        df = self.filter_data(df)

        features, targets, arxiv_ids = self.build_feature_matrix(df)

        logger.info("Feature engineering completed")
        return features, targets, arxiv_ids

    def generate_report(self) -> str:
        """Generate comprehensive documentation report."""
        report = []
        report.append("# FEATURE ENGINEERING REPORT\n")
        report.append(f"Generated: {datetime.now().isoformat()}\n")

        report.extend(self.filtering_log)
        report.extend(self.feature_log)
        report.extend(self.target_log)

        report.append("\n### FEATURE MATRIX DETAILS ###\n")
        report.append("The feature matrix is saved as `features.parquet` with the following feature groups:\n")
        report.append("1. **Text Features** (5 numeric + 500 TF-IDF):\n")
        report.append("   - abstract_length, title_length, flesch_reading_ease, equation_count, citation_count\n")
        report.append("   - tfidf_* (500 features from abstract+title, unigrams and bigrams)\n\n")
        report.append("2. **Metadata Features** (~15-20 depending on data):\n")
        report.append("   - author_count, submission_month, submission_year\n")
        report.append("   - cat_* (one-hot encoded categories)\n")
        report.append("   - has_* (keyword presence indicators)\n\n")

        report.append("### TARGET VARIABLES ###\n")
        report.append("The targets are saved as `targets.parquet`:\n")
        report.append("1. **citations_24mo**: Raw citation count for regression\n")
        report.append("2. **citations_24mo_log**: Log-transformed for improved distribution\n")
        report.append("3. **top_quartile**: Binary classification target (1 if > p75)\n\n")

        report.append("### FILTERING RATIONALE ###\n")
        report.append("**Time-based cutoff (24 months before today):**\n")
        report.append("Papers submitted less than 24 months ago would not have complete citation history.\n")
        report.append("This ensures all papers have had equal opportunity to accumulate citations.\n\n")

        report.append("**Null citations handling:**\n")
        report.append("Papers with missing citations_24mo are excluded because:\n")
        report.append("- Cannot compute the target variable\n")
        report.append("- OpenAlex may not have indexed the paper\n")
        report.append("- Would introduce systematic bias if imputed\n\n")

        report.append("**Zero-citation papers:**\n")
        report.append("INCLUDED in the dataset because:\n")
        report.append("- Represent a valid outcome (legitimate low-impact papers)\n")
        report.append("- Important for models to learn what characteristics lead to low citations\n")
        report.append("- Removing them would bias toward high-impact papers\n")
        report.append("- Log transformation (log1p) handles zero values gracefully\n\n")

        report.append("**Invalid text:**\n")
        report.append("Papers with missing/empty abstracts or titles are excluded:\n")
        report.append("- Cannot extract text features\n")
        report.append("- TF-IDF requires meaningful text\n")
        report.append("- Rare occurrence in practice\n\n")

        report.append("### USAGE ###\n")
        report.append("```python\n")
        report.append("import pandas as pd\n")
        report.append("features = pd.read_parquet('features.parquet')\n")
        report.append("targets = pd.read_parquet('targets.parquet')\n")
        report.append("arxiv_ids = pd.read_parquet('arxiv_ids.parquet')\n")
        report.append("\n# For regression: use targets['citations_24mo_log']\n")
        report.append("# For classification: use targets['top_quartile']\n")
        report.append("```\n")

        return "".join(report)


def main():
    """Run feature engineering pipeline."""
    engineer = FeatureEngineer()

    features, targets, arxiv_ids = engineer.run()

    output_dir = Path(".")
    features.to_parquet(output_dir / "features.parquet", index=False)
    targets.to_parquet(output_dir / "targets.parquet", index=False)
    arxiv_ids.to_parquet(output_dir / "arxiv_ids.parquet", index=False)

    logger.info(f"Features saved: features.parquet ({features.shape[0]} × {features.shape[1]})")
    logger.info(f"Targets saved: targets.parquet ({targets.shape[0]} × {targets.shape[1]})")
    logger.info(f"ArXiv IDs saved: arxiv_ids.parquet ({arxiv_ids.shape[0]} × {arxiv_ids.shape[1]})")

    report = engineer.generate_report()
    report_path = output_dir / "FEATURE_ENGINEERING_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Report saved: {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
