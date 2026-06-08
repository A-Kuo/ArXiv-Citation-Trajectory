# ArXiv Citation Trajectory Data Pipeline

A comprehensive data pipeline to fetch papers from ArXiv and their citation counts from OpenAlex, with built-in rate limiting, resume capability, and data quality reporting.

## Features

- **ArXiv API Integration**: Queries papers in cs.LG, cs.AI, econ.GN, and stat.ML categories (2019-2022)
- **OpenAlex API Integration**: Fetches citation counts at 12 and 24 months post-submission
- **SQLite Database**: Persistent storage with two tables: papers and citations
- **Rate Limiting**: 1 req/sec for ArXiv, 10 req/sec for OpenAlex
- **Resume Capability**: Skips papers already in database, continues from where it left off
- **Data Quality Report**: Summary statistics on null rates, citation distribution, category coverage, date ranges

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:

```bash
python pipeline.py
```

The pipeline will:
1. Fetch ~5,000 papers from ArXiv (2019-2022)
2. Query OpenAlex for citation counts
3. Store all data in `citations.db`
4. Generate and print a quality report
5. Save the report to `quality_report.txt`

## Database Schema

### papers table
- `arxiv_id` (TEXT, PRIMARY KEY): ArXiv identifier
- `title` (TEXT): Paper title
- `abstract` (TEXT): Paper abstract
- `authors` (TEXT): Comma-separated authors
- `category` (TEXT): ArXiv category
- `submitted_date` (TEXT): Submission date (YYYY-MM-DD)

### citations table
- `arxiv_id` (TEXT, PRIMARY KEY): ArXiv identifier
- `citations_12mo` (INTEGER): Citation count within 12 months
- `citations_24mo` (INTEGER): Citation count within 24 months
- `fetched_date` (TEXT): When the citation data was fetched

## Resume Capability

The pipeline automatically detects papers already in the database and skips them. This allows:
- Resuming interrupted runs without re-fetching
- Incrementally adding new data
- Running multiple pipeline instances safely

## Rate Limiting

To avoid overloading the APIs:
- **ArXiv**: 1 request per second (as per API guidelines)
- **OpenAlex**: 10 requests per second (API allows higher but we're conservative)

The RateLimiter class ensures smooth request distribution.

## Output

The pipeline generates:
- `citations.db`: SQLite database with all papers and citations
- `quality_report.txt`: Data quality statistics
- Console output with progress logs

## Quality Report Contents

- Total paper count and citation coverage
- Null rates for key fields
- Citation distribution (mean, median, P90, P99)
- Papers per category breakdown
- Date coverage (earliest to latest submission)

## Configuration

To modify the pipeline, edit these variables in `pipeline.py`:
- `DB_PATH`: Database file location (default: `citations.db`)
- `STATE_FILE`: Pipeline state file (default: `pipeline_state.json`)
- `categories`: ArXiv categories to fetch
- `start_date` / `end_date`: Date range for paper selection
- `target_count`: Target number of papers to fetch

## Performance Notes

- Full pipeline run (5,000 papers) takes approximately:
  - ArXiv fetch: ~30 minutes (rate-limited to 1 req/sec)
  - OpenAlex fetch: ~8 minutes (rate-limited to 10 req/sec)
  - Database operations: ~1 minute
  - Total: ~40 minutes

## Error Handling

- Network errors are logged and skipped (pipeline continues)
- Duplicate papers are silently skipped
- Missing OpenAlex data is handled gracefully
- All errors are logged for debugging
