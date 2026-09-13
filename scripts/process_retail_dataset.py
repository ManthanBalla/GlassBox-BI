#!/usr/bin/env python3
"""Execute GenericBusinessDataProcessor on Retail Dataset for GlassBox-BI.

Processes raw synthetic retail dataset (50,000 rows), generates calendar, lag,
rolling, and business features, outputs processed dataset to data/processed/synthetic/
(ignored by Git), and outputs a 150-row sample to data/sample/ (tracked by Git).
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backend.app.schemas.data_contract import SYNTHETIC_RETAIL_MAPPING
from backend.app.schemas.processing import DataProcessingConfig
from backend.app.data_processing.processing.processor import GenericBusinessDataProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Process synthetic retail dataset with GlassBox-BI Data Processing Agent")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/synthetic/retail_50k.csv",
        help="Path to input raw CSV dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/synthetic/retail_processed.csv",
        help="Path to output processed CSV dataset",
    )
    parser.add_argument(
        "--sample-output",
        type=str,
        default="data/sample/retail_processed_sample.csv",
        help="Path to output processed sample CSV",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=150,
        help="Number of rows for the committed sample output",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input dataset not found at {input_path}")
        print("Please run 'python scripts/generate_synthetic_retail_data.py' first.")
        return

    print(f"Loading raw dataset from {input_path}...")
    load_start = time.perf_counter()
    raw_df = pd.read_csv(input_path)
    print(f"Loaded {len(raw_df):,} rows and {len(raw_df.columns)} columns in {time.perf_counter() - load_start:.2f}s")

    processor = GenericBusinessDataProcessor()
    config = DataProcessingConfig()

    print("\nExecuting Data Processing Agent pipeline...")
    proc_start = time.perf_counter()
    processed_df, result = processor.process(
        df=raw_df,
        mapping=SYNTHETIC_RETAIL_MAPPING,
        config=config,
        dataset_id="proc_retail_50k_001",
        dataset_name="Synthetic_Retail_50K",
    )
    proc_duration = time.perf_counter() - proc_start

    print(f"\nProcessing Complete in {proc_duration:.2f} seconds!")
    print(f"Status: {result.status}")
    print(f"Rows: Input {result.input_row_count:,} -> Output {result.output_row_count:,} (Removed: {result.removed_rows_count:,})")
    print(f"Columns: Input {result.input_column_count} -> Output {result.output_column_count}")
    print(f"Generated Features ({len(result.generated_features)}): {result.generated_features}")
    print(f"Data Quality Score: Before {result.quality_score_before.overall_score:.1f}/100 -> After {result.quality_score_after.overall_score:.1f}/100")
    print(f"Outlier Summary: {result.outlier_summary.total_outliers} outliers flagged ({result.outlier_summary.outlier_percentage}%)")
    print(f"Temporal Split: Train {result.temporal_split.train_rows:,} ({result.temporal_split.train_start_date} to {result.temporal_split.train_end_date}) | Val {result.temporal_split.val_rows:,} | Test {result.temporal_split.test_rows:,}")
    print(f"Audit Trail Entries: {len(result.audit_trail)}")

    # Ensure output directories exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_output_path = Path(args.sample_output)
    sample_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full processed dataset
    print(f"\nSaving full processed dataset to {output_path}...")
    processed_df.to_csv(output_path, index=False)
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(processed_df):,} rows ({output_size_mb:.2f} MB)")

    # Save small processed sample
    print(f"Saving {args.sample_rows}-row sample to {sample_output_path}...")
    sample_df = processed_df.head(args.sample_rows)
    sample_df.to_csv(sample_output_path, index=False)
    sample_size_kb = sample_output_path.stat().st_size / 1024
    print(f"Saved sample ({sample_size_kb:.2f} KB)")


if __name__ == "__main__":
    main()
