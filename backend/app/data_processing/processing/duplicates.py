"""Duplicate Record Handling for GlassBox-BI.

Detects exact duplicate rows and business-key collisions (date + entity_id + product_id),
identifies conflicting values, and applies deterministic resolution policies with
comprehensive audit reporting.
"""

from typing import List, Tuple
import pandas as pd

from backend.app.schemas.data_contract import ColumnMapping
from backend.app.schemas.processing import DataProcessingConfig, DuplicateStrategy
from backend.app.data_processing.processing.audit import AuditTrailTracker


class DuplicateHandler:
    """Detects and deterministically resolves duplicate business records."""

    def resolve_duplicates(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        config: DataProcessingConfig,
        audit: AuditTrailTracker,
    ) -> Tuple[pd.DataFrame, int, int, int]:
        """Resolves duplicate rows and business key collisions.

        Args:
            df: Cleaned dataframe.
            mapping: Column mapping.
            config: Processing configuration.
            audit: Audit trail tracker.

        Returns:
            Tuple of (resolved_df, duplicates_before, duplicates_removed, conflict_count).
        """
        # 1. Determine Dynamic Business Key
        business_key: List[str] = [mapping.date, mapping.entity_id]
        if mapping.product_id and mapping.product_id in df.columns:
            business_key.append(mapping.product_id)

        # Ensure all key components exist in dataframe
        active_key = [k for k in business_key if k in df.columns]

        # 2. Count Exact Duplicates (all columns identical)
        exact_dups_mask = df.duplicated(keep="first")
        exact_dups_count = int(exact_dups_mask.sum())

        # 3. Count Business Key Duplicates
        key_dups_mask = df.duplicated(subset=active_key, keep=False)
        total_key_dups = int(key_dups_mask.sum())

        # 4. Check for Conflicting Duplicates (same key, but target or features differ)
        conflict_count = 0
        if total_key_dups > 0 and mapping.target in df.columns:
            # Group by key and inspect target variance/uniqueness
            dup_sub = df[key_dups_mask]
            conflicts = (
                dup_sub.groupby(active_key)[mapping.target]
                .nunique()
                .reset_index()
            )
            conflict_count = int((conflicts[mapping.target] > 1).sum())

        duplicates_before = int(df.duplicated(subset=active_key, keep="first").sum())
        resolved_df = df.copy()
        removed_count = 0

        # 5. Apply Strategy
        if duplicates_before > 0:
            if config.duplicate_strategy == DuplicateStrategy.KEEP_FIRST:
                resolved_df = df.drop_duplicates(subset=active_key, keep="first").copy()
                removed_count = len(df) - len(resolved_df)
            elif config.duplicate_strategy == DuplicateStrategy.KEEP_LAST:
                resolved_df = df.drop_duplicates(subset=active_key, keep="last").copy()
                removed_count = len(df) - len(resolved_df)
            elif config.duplicate_strategy == DuplicateStrategy.FLAG:
                resolved_df["is_duplicate_key"] = key_dups_mask.astype(int)
                removed_count = 0

            audit.record(
                operation="duplicate_resolution",
                reason="Resolve multiple records sharing the same primary business key",
                strategy=config.duplicate_strategy.value,
                rows_affected=removed_count if removed_count > 0 else duplicates_before,
                before_summary={
                    "business_key": active_key,
                    "exact_duplicates": exact_dups_count,
                    "key_duplicates": duplicates_before,
                    "conflicting_groups": conflict_count,
                },
                after_summary={
                    "retained_rows": len(resolved_df),
                    "removed_rows": removed_count,
                },
            )
        else:
            audit.record(
                operation="duplicate_inspection",
                reason="Verification of dataset uniqueness against dynamic business key",
                strategy="exact_and_key_matching",
                rows_affected=0,
                before_summary={"business_key": active_key, "duplicates_found": 0},
                after_summary={"status": "CLEAN"},
            )

        return resolved_df, duplicates_before, removed_count, conflict_count
