"""Processing Audit Trail Tracker for GlassBox-BI.

Maintains an explainable, step-by-step audit log of every transformation
performed by the Data Processing Agent, enabling complete transparency and
downstream explainability.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from backend.app.schemas.processing import ProcessingAuditEntry


class AuditTrailTracker:
    """Accumulates and structures transparent transformation audit entries."""

    def __init__(self) -> None:
        self._entries: List[ProcessingAuditEntry] = []

    def record(
        self,
        operation: str,
        reason: str,
        strategy: str,
        column: Optional[str] = None,
        rows_affected: int = 0,
        before_summary: Optional[Dict[str, Any]] = None,
        after_summary: Optional[Dict[str, Any]] = None,
    ) -> ProcessingAuditEntry:
        """Records a new transformation event into the audit trail."""
        entry = ProcessingAuditEntry(
            timestamp=datetime.now(timezone.utc),
            operation=operation,
            column=column,
            rows_affected=rows_affected,
            reason=reason,
            strategy=strategy,
            before_summary=before_summary,
            after_summary=after_summary,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> List[ProcessingAuditEntry]:
        """Returns a shallow copy of recorded audit entries."""
        return list(self._entries)

    @property
    def total_operations(self) -> int:
        """Returns the number of recorded audit entries."""
        return len(self._entries)

    @property
    def total_rows_affected(self) -> int:
        """Computes aggregate row count affected across all operations."""
        return sum(e.rows_affected for e in self._entries)

    def clear(self) -> None:
        """Resets the audit trail."""
        self._entries.clear()
