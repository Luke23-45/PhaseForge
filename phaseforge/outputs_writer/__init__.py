"""PhaseForge outputs package.

Adapted from the ``csd_observer.outputs`` reference (originally from a
change-point-detection benchmark). The reference provides the *mechanisms*
— atomic append-only writes with a cross-process lock, schema-validated
result rows, environment fingerprinting, lifecycle markers, and a global
ledger — re-cast for PhaseForge's per-run layout and offline-evaluation
result schema.

Public API (re-exported below): :class:`RunWriter`, :class:`RunLedger`,
:class:`LedgerRow`, :class:`ResultRow`, :func:`validate_row`,
:func:`collect_environment`, :func:`append_result_row`,
:func:`read_result_rows`, the aggregation and statistical helpers
(``aggregate_rows``, ``bootstrap_ci``, ``paired_wilcoxon`` and their CSV
writers), and :func:`summarize_all`.
"""

from phaseforge.outputs_writer.ledger import LedgerRow, RunLedger
from phaseforge.outputs_writer.metadata import collect_environment
from phaseforge.outputs_writer.results import append_result_row, read_result_rows
from phaseforge.outputs_writer.schema import (
    OPTIONAL_METRIC_FIELDS,
    ResultRow,
    SchemaError,
    validate_row,
)
from phaseforge.outputs_writer.summarize import summarize_all
from phaseforge.outputs_writer.tables import (
    aggregate_rows,
    bootstrap_ci,
    paired_wilcoxon,
    write_aggregates_csv,
    write_bootstrap_csv,
    write_paired_wilcoxon_csv,
)
from phaseforge.outputs_writer.writer import RunWriter, parse_run_dir

__all__ = [
    "RunWriter",
    "parse_run_dir",
    "RunLedger",
    "LedgerRow",
    "ResultRow",
    "SchemaError",
    "validate_row",
    "OPTIONAL_METRIC_FIELDS",
    "collect_environment",
    "append_result_row",
    "read_result_rows",
    "aggregate_rows",
    "bootstrap_ci",
    "paired_wilcoxon",
    "write_aggregates_csv",
    "write_bootstrap_csv",
    "write_paired_wilcoxon_csv",
    "summarize_all",
]
